"""Deadline-bounded JSON IPC and owned compiler process-group shutdown."""
import json
import os
import selectors
import signal
import subprocess
import sys
from pathlib import Path
from collections import deque
from threading import Lock
from time import monotonic
from uuid import uuid4

TRANSPORT_VERSION = 2
_REQUEST_TIMEOUT = 300.0  # Entire request write, compile and response read.
_MAX_LOG_BYTES = 16 * 1024


class WorkerLogs:
    """Service-owned bounded diagnostic files, separate from compiler artifacts."""

    def __init__(self, cache, service_id):
        self.directory = None if cache is None else Path(cache).resolve() / "reuse-worker-logs" / service_id
        self.paths = deque()
        self.lock = Lock()

    def write(self, request_id, attempt, error, output):
        if self.directory is None:
            return None
        # Only serial compiler jobs call this. No service/registry lock is held
        # during file I/O; cleanup touches only this service's diagnostic files.
        data = (type(error).__name__ + ": " + str(error) + "\n" + output).encode(errors="replace")
        with self.lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / (request_id + "-" + str(attempt) + ".log")
            path.write_bytes(data[-_MAX_LOG_BYTES:])
            if path not in self.paths:
                self.paths.append(path)
            while len(self.paths) > 64:
                self.paths.popleft().unlink(missing_ok=True)
        return str(path)


class WorkerProtocolError(OSError):
    pass


class WorkerCompileError(ValueError):

    def __init__(self, response, log):
        self.request_id = response["request_id"]
        self.stage = response["stage"]
        self.log = log
        super().__init__(response["error"] + ": " + response["message"])


class CompileWorker:

    def __init__(self, cache=None):
        self.process = None
        self.read_fd = None
        self.state_lock = Lock()
        self.io_lock = Lock()
        self.closed = False
        self.log = b""
        if cache is None:
            from triton import knobs

            cache_knobs = getattr(knobs, "cache", None)
            cache = None if cache_knobs is None else cache_knobs.dir
        child_env = os.environ.copy()
        if cache is not None:
            child_env["TRITON_CACHE_DIR"] = str(Path(cache).resolve())
        read_fd, write_fd = os.pipe()
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", "-m", "triton.backends.ascend.reuse.worker",
                 str(write_fd)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
                pass_fds=(write_fd, ), start_new_session=True, env=child_env)
            self.read_fd = read_fd
            for fd in (read_fd, self.process.stdin.fileno(), self.process.stdout.fileno()):
                os.set_blocking(fd, False)
        except Exception:
            if self.read_fd is None:
                os.close(read_fd)
            self.close()
            raise
        finally:
            os.close(write_fd)

    def compile(self, request):
        with self.io_lock:
            if self.closed:
                raise BrokenPipeError("compile worker is closed")
            deadline = monotonic() + _REQUEST_TIMEOUT
            request_id = request.get("request_id", uuid4().hex)
            self.log = b""
            payload = json.dumps(dict(transport=TRANSPORT_VERSION, request_id=request_id,
                                      request=request)).encode() + b"\n"
            received, sent = bytearray(), 0
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE, "write")
                selector.register(self.read_fd, selectors.EVENT_READ, "reply")
                selector.register(self.process.stdout, selectors.EVENT_READ, "log")
                while True:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError("compile worker message deadline exceeded")
                    events = selector.select(remaining)
                    if not events:
                        raise TimeoutError("compile worker message deadline exceeded")
                    for key, _ in events:
                        try:
                            if key.data == "write":
                                sent += os.write(key.fd, payload[sent:sent + 65536])
                                if sent == len(payload):
                                    selector.unregister(key.fileobj)
                                continue
                            data = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if key.data == "log":
                            if data:
                                self.log = (self.log + data)[-_MAX_LOG_BYTES:]
                            else:
                                selector.unregister(key.fileobj)
                            continue
                        if not data:
                            raise BrokenPipeError("compile worker exited before a complete response")
                        received.extend(data)
                        if len(received) > 1024 * 1024:
                            raise WorkerProtocolError("compile worker response exceeds limit")
                        if b"\n" in received:
                            line, extra = received.split(b"\n", 1)
                            if extra.strip() or sent != len(payload):
                                raise WorkerProtocolError("unexpected compile worker response")
                            self._drain_log(deadline)
                            return self._response(line, request_id)

    def _drain_log(self, deadline):
        while True:
            if monotonic() >= deadline:
                raise TimeoutError("compile worker message deadline exceeded")
            try:
                data = os.read(self.process.stdout.fileno(), 65536)
            except BlockingIOError:
                return
            if not data:
                return
            self.log = (self.log + data)[-_MAX_LOG_BYTES:]

    def _response(self, line, request_id):
        try:
            response = json.loads(line)
            if (response["transport"] != TRANSPORT_VERSION or response["request_id"] != request_id
                    or response["stage"] != "compile"):
                raise ValueError("response identity differs")
            if "error" in response:
                if not isinstance(response["error"], str) or not isinstance(response["message"], str):
                    raise ValueError("invalid error envelope")
            elif not isinstance(response["variant_key"], str):
                raise ValueError("invalid compiled variant key")
        except (ValueError, KeyError, TypeError) as error:
            raise WorkerProtocolError("invalid compile worker response") from error
        if "error" in response:
            raise WorkerCompileError(response, self.log.decode(errors="replace"))
        return response["variant_key"]

    def close(self):
        # Process termination is independent of the I/O owner. Kill the whole
        # session, including compiler children, then let its EOF wake the owner.
        with self.state_lock:
            if self.closed:
                return
            self.closed = True
        if self.process is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(self.process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    continue
            with self.io_lock:
                if self.read_fd is not None:
                    os.close(self.read_fd)
                    self.read_fd = None
                self.process.stdin.close()
                self.process.stdout.close()
