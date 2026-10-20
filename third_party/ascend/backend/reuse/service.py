"""Bounded background compilation into the native Triton cache.

Completion hints only select the ordinary foreground JIT continuation. The
service never constructs, loads, or owns a business CompiledKernel.
"""
import atexit
import json
import os
from pathlib import Path
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from threading import Lock
from time import monotonic
from uuid import uuid4

from . import config
from .executor import SerialExecutor
from .context import requires_sync_compilation
from .protocol import build_request, freeze_source, serialize_request
from .snapshot import SourceNotSerializable
from .transport import CompileWorker, WorkerLogs

# Each service bookkeeping collection retains at most this many records.
_MAX_RECORDS = 64


@dataclass(frozen=True)
class Capabilities:
    compile: bool = False
    reason: str = "BACKGROUND_COMPILE_UNAVAILABLE"


def capabilities():
    if config.mode() != 2:
        return Capabilities(reason="BACKGROUND_COMPILE_DISABLED_BY_MODE")
    from triton import knobs
    from triton.runtime import driver

    if knobs.cache.manager_class is not None or knobs.cache.remote_manager_class is not None:
        return Capabilities(reason="CUSTOM_CACHE_MANAGER_UNSUPPORTED")
    target = driver.active.get_current_target()
    if not config.supports_target(target):
        return Capabilities(reason="TARGET_NOT_SUPPORTED_FOR_BACKGROUND_COMPILE")
    return Capabilities(compile=True, reason="AUTOMATIC_COMPILE_SUPPORTED")


@dataclass
class TaskRecord:
    key: str
    request_id: str
    generation: int
    state: str = "QUEUED"
    attempt: int = 0
    log_path: str | None = None
    times: dict = field(default_factory=lambda: {"QUEUED": monotonic()})


@dataclass(frozen=True)
class FailureRecord:
    classification: str
    message: str
    stage: str
    request_id: str
    count: int
    next_retry_at: float
    log_path: str | None = None


class PreparationService:

    def __init__(self, registry, environment, capability=None):
        self.registry = registry
        self.environment = json.loads(serialize_request(environment))
        self.capability = capability or capabilities()
        self.lock = Lock()
        self.tasks = {}
        self.history = deque(maxlen=_MAX_RECORDS)
        self.failures = OrderedDict()
        self.completed = OrderedDict()
        self.generation = 0
        self.service_id = uuid4().hex
        self.logs = WorkerLogs(self.environment.get("cache"), self.service_id)
        self.closed = False
        self.worker = None
        self.compiler_executor = None
        atexit.register(self.close)

    def is_pending(self, key):
        if not self.lock.acquire(blocking=False):
            return True
        try:
            return self.closed or key in self.tasks
        finally:
            self.lock.release()

    def matches_context(self, context):
        from triton import knobs

        if context != (os.getpid(), self.environment["device"]):
            return False
        cache = knobs.cache.dir
        return cache == self.environment["cache"] or str(Path(cache).resolve()) == self.environment["cache"]

    def try_submit(self, spec, context, frozen=None):
        if config.mode() != 2 or not self.capability.compile:
            self.registry.event("queue_skipped")
            return False
        if requires_sync_compilation():
            self.registry.event("background_hook_skipped")
            return False
        if "cache" in self.environment and not self.matches_context(context):
            self.registry.event("background_environment_changed")
            return False
        if self.registry.lookup_exact_ready(spec.variant_key, context) is not None:
            return False
        if not self.lock.acquire(blocking=False):
            self.registry.event("queue_busy")
            return False
        try:
            key = spec.variant_key
            if self.closed:
                return False
            failure = self.failures.get(key)
            if (key in self.tasks or key in self.completed or len(self.tasks) >= 32
                    or (failure is not None and monotonic() < failure.next_retry_at)):
                return False
            task = TaskRecord(key, uuid4().hex, self.generation)
            self.tasks[key] = task
        finally:
            self.lock.release()
        try:
            request = build_request(spec, self.environment, task.generation, service_id=self.service_id,
                                    request_id=task.request_id, frozen=frozen)
            payload = serialize_request(request)
            source = freeze_source(spec, request)
            with self.lock:
                if self.closed or self.tasks.get(key) is not task:
                    executor = None
                else:
                    task.times["SNAPSHOT_DONE"] = monotonic()
                    if self.compiler_executor is None:
                        self.compiler_executor = SerialExecutor("ascend-reuse-compile")
                    executor = self.compiler_executor
            if executor is None:
                self._finish(task, "CANCELLED")
                return False
            # The job owns frozen source and compiler data, never the original
            # JIT/BuildSpec or invocation tensors. Live invalidation uses weakrefs.
            future = executor.submit(self._prepare, task, payload, source)
            future.add_done_callback(lambda f: self._observe(task, f))
            return True
        except Exception as error:  # admission failure must not affect the completed foreground launch
            state = "SOURCE_UNAVAILABLE" if isinstance(error, SourceNotSerializable) else "SUBMIT_FAILED"
            self._finish(task, state, error)
            return False

    def _observe(self, task, future):
        if future.cancelled():
            self._finish(task, "CANCELLED")
        else:
            # Future.exception observes even system-level failures without
            # storing the exception/traceback in any service record.
            error = future.exception()
            if error is not None:
                self._finish(task, "EXECUTOR_FAILED", error)

    def _transition(self, task, state):
        with self.lock:
            if self.closed or task.generation != self.generation or self.tasks.get(task.key) is not task:
                return False
            task.state = state
            task.times[state] = monotonic()
            return True

    def _finish(self, task, state, error=None, source=None):
        with self.lock:
            if self.tasks.get(task.key) is not task:
                return
            if self.closed or task.generation != self.generation:
                state = "CANCELLED" if task.state == "QUEUED" else "STALE"
            task.state = state
            task.times[state] = monotonic()
            self.tasks.pop(task.key)
            self.history.append(
                dict(key=task.key, request_id=task.request_id, state=state, attempt=task.attempt,
                     times=dict(task.times), log_path=task.log_path))
            if error is not None and not self.closed:
                previous = self.failures.get(task.key)
                self.failures[task.key] = FailureRecord(
                    type(error).__name__,
                    str(error).splitlines()[0][:256] if str(error) else "", getattr(error, "stage", state),
                    getattr(error, "request_id", task.request_id), 1 if previous is None else previous.count + 1,
                    monotonic() + 30.0, task.log_path)  # Retry a failed variant after 30 seconds.
                self.failures.move_to_end(task.key)
                while len(self.failures) > _MAX_RECORDS:
                    self.failures.popitem(last=False)
            elif state == "CACHE_READY":
                self.failures.pop(task.key, None)
            if source is not None and state == "CACHE_READY":
                self.completed[task.key] = source
                while len(self.completed) > _MAX_RECORDS:
                    self.completed.popitem(last=False)
        self.registry.event("background_" + state.lower())

    def _get_worker(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("service is closed")
            worker = self.worker
        if worker is not None:
            return worker
        # Only the serial compiler creates workers. close may race construction;
        # the candidate is installed or terminated after the second state check.
        cache = self.environment.get("cache")
        candidate = CompileWorker() if cache is None else CompileWorker(cache=cache)
        with self.lock:
            if self.closed:
                accepted = False
            else:
                self.worker = candidate
                accepted = True
        if not accepted:
            candidate.close()
            raise RuntimeError("service closed during worker creation")
        return candidate

    def _discard_worker(self, worker):
        with self.lock:
            if self.worker is worker:
                self.worker = None
        worker.close()

    def _prepare(self, task, payload, source):
        state, error, worker = "STALE", None, None
        try:
            if not self._transition(task, "BUILDING"):
                return
            if not source.is_current():
                return
            state = "BUILD_FAILED"
            # One worker restart after an I/O failure or timeout.
            for attempt in range(2):
                request = json.loads(payload)
                task.attempt = request["attempt"] = attempt
                worker = None
                try:
                    worker = self._get_worker()
                    if worker.compile(request) != task.key:
                        raise ValueError("worker returned a different compiled variant")
                    break
                except (OSError, TimeoutError):
                    if worker is not None:
                        self._discard_worker(worker)
                    if attempt == 1 or self.closed:
                        raise
                    self.registry.event("retry")
            if not self._transition(task, "CACHE_READY"):
                return
            if not source.is_current():
                state = "STALE"
                return
            state = "CACHE_READY"
        except Exception as caught:
            error = caught
            state = "BUILD_FAILED"
            output = getattr(caught, "log", None)
            if output is None:
                output = getattr(worker, "log", b"")
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
            try:
                task.log_path = self.logs.write(task.request_id, task.attempt, caught, output)
            except Exception:
                self.registry.event("background_log_failed")
        finally:
            self._finish(task, state, error, source)

    def has_compiled(self, spec, context):
        """Select the normal JIT tail; it owns cache lookup, load and launch.

        No files or handles are handed off here. Cache eviction or corruption is
        handled by Triton's normal compile/cache path on the calling thread.
        """
        with self.lock:
            source = None if self.closed else self.completed.get(spec.variant_key)
        if source is None:
            return False
        if not self.matches_context(context) or not source.is_current():
            with self.lock:
                if self.completed.get(spec.variant_key) is source:
                    self.completed.pop(spec.variant_key)
            self.registry.event("compiled_source_stale")
            return False
        with self.lock:
            return not self.closed and self.completed.get(spec.variant_key) is source

    def diagnostics(self):
        with self.lock:
            return dict(
                active_tasks=len(self.tasks), states=[t.state for t in self.tasks.values()], history=list(self.history),
                failures={
                    k:
                    dict(classification=f.classification, message=f.message, stage=f.stage, request_id=f.request_id,
                         count=f.count, log_path=f.log_path)
                    for k, f in self.failures.items()
                }, completed_builds=len(self.completed), closed=self.closed)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.generation += 1
            worker, self.worker = self.worker, None
            executor = self.compiler_executor
            self.completed.clear()
            tasks = tuple(self.tasks.values())
        for task in tasks:
            self._finish(task, "CANCELLED")
        if worker is not None:
            worker.close()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        atexit.unregister(self.close)
