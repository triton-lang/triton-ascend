"""Single daemon consumer with explicit, cancellable shutdown.

ThreadPoolExecutor joins its workers *before* user atexit handlers. A compiler
blocked on IPC would therefore prevent PreparationService.close from running.
This executor deliberately has no interpreter-exit join; explicit close owns
process cleanup. Futures expose failures to the service completion callback;
this executor never initializes or launches business kernels.
"""
from concurrent.futures import Future
from queue import Queue, Empty
from threading import Lock, Thread, current_thread


class SerialExecutor:

    def __init__(self, thread_name_prefix):
        self.lock = Lock()
        self.closed = False
        self.queue = Queue()
        self.thread = Thread(target=self._run, name=thread_name_prefix, daemon=True)
        self.thread.start()

    def submit(self, fn, *args):
        with self.lock:
            if self.closed:
                raise RuntimeError("executor is closed")
            future = Future()
            self.queue.put((future, fn, args))
            return future

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            future, fn, args = item
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args)
            except BaseException as error:  # Future's contract; service catches only Exception
                future.set_exception(error)
            else:
                future.set_result(result)
                del result
            # Do not retain the last task's spec/Future/traceback while idle.
            del item, future, fn, args

    def shutdown(self, wait=True, cancel_futures=False):
        cancelled = []
        with self.lock:
            if not self.closed:
                self.closed = True
                if cancel_futures:
                    while True:
                        try:
                            cancelled.append(self.queue.get_nowait()[0])
                        except Empty:
                            break
                self.queue.put(None)
        # Future callbacks must not run under the executor's lock.
        for future in cancelled:
            future.cancel()
        if wait and current_thread() is not self.thread:
            self.thread.join()
