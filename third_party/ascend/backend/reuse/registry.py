from collections import Counter
from dataclasses import dataclass, field
from threading import RLock, get_ident, local

from .model import ExecutableVariant

_MAX_READY = 64
_MAX_FAMILY_READY = 8
_FAILURE_THRESHOLD = 3


@dataclass
class InitializationSlot:
    lock: object = field(default_factory=RLock)
    users: int = 0
    owner: int | None = None
    failed: bool = False


class VariantRegistry:

    def __init__(self):
        self.lock = RLock()
        self.exact = {}
        self.families = {}
        self.counts = Counter()
        self.failures = Counter()
        self.initialization_locks = {}
        self.initializing = local()

    def initialize(self, kernel, spec, load_context):
        key = (spec.variant_key, load_context)
        held = getattr(self.initializing, "keys", ())
        with self.lock:
            slot = self.initialization_locks.setdefault(key, InitializationSlot())
            if slot.owner == get_ident() or (held and slot.owner is not None):
                raise RuntimeError("nested reuse initialization cannot wait on another owner or reenter its key")
            slot.users += 1
        try:
            with slot.lock:
                with self.lock:
                    slot.owner = get_ident()
                self.initializing.keys = (*held, key)
                try:
                    if slot.failed:
                        raise RuntimeError("concurrent reuse initialization failed; cannot replay this load")
                    ready = self.lookup_exact_ready(spec.variant_key, load_context)
                    if ready is not None:
                        self.register_ready_if_eligible(ready.kernel, spec, load_context)
                        return ready.kernel
                    was_initialized = getattr(kernel, "_initialization_complete", False)
                    kernel._init_handles()
                    if not was_initialized:
                        self.event("native_init")
                    self.register_ready_if_eligible(kernel, spec, load_context)
                    return kernel
                except BaseException:
                    slot.failed = True
                    raise
                finally:
                    self.initializing.keys = held
                    with self.lock:
                        slot.owner = None
        finally:
            with self.lock:
                slot.users -= 1
                if slot.users == 0:
                    self.initialization_locks.pop(key)

    def event(self, name):
        with self.lock:
            self.counts[name] += 1

    def lookup_exact_ready(self, variant_key, load_context):
        with self.lock:
            return self.exact.get((variant_key, load_context))

    def lookup_family_ready(self, family_key, load_context):
        with self.lock:
            return tuple(v for v in self.families.get(family_key, ()) if v.load_context == load_context)

    def has_registration(self, executable, family_key):
        with self.lock:
            return any(v is executable for v in self.families.get(family_key, ()))

    @staticmethod
    def _eligible(kernel, spec):
        return not (not getattr(kernel, "_initialization_complete", False) or not kernel.module or not kernel.function
                    or not callable(kernel._run) or kernel.hash != spec.variant_key
                    or kernel.src.hash() != spec.source.hash() or kernel.src.signature != spec.source.signature
                    or kernel.src.constants != spec.source.constants or kernel.src.attrs != spec.source.attrs)

    def register_ready_if_eligible(self, kernel, spec, load_context):
        if not self._eligible(kernel, spec):
            self.event("registration_skipped")
            return None
        return self._register_ready(kernel, spec, load_context)

    def _register_ready(self, kernel, spec, load_context):
        with self.lock:
            failure_key = (spec.family_key, spec.variant_key)
            if self.failures[failure_key] >= _FAILURE_THRESHOLD:
                self.event("registration_skipped")
                return None
            key = (spec.variant_key, load_context)
            existing = self.exact.get(key)
            family = self.families.get(spec.family_key, ())
            if existing is not None:
                self.event("registration_duplicate")
                if existing not in family:
                    if len(family) >= _MAX_FAMILY_READY:
                        self.event("registration_capacity_skipped")
                        return None
                    self.families[spec.family_key] = (*family, existing)
                return existing
            if len(self.exact) >= _MAX_READY or len(family) >= _MAX_FAMILY_READY:
                self.event("registration_skipped")
                return None
            value = ExecutableVariant(kernel, kernel.src, spec.variant_key, spec.family_key, load_context)
            self.exact[key] = value
            self.families[spec.family_key] = (*family, value)
            self.event("ready_registered")
            self.failures.pop(failure_key, None)
            return value

    def record_transform_failure(self, family, variant_key):
        with self.lock:
            key = (family, variant_key)
            self.failures[key] += 1
            self.event("transform_failure")
            if self.failures[key] >= _FAILURE_THRESHOLD:
                self.event("circuit_open")

    def disabled(self, family, variant_key):
        with self.lock:
            return self.failures[(family, variant_key)] >= _FAILURE_THRESHOLD

    def snapshot(self):
        with self.lock:
            return dict(self.counts)
