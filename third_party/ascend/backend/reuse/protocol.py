"""Frozen compiler inputs; compiled outputs use Triton's native cache."""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4
from weakref import ref

from . import config
from .snapshot import bundle, decode, encode, rebuild


def environment(device, target):
    from triton import knobs
    from triton.runtime import driver

    utils = driver.active.utils
    return dict(device=device, target=[target.backend, target.arch, target.warp_size],
                cache=str(Path(knobs.cache.dir).resolve()), physical_cores=list(utils.get_device_core()),
                core_limit=[utils.get_aicore_num(), utils.get_aivector_core_num()])


def serialize_request(request):
    return json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False)


def freeze_request(spec):
    """Check source transport before a compatible tile is allowed to launch."""
    return serialize_request(
        dict(schema=config.SCHEMA_VERSION, variant_key=spec.variant_key,
             source_kind="reuse" if hasattr(spec.source.fn, "reuse_identity") else "original",
             source=bundle(spec.source.fn), signature=encode(spec.source.signature),
             constants=encode(spec.source.constants), attrs=encode(spec.source.attrs),
             options=encode(spec.options.__dict__)))


def build_request(spec, env, generation, attempt=0, *, service_id=None, request_id=None, frozen=None):
    request = json.loads(freeze_request(spec) if frozen is None else frozen)
    request.update(environment=env, generation=generation, attempt=attempt, service_id=service_id or uuid4().hex,
                   request_id=request_id or uuid4().hex)
    return json.loads(serialize_request(request))


def rebuild_source(request):
    from triton.compiler import ASTSource
    from .transform import ReuseASTSource

    if request["schema"] != config.SCHEMA_VERSION:
        raise ValueError("protocol version differs")
    fn = rebuild(request["source"])
    if request["source_kind"] not in ("original", "reuse"):
        raise ValueError("unknown source kind")
    cls = ASTSource if request["source_kind"] == "original" else ReuseASTSource
    return cls(fn, decode(request["signature"]), decode(request["constants"]), decode(request["attrs"]))


@dataclass(frozen=True)
class FrozenSource:
    live_fn: object
    source_digest: str
    source_guard: object = None

    def is_current(self):
        fn = self.live_fn()
        if fn is None:
            return False
        try:
            return ((self.source_guard is None or self.source_guard.is_current())
                    and request_digest(bundle(fn)) == self.source_digest)
        except Exception:
            return False


def freeze_source(spec, request):
    return FrozenSource(ref(spec.source.fn), request_digest(request["source"]), getattr(spec, "source_guard", None))


def request_digest(request):
    return hashlib.sha256(serialize_request(request).encode()).hexdigest()
