"""New-interpreter compile worker. No wrapper invocation, launcher or device load."""
import json
import os
import sys


def compile_request(request):
    import torch
    import torch_npu  # noqa: F401
    from triton._C.libtriton import get_cache_invalidating_env_vars
    from triton.compiler import compile, make_backend
    from triton.runtime import driver
    from triton import knobs
    from pathlib import Path

    from .identity import variant_key
    from .protocol import environment, rebuild_source
    from .snapshot import decode

    # Align Python-side knobs too; parent changes may not reach os.environ.
    # The common cache root is fixed for the lifetime of this worker.
    root = str(Path(request["environment"]["cache"]).resolve())
    previous = getattr(compile_request, "cache_root", str(Path(knobs.cache.dir).resolve()))
    if previous != root:
        raise ValueError("worker cache root differs from fixed startup configuration")
    knobs.cache.dir = root
    if str(Path(knobs.cache.dir).resolve()) != root:
        raise ValueError("worker cache configuration could not be aligned")
    compile_request.cache_root = root
    torch.npu.set_device(request["environment"]["device"])
    target = driver.active.get_current_target()
    if environment(request["environment"]["device"], target) != request["environment"]:
        raise ValueError("worker environment mismatch")
    source = rebuild_source(request)
    backend = make_backend(target)
    options = backend.parse_options(decode(request["options"]))
    env = get_cache_invalidating_env_vars()
    if variant_key(source, backend, options, env) != request["variant_key"]:
        raise ValueError("worker VariantKey differs")
    kernel = compile(source, target=target, options=options.__dict__, _env_vars=env)
    if kernel.hash != request["variant_key"]:
        raise ValueError("compiled VariantKey differs")
    return kernel.hash


def main():
    # One persistent worker, one complete JSON request per line. Stdout belongs to
    # toolchain diagnostics; replies use a dedicated inherited pipe descriptor.
    with os.fdopen(int(sys.argv[1]), "w", buffering=1) as replies:
        for line in sys.stdin:
            request_id = None
            try:
                message = json.loads(line)
                request_id = message["request_id"]
                if message["transport"] != 2 or not isinstance(request_id, str):
                    raise ValueError("worker transport version/identity differs")
                result = {"variant_key": compile_request(message["request"])}
            except Exception as error:  # noqa: BLE001 - explicit cross-process error envelope
                result = {"error": type(error).__name__, "message": str(error)[:4096]}
            result.update(transport=2, request_id=request_id, stage="compile")
            replies.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
