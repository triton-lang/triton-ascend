# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Optional CPU selector at the existing Ascend JIT continuation boundary.

main-dev does not supply a candidate provider. Without one this handler keeps
the original JIT/reuse path, without compilation, scoring or option changes.
"""


class _LaunchHandler:

    def run(self, fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail):
        from .reuse.config import enabled

        def continuation(*arguments, **keywords):
            if enabled():
                from .reuse.bridge import get_handler
                return get_handler().run(fn, *arguments, backend, jit_tail, **keywords)
            return jit_tail(*arguments, **keywords)

        provider = getattr(backend, "run_block_auto", None)
        if provider is not None and not warmup:
            from .block_auto import select_candidates
            # The provider owns compilation, cache identity, capture handling
            # and guarded dispatch. It must call continuation exactly once,
            # retaining the original launch when no candidate can be proved.
            return provider(fn, args, kwargs, grid, warmup, device, stream, select_candidates=select_candidates,
                            continuation=continuation)
        return continuation(args, kwargs, grid, warmup, device, stream)


_handler = _LaunchHandler()


def get_launch_handler():
    return _handler
