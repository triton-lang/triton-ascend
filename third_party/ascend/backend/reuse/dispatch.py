from .debug import log_dispatch
from .footprint import adapt_grid
from .guards import adapt_tile
from .model import JitTailRequest, PreparedLaunch


def resolve_dispatch(registry, spec, profile, arguments, grid, load_context, backend, *, resolve_grid=None,
                     target_cached=None):
    exact = registry.lookup_exact_ready(spec.variant_key, load_context)

    def adapt(candidate):

        def reject(reason):
            log_dispatch(getattr(spec, "owner", None), profile, "candidate_rejected", arguments, candidate=candidate,
                         reason=reason)

        # Registration proof and compilation compatibility are separate checks.
        if not registry.has_registration(candidate, spec.family_key):
            return reject("FAMILY_REGISTRATION_MISSING")
        if candidate.variant_key == spec.variant_key:
            return PreparedLaunch(candidate, arguments)
        if profile.recipe is None:
            return reject("TILE_REUSE_NOT_PROVEN")
        candidate_grid = grid
        if callable(candidate_grid) and resolve_grid is not None:
            candidate_grid = resolve_grid()
        if type(candidate_grid) is list:
            candidate_grid = tuple(candidate_grid)
        actual = adapt_tile(profile.recipe, arguments, candidate.source.constants, candidate_grid, load_context[1],
                            compile_mode=spec.options.compile_mode)
        if actual is None:
            return reject("TILE_GUARD_FAILED: tile shape, index/overflow or memory coverage not proven")
        selected_grid = None
        if getattr(profile.recipe, "grid_axis", None) is not None:
            selected_grid = adapt_grid(profile.recipe, arguments, candidate.source.constants, candidate_grid)
            if selected_grid is None:
                return reject("GRID_ADAPTATION_FAILED")
            # An explicit compiler grid contract must remain exact.
            original_dimensions = (*candidate_grid, *((1, ) * (3 - len(candidate_grid))))
            selected_dimensions = (*selected_grid, *((1, ) * (3 - len(selected_grid))))
            if selected_dimensions != original_dimensions and getattr(spec.options, "grid_num_tiles", None) is not None:
                return reject("COMPILER_GRID_CONTRACT_MISMATCH")
        _, _, _, _, binder = spec.owner.device_caches[load_context[1]]
        # Only specialization policy is needed here. Serialized compiler options
        # contain derived/deprecated fields that are not user launch keywords.
        options = {"compile_mode": spec.options.compile_mode}
        _, specialization, _ = binder(*actual, **options)

        # Options were checked for the desired BuildSpec. Recheck the adapted
        # ABI without repeating option normalization on a compatible READY hit.
        from triton._utils import find_paths_if, get_iterable_path

        types = [item[0] for item in specialization]
        signature = dict(zip(spec.owner.arg_names, types))
        constants = {
            path: get_iterable_path(actual, path)
            for path in find_paths_if(types, lambda _, value: value == "constexpr")
        }
        attributes = [item[1] for item in specialization]
        attrs = {
            path: backend.parse_attr(get_iterable_path(attributes, path))
            for path in find_paths_if(attributes, lambda _, value: isinstance(value, str))
        }
        if (signature != candidate.source.signature or constants != candidate.source.constants
                or attrs != candidate.source.attrs):
            return reject("ADAPTED_ABI_MISMATCH: signature, constexpr constants or attributes differ")
        return PreparedLaunch(candidate, actual, selected_grid)

    if exact is not None:
        result = adapt(exact)
        if result is not None:
            registry.event("ready_exact_hit")
            log_dispatch(getattr(spec, "owner", None), profile, "ready_exact_hit", arguments,
                         candidate=result.executable, actual=result.arguments)
            return result
        registry.event("candidate_rejected")
    if target_cached is not None and target_cached(spec, load_context):
        registry.event("native_cache_target_selected")
        return JitTailRequest(spec)
    registry.event("exact_unavailable")
    for candidate in registry.lookup_family_ready(spec.family_key, load_context):
        if candidate is exact:
            continue
        result = adapt(candidate)
        if result is not None:
            registry.event("ready_compatible_hit")
            log_dispatch(getattr(spec, "owner", None), profile, "ready_compatible_hit", arguments,
                         candidate=result.executable, actual=result.arguments)
            return result
        registry.event("candidate_rejected")
    registry.event("family_exhausted")
    registry.event("jit_tail_miss")
    log_dispatch(getattr(spec, "owner", None), profile, "jit_tail_miss", arguments,
                 reason="NO_COMPATIBLE_READY_VARIANT")
    return JitTailRequest(spec)
