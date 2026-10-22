import hashlib
import struct

from . import config


def typed(value):
    if value is None:
        return ("none", )
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float64", struct.pack("!d", value).hex())
    if type(value) is str:
        return ("str", value)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(typed(v) for v in value))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((typed(k), typed(v)) for k, v in value.items())))

    import triton.language as tl

    if isinstance(value, tl.dtype):
        return ("dtype", str(value))
    raise ValueError("constant cannot be represented without object identity")


def digest(value):
    return hashlib.sha256(repr(value).encode()).hexdigest()


class CachedNativeHash:
    """The native key builder only needs hash(); keep its algorithm unchanged."""

    def __init__(self, value):
        self.value = value

    def hash(self):
        return self.value


def variant_key(source, backend, options, env, options_hash=None):
    from triton.runtime.cache import get_cache_key

    if options_hash is not None:
        options = CachedNativeHash(options_hash)
    return hashlib.sha256(get_cache_key(source, backend, options, env).encode()).hexdigest()


def family_key(profile, source, options, target, env, options_hash=None):
    tiles = {(tile, ) for tile in profile.recipe.tiles} if profile.recipe else set()
    constants = {k: v for k, v in source.constants.items() if k not in tiles}
    # Runtime ==1 specializations remain exact, including the adaptable loop bound.
    return (profile.source_key, config.RULE_VERSION, config.ABI_VERSION, tuple(profile.plan.dynamic),
            typed(source.signature), typed(constants), typed(source.attrs),
            options.hash() if options_hash is None else options_hash, repr(target), typed(env))
