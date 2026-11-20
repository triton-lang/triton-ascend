from dataclasses import dataclass


def kind_of(value):
    if hasattr(value, "dtype") and hasattr(value, "data_ptr"):
        return "ptr:" + str(value.dtype).removeprefix("torch.")
    if isinstance(value, float):
        return "float"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, str):
        return "str"
    if callable(value):
        import triton.language as tl

        if not isinstance(value, tl.dtype):
            return "callable"
    return "unknown"


def constant_value(value):
    import triton.language as tl

    if isinstance(value, tl.constexpr):
        value = value.value
    return Fact(kind=kind_of(value))


@dataclass(frozen=True)
class FactResult:
    facts: tuple
    complete: bool
    diagnostics: tuple


@dataclass(frozen=True)
class Fact:
    # Dependencies relevant to specialization, not full runtime data taint.
    deps: frozenset = frozenset()
    runtime: bool = False
    kind: str = "int"
    opaque: bool = False  # The value's compile-time/runtime behavior is unknown.


def merge(*values, runtime=False, kind=None):
    return Fact(frozenset().union(*(v.deps for v in values)), runtime or any(v.runtime for v in values), kind
                or (values[0].kind if values else "int"), any(v.opaque for v in values))
