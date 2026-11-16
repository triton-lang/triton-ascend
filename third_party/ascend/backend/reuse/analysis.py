from .dsl.facts import Fact, kind_of
from .dsl.uses import Uses
from .identity import typed
from .local_axis import match_tile_axes
from .model import AutoReuseProfile, FunctionPlan, ParameterClass, ParameterDecision
from .rules import match_grid_stride
from .syntax import parse_scope

_INTEGER_ANNOTATIONS = {
    **{f"i{bits}": f"int{bits}"
       for bits in (8, 16, 32, 64)},
    **{f"u{bits}": f"uint{bits}"
       for bits in (8, 16, 32, 64)},
}
_ANNOTATION_KINDS = {**_INTEGER_ANNOTATIONS, "i1": "bool", "u1": "bool"}


def specialized_one(parameter, value):
    annotation = parameter.annotation_type
    return (not parameter.is_constexpr and not parameter.do_not_specialize and isinstance(value, int)
            and not isinstance(value, bool) and int.__eq__(value, 1) is True
            and (not annotation or annotation in _INTEGER_ANNOTATIONS))


def parameter_type(parameter, value):
    """Return (structural kind, native ==1 specialization) for one argument."""
    one = specialized_one(parameter, value)
    annotation = parameter.annotation_type
    if annotation and not (parameter.is_constexpr or one):
        pointer = annotation.startswith("*")
        scalar = annotation.removeprefix("*k").removeprefix("*")
        return ("ptr:" if pointer else "") + _ANNOTATION_KINDS.get(scalar, scalar), one
    # This is structural analysis context, not native scalar type inference.
    # Width and signedness of unannotated arguments belong to the JIT binder.
    return kind_of(value), one


def type_context(fn, arguments):
    return tuple(parameter_type(p, value) for p, value in zip(fn.params, arguments))


def static_context(fn, arguments):
    """Keep constexpr-controlled paths distinct in the structural-analysis cache."""
    return tuple((i, typed(value))
                 for i, (p, value) in enumerate(zip(fn.params, arguments))
                 if p.is_constexpr and type(value) in (bool, int, float))


def analyze(fn, bound, source_key, *, allow_tile_reuse=True):
    with parse_scope():
        return _analyze(fn, bound, source_key, allow_tile_reuse)


def value_independent_parameters(fn, bound, plan, recipe):
    # Uses is symbolic in scalar values. Tile matching additionally evaluates
    # constexpr control/partition expressions; do not infer cache independence
    # from a successful proof for just one concrete invocation.
    if recipe is None:
        return plan.dynamic
    relaxed = tuple(sorted(set(plan.dynamic) | set(recipe.tiles)))
    names = {fn.arg_names[i] for i in relaxed}
    symbolic_bound = {name: value for name, value in bound.items() if name not in names}
    symbolic_recipe = match_tile_axes(fn, symbolic_bound) or match_grid_stride(fn)
    # Equality includes static/control dependencies, accesses, loops and index
    # checks. Failed or changed proofs keep the full analysis context exact.
    return relaxed if symbolic_recipe == recipe else ()


def _analyze(fn, bound, source_key, allow_tile_reuse):
    candidates = {p.num for p in fn.params if p.is_constexpr and isinstance(bound[p.name], (int, float))}
    inputs = {}
    for p in fn.params:
        kind, one = parameter_type(p, bound[p.name])
        inputs[p.name] = Fact(
            frozenset((p.num, )) if p.is_constexpr else frozenset(), not p.is_constexpr and not one, kind)
    uses = Uses(fn, inputs)
    facts = uses.run()
    dynamic = candidates - uses.reasons.keys() if facts.complete else set()
    recipe = match_tile_axes(fn, bound) or match_grid_stride(fn)
    if recipe:
        dynamic.difference_update(recipe.tiles)
        dynamic.difference_update(getattr(recipe, "static_parameters", ()))
    decisions = []
    for p in fn.params:
        if not p.is_constexpr:
            continue
        reasons = tuple(sorted(uses.reasons.get(p.num, ())))
        if recipe and p.num in recipe.tiles:
            classification = ParameterClass.ScheduleReusable if allow_tile_reuse else ParameterClass.Unknown
            if not allow_tile_reuse:
                reasons += (("TILE_REUSE_TARGET_UNSUPPORTED", 0), )
        elif p.num in dynamic:
            classification = ParameterClass.RuntimeEligible
        elif any(
            (reason.startswith("STATIC_") and reason != "STATIC_ARANGE") or reason == "CONTROL_SPECIALIZATION_PRESERVED"
                for reason, _ in reasons):
            classification = ParameterClass.StaticRequired
        else:
            classification = ParameterClass.Unknown
            reasons = reasons or facts.diagnostics or (("TYPE_OR_USE_NOT_SUPPORTED", 0), )
            if any(reason == "STATIC_ARANGE" for reason, _ in reasons):
                reasons += (("TILE_REUSE_UNPROVEN", 0), )
        decisions.append(ParameterDecision(p.num, classification, reasons))

    def plan(context):
        current = context["fn"]
        slots = tuple(p.num for p in current.params if p.is_constexpr and context["inputs"][p.name].deps & dynamic)
        lines = tuple(line for line, deps in context["locals"].items() if deps & dynamic)
        helpers = tuple((site, plan(child)) for site, child in sorted(context["helpers"].items()))
        return FunctionPlan(current, slots, lines, helpers)

    transform = plan(facts.facts[0]) if facts.complete else FunctionPlan(fn, (), (), ())
    recipe = recipe if allow_tile_reuse else None
    independent = value_independent_parameters(fn, bound, transform, recipe)
    return AutoReuseProfile(source_key, tuple(decisions), transform, recipe, independent)
