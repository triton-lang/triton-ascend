"""Host diagnostics gated by Triton's TRITON_DEBUG runtime knob."""
import sys

from .model import ParameterClass


def _enabled():
    from triton import knobs

    return knobs.runtime.debug


def _write(fn, message):
    name = getattr(fn, "__qualname__", getattr(fn, "name", "<unknown>"))
    module = getattr(fn, "__module__", "")
    kernel = f"{module}.{name}" if module else name
    print(f"[ascend-reuse] kernel={kernel} {message}", file=sys.stderr, flush=True)


def _names(fn, positions):
    return ", ".join(fn.arg_names[i] for i in positions) or "-"


def log_profile(fn, profile):
    if not _enabled():
        return
    _write(fn, "analysis: constexpr decisions (eligibility, before dispatch)")
    for decision in profile.decisions:
        name = fn.arg_names[decision.position]
        classification = decision.classification
        if classification is ParameterClass.RuntimeEligible:
            _write(fn, f"constexpr={name} dynamicized: constexpr -> runtime argument")
        elif classification is ParameterClass.ScheduleReusable:
            _write(fn, f"constexpr={name} schedule-reusable: rule={profile.recipe.rule}; remains constexpr")
        else:
            reasons = decision.reasons or (("SCHEDULE_STATIC_PARAMETER", 0), )
            detail = "; ".join(f"{reason} (JIT source line {line})" if line else reason for reason, line in reasons)
            _write(fn, f"constexpr={name} rejected: classification={classification.value}; reasons={detail}")


def log_bypass(fn, reason):
    if not _enabled():
        return
    names = _names(fn, (p.num for p in getattr(fn, "params", ()) if p.is_constexpr))
    _write(fn, f"original JIT: constexpr=[{names}]; reason={reason}")


def log_dispatch(fn, profile, outcome, arguments, *, candidate=None, actual=None, reason=None):
    if not _enabled():
        return
    dynamic = profile.plan.dynamic if profile is not None else ()
    recipe = profile.recipe if profile is not None else None
    tiles = recipe.tiles if recipe is not None else ()
    message = f"dispatch={outcome}; dynamicized=[{_names(fn, dynamic)}]"
    if candidate is not None:
        message += f"; candidate={candidate.variant_key}"
    if actual is not None and tiles:
        # Tile guards permit only builtin integers. Never repr tensors or user objects.
        reused = ", ".join(f"{fn.arg_names[i]}: {arguments[i]} -> {actual[i]}" for i in tiles
                           if type(arguments[i]) is int and type(actual[i]) is int and arguments[i] != actual[i]) or "-"
        message += f"; schedule-reused=[{reused}]"
    if reason is not None:
        message += f"; schedule-constexpr=[{_names(fn, tiles)}]; reason={reason}"
    _write(fn, message)
