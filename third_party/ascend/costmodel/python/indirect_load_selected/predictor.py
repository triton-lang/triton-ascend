"""Existing frozen Load formulas; no fitting, filesystem or device access."""
import math
from indirect_store_selected.visible_features import visible_features

STATIC_MASKS = frozenset(("none", "box", "eighth", "quarter", "half", "full", "interior_box", "row_half",
                          "three_quarters", "full_minus_one", "one", "zero"))


def predict(model, case, runtime_coefficients=None, *, enabled=True, stage_ownership=False, matched_increment=False):
    if not enabled or not (stage_ownership and matched_increment):
        return dict(eligible=False, reason="Load_target_contract_missing_or_disabled")
    if case.get("op") != "load":
        raise ValueError("Load estimator requires an ordinary Load")
    dtype = case.get("payload_dtype")
    if not isinstance(dtype, str) or not dtype or ((dtype == "int1" or dtype.startswith("float8"))
                                                   and case["bytes"] != 1):
        raise ValueError("Explicit Load payload type and storage width must agree")
    if case.get("mask") not in STATIC_MASKS or case.get("K_visible") is False:
        raise ValueError("Data-dependent mask needs a proven Host contract or fallback")
    if not 4 <= math.prod(case["shape"]) <= 8192 or any(not 1 <= n <= 4096 for n in case["shape"]):
        raise ValueError("Load tile outside the frozen domain")
    w = case["W"]
    if type(w) is not int or not 1 <= w <= 64 or w & (w - 1):
        raise ValueError("Load base warp count outside the frozen domain")
    if case["mask"] == "none" and (any(case["lo"]) or case["hi"] != case["shape"]):
        raise ValueError("Unmasked Load bounds mismatch")
    if case["K"] and case["visibility"] != "device":
        coefficients = case.get("coefficients") if case["visibility"] == "known" else runtime_coefficients
        if coefficients is None or len(coefficients) < len(case["shape"]) or any(
                type(v) is not int or not 0 < v <= 2**31 for v in coefficients[:len(case["shape"])]):
            raise ValueError("Load visible coefficients must be explicit positive scalars")
    features = visible_features(case, runtime_coefficients, operation="load")
    if "semantic_zero" in features:
        return dict(eligible=True, ticks=0., leaf="", accepted=False)
    if model["strategy"] == "physical_short8_program_v1":
        from .features import visible_features as program_features
        from .physical_program import evaluate
        logical_case = dict(case)
        logical_case.setdefault("dtype", dtype)
        x = program_features(logical_case, runtime_coefficients)
        x.update(width=case["bytes"], mask=int(case["mask"] != "none"), rank=len(case["shape"]),
                 warp_band=0 if w <= 4 else 1 if w <= 16 else 2,
                 K_band=0 if x["K"] <= 16 else 1 if x["K"] <= 128 else 2,
                 leading_band=0 if x["leading"] <= 1 else 1 if x["leading"] <= 8 else 2)
        value, terms = evaluate(model, x)
        if not math.isfinite(value) or not 1 <= terms <= 8:
            raise ValueError("Invalid physical Load formula")
        return dict(eligible=True, ticks=value, leaf="physical_short8", terms=terms, accepted=False)
    strategy = model["strategy"]
    if strategy == "width_mask_prefix":
        branch = any(hi - lo != n for lo, hi, n in zip(case["lo"][:-1], case["hi"][:-1], case["shape"][:-1]))
    elif strategy == "width_mask_ns":
        branch = features["nS"] == 1
    else:
        raise ValueError("Unknown frozen Load routing strategy")
    key = f'{case["bytes"]}_{case["mask"] != "none"}_{branch}'
    leaf = model["models"].get(key, model["fallback"])
    if not 1 <= len(leaf["features"]) == len(leaf["coefficients"]) <= 8:
        raise ValueError("Load formula exceeds eight terms")
    value = sum(features[n] * c for n, c in zip(leaf["features"], leaf["coefficients"]))
    if not math.isfinite(value):
        raise ValueError("Nonfinite Load prediction")
    return dict(eligible=True, ticks=max(1., value), leaf=key if key in model["models"] else "fallback", accepted=False)
