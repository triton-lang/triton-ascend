"""Single selected model arithmetic; explicit metadata, no fit or labels."""
import math
from .visible_features import visible_features


def predict(model, case, runtime_coefficients=None, *, enabled=True, stage_ownership=False, first_store=False,
            matched_increment=False):
    if not enabled:
        return dict(eligible=False, reason='selected_formula_candidate_disabled', accepted=False)
    if not (stage_ownership and first_store and matched_increment):
        return dict(eligible=False, reason='caller_contract_missing', accepted=False)
    shape = case['shape']
    if not 4 <= math.prod(shape) <= 8192 or any(not 1 <= v <= 4096 for v in shape):
        raise ValueError('Outside existing bounded F1 metadata domain')
    w = case['W']
    if type(w) is not int or not 1 <= w <= 64 or w & (w - 1):
        raise ValueError('Outside existing base warp domain')
    dtype = case['payload_dtype']
    if not isinstance(dtype, str) or not dtype:
        raise ValueError('Explicit actual payload type is required')
    if (dtype == 'int1' or dtype.startswith('float8')) and case['bytes'] != 1:
        raise ValueError('Actual payload type/storage width conflict')
    if case['mask'] == 'none' and (any(case['lo']) or case['hi'] != shape):
        raise ValueError('Unmasked bounds mismatch')
    if case['K'] and case['visibility'] != 'device':
        co = case.get('coefficients') if case['visibility'] == 'known' else runtime_coefficients
        if co is None or len(co) < len(shape) or any(type(v) is not int or not 0 < v <= 2**31 for v in co[:len(shape)]):
            raise ValueError('Explicit safe positive visible coefficients required')
    x = visible_features(case, runtime_coefficients)
    if 'semantic_zero' in x:
        return dict(eligible=True, ticks=0., reason='semantic_zero', leaf='', accepted=False)
    strategy = model['strategy']
    key = str(case['bytes'])
    if strategy == 'width_mask_size':
        key += f"_{case['mask'] != 'none'}_{case['E'] <= 128}"
    elif strategy != 'width':
        raise ValueError('Unsupported selected model strategy')
    leaf_key = key if key in model['models'] else 'fallback'
    leaf = model['models'].get(key, model['fallback'])
    ticks = sum(x[name] * coefficient for name, coefficient in zip(leaf['features'], leaf['coefficients']))
    if not math.isfinite(ticks):
        raise ValueError('Prediction is not finite')
    return dict(eligible=True, ticks=max(1., ticks), leaf=leaf_key,
                reason='eligible_selected_formula_production_unaccepted', accepted=False)
