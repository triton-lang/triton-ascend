"""Visible physical facts; no model, labels, files, or device IO."""
import math


def visible_axis_features(case, runtime_coefficients=None):
    from indirect_store_selected.visible_features import ORIGINAL_FEATURES, visible_features
    original = visible_features(case, runtime_coefficients, operation='load')
    if 'semantic_zero' in original: return original
    out = {name: original[name] for name in ORIGINAL_FEATURES + ['Dunder128', 'D_SspanRatio']}
    shape = case['shape']
    random = case['random']
    rank = len(shape)
    if not 2 <= rank <= 5 or len(random) != rank: raise ValueError('supported rank2..5 physical axes required')
    active = [int(hi - lo) for lo, hi in zip(case['lo'], case['hi'])]
    width = case['bytes']
    if len(active) != rank or min(active) < 0 or width not in (1, 2, 4, 8): raise ValueError('invalid visible geometry')
    visibility = case['visibility']
    if visibility == 'device': co = None  # Neither the scalar buffer nor its length is accessed.
    elif visibility == 'host': co = runtime_coefficients
    elif visibility == 'known': co = case['coefficients']
    else: raise ValueError('unknown coefficient visibility')
    if visibility != 'device':
        if co is None or len(co) < rank: raise ValueError('actual positive visible coefficients required')
        if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in co[:rank]):
            raise ValueError('invalid visible stride')
    for axis in range(5):
        exists = axis < rank
        is_d = exists and bool(random[axis])
        is_s = exists and not random[axis]
        out[f'D_axis{axis}_extent'] = float(active[axis]) if is_d else 0.
        out[f'S_axis{axis}_extent'] = float(active[axis]) if is_s else 0.
        out[f'D_axis{axis}_stride_log2'] = math.log2(co[axis] * width) if is_d and co is not None else 0.
        out[f'S_axis{axis}_stride_log2'] = math.log2(co[axis] * width) if is_s and co is not None else 0.
        out[f'D_axis{axis}_inner_payload_bytes'] = float(math.prod(active[axis + 1:]) * width) if is_d else 0.
    out['D_runs'] = float(sum(bool(v) and (i == 0 or not random[i - 1]) for i, v in enumerate(random)))
    out['S_to_D_transitions'] = float(sum(not a and b for a, b in zip(random, random[1:])))
    out['D_to_S_transitions'] = float(sum(a and not b for a, b in zip(random, random[1:])))
    dtype = case['dtype'].lower()
    out['payload_integer'] = float(dtype.startswith(('int', 'uint')) or dtype == 'bool')
    out['payload_floating'] = float(dtype.startswith(('float', 'bfloat')))
    out['payload_bfloat16'] = float(dtype == 'bfloat16')
    assert len(out) == 119 and all(isinstance(x, (int, float)) and math.isfinite(x) for x in out.values())
    return out


def visible_features(case, runtime_coefficients=None):
    x = visible_axis_features(case, runtime_coefficients)
    if 'semantic_zero' in x: return x
    K = x['K']
    for i in range(5):
        for ds in ('D', 'S'):
            x[f'K_{ds}_axis{i}_stride_log2'] = K * x[f'{ds}_axis{i}_stride_log2']
            x[f'K_{ds}_axis{i}_extent_log2'] = K * math.log2(max(1., x[f'{ds}_axis{i}_extent']))
        x[f'K_D_axis{i}_inner_log2'] = K * math.log2(max(1., x[f'D_axis{i}_inner_payload_bytes']))
        d = x[f'D_axis{i}_extent']
        x[f'K_over_D_axis{i}_extent'] = K / d if d else 0.
    for name in ('D_runs', 'S_to_D_transitions', 'D_to_S_transitions', 'payload_bfloat16', 'payload_floating',
                 'payload_integer', 'hidden', 'prefixMasked', 'tailMasked'):
        x['K_' + name] = K * x[name]
    x['activeRows_logW'] = x['activeRows'] * x['logW']
    assert len(x) == 159 and all(math.isfinite(v) for v in x.values())
    return x
