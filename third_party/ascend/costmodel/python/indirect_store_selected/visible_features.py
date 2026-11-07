"""Frozen pure geometry/visible-stride features. No filesystem or labels."""

import math

ORIGINAL_FEATURES=['one', 'K', 'E', 'sqrtK', 'activeD', 'activeS', 'D', 'S', 'KlogTail', 'KactiveDlog', 'KactiveSlog', 'activeRows', 'leading', 'suffixD', 'KlogDstride', 'KlogDstride2', 'KlogStride', 'KmaxSlog', 'KsumInvActiveD', 'KsmallActiveD', 'KoverSqrtActiveD', 'KoverSqrtS', 'KoverTail', 'Kfp8', 'Kbool', 'hidden', 'KmaskedAxes', 'prefixMasked', 'tailMasked', 'KnD', 'Krank', 'sqrtE', 'groups32', 'activeRuns', 'partialLogical32', 'activeLogical32', 'EminusK', 'KlogRuns', 'logW', 'KlogW', 'KlogW2', 'sqrtW', 'baseWarps', 'activeWarpProxy', 'idleWarpProxy', 'highWarp32', 'logE', 'logK', 'sqrtD', 'sqrtS', 'activeDlog', 'activeSlog', 'logDstride', 'maxSlog', 'logTail', 'nD', 'nS', 'logW2', 'logWAbove2', 'logWAbove4', 'minW4', 'minW8', 'oneWarp', 'KAbove32', 'KAbove64', 'KAbove128', 'KAbove256', 'smallE32', 'smallE128', 'EAbove32', 'EAbove64', 'EAbove128', 'EAbove256', 'KoneWarp', 'KlogWAbove2', 'KAbove8', 'KAbove16', 'EAbove8', 'EAbove16', 'KsuffixLog', 'lastSSize', 'fullSuffix', 'KsmallD', 'KsumInvD', 'minSbytes', 'KunitS']

def feature(c, obs):
    e, k, w, f = (c['E'], c['K'], c['W'], c['F'])
    b = c['bytes']
    shape = c['shape']
    d = c['random']
    sprod = math.prod((n for n, is_d in zip(shape, d) if not is_d))
    dprod = math.prod((n for n, is_d in zip(shape, d) if is_d))
    known = c['visibility'] != 'device'
    co = obs['runtime_coefficients'][:len(shape)] if c['visibility'] == 'host' else c['coefficients'] if known else []
    ss = [abs(x) * b for x, is_d in zip(co, d) if not is_d]
    sl = math.log2(max(min(ss), 1)) if ss else 0
    return dict(one=1, E=e, K=k, bytes=k * b, groups32=math.ceil(k / 32), sqrtK=math.sqrt(k), logW=math.log2(w), Fminus1=f - 1, totalK=k * f, totalGroups32=math.ceil(k / 32) * f, sqrtTotalK=math.sqrt(k * f), S=sprod, D=dprod, logS=math.log2(sprod), logD=math.log2(dprod), KlogW=k * math.log2(w), KlogStride=k * sl, logStride=sl, hidden=int(not known), rank=len(shape), mask=int(c['mask'] != 'none'))

def extended_features(c, a):
    x = feature(c, a)
    nd = sum(c['random'])
    ns = len(c['shape']) - nd
    co = a['runtime_coefficients'][:len(c['shape'])] if c['visibility'] == 'host' else c['coefficients'] if c['visibility'] == 'known' else []
    ds = [abs(v) * c['bytes'] for v, d in zip(co, c['random']) if d]
    ls = math.log2(max(min(ds), 1)) if ds else 0
    x.update(width=c['bytes'], Ebytes=c['E'] * c['bytes'], totalBytes=c['E'] * c['F'] * c['bytes'], rows=c['E'] / c['shape'][-1], Krows=c['K'] / c['shape'][-1], Kfp8=c['K'] * int(c['payload_dtype'].startswith('float8')), Kbool=c['K'] * int(c['payload_dtype'] == 'int1'), logDstride=ls, KlogDstride=c['K'] * ls, known=int(bool(co)), nD=nd, nS=ns, KnD=c['K'] * nd, EnD=c['E'] * nd, Krank=c['K'] * len(c['shape']), EsqrtS=c['E'] * math.sqrt(x['S']), sumD=sum((n for n, d in zip(c['shape'], c['random']) if d)))
    tail = c['shape'][-1]
    active_tail = c['hi'][-1] - c['lo'][-1]
    x.update(tail=tail, activeTail=active_tail, logTail=math.log2(tail), KlogTail=c['K'] * math.log2(tail), ElogTail=c['E'] * math.log2(tail), KlogD=c['K'] * math.log2(x['D']), KlogS=c['K'] * math.log2(x['S']), KsqrtTail=c['K'] / math.sqrt(tail), KperTail=c['K'] / tail, tailBytes=tail * c['bytes'], rankK=c['K'] * len(c['shape']), KlogW2=c['K'] * math.log2(c['W']) ** 2)
    physical = [abs(v) * c['bytes'] for v in co]
    ss = [v for v, d in zip(physical, c['random']) if not d]
    sizes = [n for n, d in zip(c['shape'], c['random']) if not d]
    unit = sum((n for n, v in zip(sizes, ss) if v == c['bytes']))
    dlengths = [n for n, d in zip(c['shape'], c['random']) if d]
    x.update(minSbytes=min(ss) if ss else 0, maxSlog=math.log2(max(ss)) if ss else 0, KmaxSlog=c['K'] * math.log2(max(ss)) if ss else 0, unitS=unit, KunitS=c['K'] * unit, smallD=sum((n <= 4 for n in dlengths)), KsmallD=c['K'] * sum((n <= 4 for n in dlengths)), sumInvD=sum((1 / n for n in dlengths)), KsumInvD=c['K'] * sum((1 / n for n in dlengths)), KlogDstride2=c['K'] * ls * ls)
    active = [hi - lo for lo, hi in zip(c['lo'], c['hi'])]
    activeS = math.prod((n for n, d in zip(active, c['random']) if not d))
    activeD = math.prod((n for n, d in zip(active, c['random']) if d))
    s_indices = [i for i, d in enumerate(c['random']) if not d]
    lastS = s_indices[-1]
    suffixD = math.prod(active[lastS + 1:])
    leading = math.prod(active[:lastS])
    fullSuffix = math.prod(c['shape'][lastS + 1:])
    x.update(activeS=activeS, activeD=activeD, activeDlog=math.log2(max(activeD, 1)), activeSlog=math.log2(max(activeS, 1)), KactiveDlog=c['K'] * math.log2(max(activeD, 1)), KactiveSlog=c['K'] * math.log2(max(activeS, 1)), activeTail=active[-1], activeRows=c['K'] / max(active[-1], 1), suffixD=suffixD, fullSuffix=fullSuffix, KsuffixLog=c['K'] * math.log2(max(suffixD, 1)), leading=leading, lastSSize=active[lastS], maskedAxes=sum((n != s for n, s in zip(active, c['shape']))), KmaskedAxes=c['K'] * sum((n != s for n, s in zip(active, c['shape']))), prefixMasked=int(any((n != s for n, s in zip(active[:-1], c['shape'][:-1])))), tailMasked=int(active[-1] != c['shape'][-1]), KoverSqrtActiveD=c['K'] / math.sqrt(max(activeD, 1)), smallActiveD=sum((n <= 4 for n, d in zip(active, c['random']) if d)), sumInvActiveD=sum((1 / max(n, 1) for n, d in zip(active, c['random']) if d)))
    return x

def positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f'{name} must be a positive integer')
    return value

def validate_case(c, operation='store'):
    if operation not in ('load', 'store') or c.get('op') != operation or c.get('mode') not in ('SIMD', 'SIMT'):
        raise ValueError('Predictor operation or mode does not match its explicit scope')
    if c.get('F') != 1:
        raise ValueError('F>1 is not accepted by this F1 predictor')
    shape = c['shape']
    random = c['random']
    low = c['lo']
    high = c['hi']
    if not 2 <= len(shape) <= 5 or not len(shape) == len(random) == len(low) == len(high):
        raise ValueError('Invalid mixed-axis rank/shape/mask')
    if not all((isinstance(v, bool) for v in random)):
        raise ValueError('random axes must be explicit booleans')
    if not random[-1] or not any(random) or all(random):
        raise ValueError('Only logical-tail-D mixed S/D scope is supported')
    for i, (n, lo, hi) in enumerate(zip(shape, low, high)):
        positive_integer(n, f'shape[{i}]')
        if not isinstance(lo, int) or not isinstance(hi, int) or (not 0 <= lo <= hi <= n):
            raise ValueError('Mask endpoints must be in the logical tile')
    if c['E'] != math.prod(shape) or c['K'] != math.prod((hi - lo for lo, hi in zip(low, high))):
        raise ValueError('E/K must match the static tile and rectangular mask')
    if c['bytes'] not in (1, 2, 4, 8):
        raise ValueError('Unsupported scalar storage width')
    positive_integer(c['W'], 'base warp W')
    if c.get('visibility') not in ('known', 'host', 'device'):
        raise ValueError('Visibility must be explicit')

def visible_coefficients(c, runtime_coefficients=None):
    """No seeds, input IDs, observed indices, addresses, or hidden values."""
    visibility = c['visibility']
    if visibility == 'device':
        return None
    raw = c.get('coefficients') if visibility == 'known' else runtime_coefficients
    if raw is None or len(raw) < len(c['shape']):
        raise ValueError('Host visibility requires explicit caller-owned runtime_coefficients')
    result = list(raw[:len(c['shape'])])
    for value in result:
        positive_integer(value, 'visible positive scalar stride')
    return result

def logical_mask_features(c):
    """Integer equivalent of frozen static_mask_features.enrich, no label I/O."""
    active = []
    for index in range(c['E']):
        suffix = c['E']
        valid = True
        for n, lo, hi in zip(c['shape'], c['lo'], c['hi']):
            suffix //= n
            coordinate = index // suffix % n
            valid = valid and lo <= coordinate < hi
        active.append(valid)
    if sum(active) != c['K']:
        raise ValueError('Static mask count is inconsistent')
    runs = sum((v and (i == 0 or not active[i - 1]) for i, v in enumerate(active)))
    blocks = [active[i:i + 32] for i in range(0, c['E'], 32)]
    alive = sum((any(v) for v in blocks))
    partial = sum((any(v) and (not all(v)) for v in blocks))
    return dict(activeRuns=runs, partialLogical32=partial, activeLogical32=alive, totalActiveLogical32=alive * c['F'], totalPartialLogical32=partial * c['F'], maskRunsF=runs * c['F'], EminusK=c['E'] - c['K'], totalInactiveK=(c['E'] - c['K']) * c['F'], KlogRuns=c['K'] * math.log2(max(runs, 1)))

class _VisibleFeatures:
    original_features=ORIGINAL_FEATURES
    extended=staticmethod(extended_features)
    def features(self, case, runtime_coefficients=None, *, operation='store'):
        validate_case(case, operation)
        if case['K'] == 0:
            return {'semantic_zero': 1}
        co = visible_coefficients(case, runtime_coefficients)
        observation = {'runtime_coefficients': co} if case['visibility'] == 'host' else {}
        x = self.extended(case, observation)
        x.update(KsumInvActiveD=case['K'] * x['sumInvActiveD'], KsmallActiveD=case['K'] * x['smallActiveD'], sqrtActiveD=math.sqrt(x['activeD']), sqrtActiveS=math.sqrt(x['activeS']), KoverSqrtS=case['K'] / math.sqrt(x['activeS']), KoverTail=case['K'] / x['activeTail'])
        x.update(logical_mask_features(case))
        w = case['W']
        work = math.ceil(case['E'] / 32)
        lw = math.log2(w)
        x.update(baseWarps=w, activeWarpProxy=min(w, work), idleWarpProxy=max(w - work, 0), highWarp32=max(w - 32, 0), logE=math.log2(case['E']), logK=math.log2(max(case['K'], 1)), sqrtD=math.sqrt(x['activeD']), sqrtS=math.sqrt(x['activeS']), sqrtE=math.sqrt(case['E']), sqrtW=math.sqrt(w), logW2=lw * lw, logWAbove2=max(lw - 2, 0), logWAbove4=max(lw - 4, 0), minW4=min(w, 4), minW8=min(w, 8), oneWarp=int(w == 1))
        for boundary in (8, 16, 32, 64, 128, 256):
            x['KAbove' + str(boundary)] = max(case['K'] - boundary, 0)
            x['EAbove' + str(boundary)] = max(case['E'] - boundary, 0)
        x.update(smallE32=int(case['E'] <= 32), smallE128=int(case['E'] <= 128), KoneWarp=case['K'] * int(w == 1), KlogWAbove2=case['K'] * max(lw - 2, 0))
        if co is None:
            x.update(D_SspanRatio=0, Dunder128=0)
        else:
            offsets = {0}
            for lo, hi, random, stride in zip(case['lo'], case['hi'], case['random'], co):
                if not random:
                    offsets = {v + i * stride * case['bytes'] for v in offsets for i in range(hi - lo)}
            if not offsets:
                raise ValueError('Positive-K case produced empty structured offsets')
            x['D_SspanRatio'] = x['activeD'] * (max(offsets) - min(offsets) + case['bytes']) / (len(offsets) * case['bytes'])
            x['Dunder128'] = sum((abs(v) * case['bytes'] < 128 for v, random in zip(co, case['random']) if random))
        missing = sorted(set(self.original_features) - x.keys())
        if missing:
            raise ValueError('Incomplete frozen 86-feature reconstruction: ' + ', '.join(missing))
        return x

def visible_features(case, runtime_coefficients=None, *, operation='store'):
    return _VisibleFeatures().features(case, runtime_coefficients, operation=operation)
