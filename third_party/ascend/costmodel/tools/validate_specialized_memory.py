#!/usr/bin/env python3
"""Validate native predictions on all991 unchanged DEV configurations. No fitting."""
import argparse, collections, hashlib, json, math, os, pathlib
if hasattr(os, 'sched_setaffinity'): os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
else:
    import ctypes
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = ctypes.c_void_p
    k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    assert k.SetProcessAffinityMask(k.GetCurrentProcess(), 1)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('predictions', type=pathlib.Path)
p.add_argument('--development', type=pathlib.Path, required=True)
p.add_argument('--direct-development', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
args = p.parse_args()
data = json.loads(args.development.read_text(encoding='utf-8'))
source = args.predictions.read_text(encoding='utf-8').splitlines()
assert source[0] == '# model_sha256 ' + data['model_sha256']
predictions = {}
for line in source[1:]:
    cid, value = line.split()
    assert cid not in predictions
    value = float(value)
    assert math.isfinite(value) and value > 0
    predictions[cid] = value
assert len(predictions) == 3271 and len(data['cases']) == 991
errors = collections.defaultdict(list)
counts = collections.Counter()
seen = set()
for c in data['cases']:
    cid = c['case_id']
    assert cid not in seen and c['split'] == 'development' and len(c['target_us']) == 8
    seen.add(cid)
    group = c['mode'] + '_' + c['operation']
    family = group + '/' + c['family']
    mask = family + '/' + ('unmasked' if c['mask'] == 'none' else 'masked')
    for y in c['target_us']:
        assert math.isfinite(y) and y > 0
        error = predictions[cid] / y - 1
        for key in [group, family, mask]:
            errors[key].append(error)
    counts[group] += 1


def metrics(e):
    a = sorted(map(abs, e))
    n = len(e)
    return dict(n=n, MAPE=sum(a) / n, MRE=sum(e) / n, P95_APE=a[min(n - 1,
                                                                    math.ceil(.95 * n) - 1)],
                fraction_over20=sum(v > .2 for v in a) / n)


result = {key: metrics(value) for key, value in errors.items()}
primary = {k: v for k, v in result.items() if k.count('/') == 0}
families = {k: v for k, v in result.items() if k.count('/') == 1}
assert len(primary) == 4 and len(families) == 8 and sum(v['n'] for v in primary.values()) == 7928
addon = json.loads(args.direct_development.read_text())
assert addon['model_sha256'] == data['model_sha256'] and len(addon['cases']) == 16
addon_errors = [predictions[c['case_id']] / c['target_us'] - 1 for c in addon['cases']]
addon_metrics = metrics(addon_errors)
accepted = all(v['MAPE'] < .2 for v in [*primary.values(), *families.values(), addon_metrics])
report = dict(accepted=accepted, acceptance_basis='complete development set only', model_sha256=data['model_sha256'],
              development_configurations=991, development_input_rows=7928, development_cases_excluded=0,
              fitting_performed=False, NPU_calls=0, groups=result, new_shape_disjoint_DEV=addon_metrics,
              input_sha256=hashlib.sha256(args.development.read_bytes()).hexdigest())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(
    json.dumps(
        dict(accepted=accepted, DEV_MAPE={k: v['MAPE']
                                          for k, v in primary.items()}, DEV_configurations=991, DEV_input_rows=7928)),
    flush=True)
raise SystemExit(0 if accepted else 1)
