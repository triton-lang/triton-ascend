"""Apply one CPU-only 0/2/4/6 rule to frozen compiled input specifications.

The manifest supplies contracts and hashes of compiler outputs. Case IDs and
kernel names label the report only; the selector receives compiled-work
features. This entry never opens measured performance data or launches NPU code.
"""
import argparse
import copy
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

from backend_cost_model import read_cost_table
from cpu_resource_gate import inspect_final_ub
from cv_binary_selector import DEFAULT_FACTORS, RANKING_MODELS, choose_factor, features
from cv_candidate_plan import coverage, merge_kind
from cv_cost_model import analyze_candidate, validated_assumptions
from artifact_io import digest, read, write
from cv_empirical_rule import select as empirical_select


def check_merge_metadata(metadata, factor, axis):
    if factor == 1:
        return
    merge = metadata['block_batching']
    # make_ttir emits REWRITTEN; block_batching_report.validate_compilation_result
    # changes it to COMPILED_GUARDED after binary generation. Both describe
    # the candidate rewrite, not a runtime APPLIED result.
    if (merge['compile_status'] not in ('REWRITTEN', 'COMPILED_GUARDED') or merge['requested_factor'] != factor
            or merge['selected_axis'] != axis):
        raise ValueError('compiled merging factor/axis mismatch')
    if merge.get('cv_lowering') in ('rank3_masked_batch_v1', 'rank3_uniform_control_batch_v1'):
        if (metadata.get('enable_hivm_batch_matmul') is not True
                or merge.get('cv_memory_lowering') != 'full_reduction_guard_v3'
                or merge.get('guard', {}).get('require_same_fractal_bounds') is not True):
            raise ValueError('finalized masked-batch memory contract missing')
    transforms = metadata['program_grid_transforms']['transforms']
    if (len(transforms) != 1 or transforms[0]['kind'] != 'ceil_div' or transforms[0]['factor'] != factor
            or transforms[0]['axis'] != axis):
        raise ValueError('compiled grid transform mismatch')


def checked_candidate(item):
    files = item['files']
    for record in files.values():
        if digest(record['path']) != record['sha256']:
            raise ValueError('changed compiler artifact: ' + record['path'])
    paths = {key: Path(record['path']) for key, record in files.items()}
    proof, metadata = read(paths['resource']), read(paths['metadata'])
    if not proof['resource_checked'] or proof['status'] != 'UB_RESOURCE_CHECKED':
        raise ValueError('candidate has no completed UB resource check')
    for key, value in (('binary', proof['artifact']['sha256']), ('ttadapter', proof['identity']['ir_sha256']),
                       ('final_ir', proof['final_ir_sha256']), ('compiler', proof['identity']['compiler_sha256'])):
        if files[key]['sha256'] != value:
            raise ValueError('resource proof identity mismatch: ' + key)
    usage = inspect_final_ub(paths['final_ir'].read_text(), proof['target'])
    if not ub_proof_matches(usage, proof['ub']) or usage['address_extent_bits'] > usage['capacity_bits']:
        raise ValueError('invalid final UB allocation')
    check_merge_metadata(metadata, item['factor'], item['axis'])
    modules = {role: paths['llvm_' + role] for role in ('aic', 'aiv') if 'llvm_' + role in paths}
    return paths, metadata, modules, dict(resource_checked=True, resource_proof=str(paths['resource']),
                                          resource_proof_sha256=files['resource']['sha256'],
                                          binary_sha256=files['binary']['sha256'], ub_bits=usage['address_extent_bits'],
                                          ub_capacity_bits=usage['capacity_bits'])


def ub_proof_matches(observed, recorded):
    """Older exact-allocation proofs predate the optional i1-bound markers."""

    def normalize(value):
        normalized = copy.deepcopy(value)
        normalized.setdefault('conservative_upper_bound', False)
        for buffer in normalized.get('buffers', []):
            buffer.setdefault('conservative_i1', False)
        return normalized

    return normalize(observed) == normalize(recorded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hardware', type=Path, required=True)
    parser.add_argument('--cost-table', type=Path, required=True)
    parser.add_argument('--factors', type=int, nargs='+', choices=DEFAULT_FACTORS, default=(2, 4))
    parser.add_argument('--ranking-model', choices=RANKING_MODELS, default='work')
    parser.add_argument('--prepared-shortlist', action='store_true',
                        help='Rank only the supplied baseline and at most two prepared merged candidates')
    parser.add_argument('--empirical-short-batch', action='store_true',
                        help='Use the development-set FP32 short-batch preference with latency ranking')
    parser.add_argument('--model-parameters', type=Path,
                        help='JSON cost profile for latency ranking; no measured performance inputs')
    args = parser.parse_args()
    if args.empirical_short_batch and (args.ranking_model != 'latency' or not args.prepared_shortlist):
        parser.error('--empirical-short-batch requires --ranking-model latency --prepared-shortlist')
    if args.empirical_short_batch and set(args.factors) != {2, 4}:
        parser.error('--empirical-short-batch uses the established F2/F4 rule')
    if args.model_parameters and args.ranking_model != 'latency':
        parser.error('--model-parameters requires --ranking-model latency')
    args.output.mkdir(parents=True, exist_ok=False)
    forbidden = []

    def audit(event, values):
        reason = None
        if event == 'import' and values[0].split('.')[0] in ('torch', 'torch_npu', 'triton', 'acl', 'pyacl'):
            reason = 'runtime import: ' + values[0]
        if event in ('subprocess.Popen', 'ctypes.dlopen'):
            reason = 'external execution: ' + event
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = str(values[0])
            if Path(path).name in ('labels.json', 'measurement.json', 'reviewed_summary.json',
                                   'evaluation.json') or any(token in path
                                                             for token in ('/dev/davinci', '/dev/devmm', '/dev/nvidia',
                                                                           '/dev/hisi', '/dev/npu')):
                reason = 'timing or device input: ' + path
        if reason:
            forbidden.append(reason)
            raise RuntimeError(reason)

    sys.addaudithook(audit)
    manifest, hardware = read(args.manifest), read(args.hardware)
    if args.prepared_shortlist:
        for case in manifest['cases']:
            validate_shortlist(case['candidates'], args.factors)
    table = read_cost_table(args.cost_table.read_text())
    profile = read(args.model_parameters) if args.model_parameters else dict(schema_version=1,
                                                                             calibration='uncalibrated', parameters={})
    if profile.get('schema_version') != 1 or not isinstance(profile.get('parameters'), dict):
        raise ValueError('cost profile requires schema_version=1 and parameters object')
    assumptions = validated_assumptions(profile['parameters'])
    model, score_field = RANKING_MODELS[args.ranking_model]
    source_hashes = {
        name: digest(Path(__file__).with_name(name))
        for name in ('run_cv_spec_experiment.py', 'cv_binary_selector.py', 'cv_cost_model.py', 'backend_cost_model.py',
                     'cpu_resource_gate.py', 'cv_candidate_plan.py', 'cv_batch_cost.py', 'cv_empirical_rule.py',
                     'artifact_io.py')
    }
    ranking_context = hashlib.sha256(
        json.dumps(
            dict(model=model, assumptions=assumptions, sources=source_hashes, hardware=digest(args.hardware),
                 cost_table=digest(args.cost_table)), sort_keys=True).encode()).hexdigest()
    write(
        args.output / 'protocol.json',
        dict(
            model=model, ranking_model=args.ranking_model, candidates=[0, *sorted(set(args.factors))],
            manifest_sha256=digest(args.manifest), source_sha256=source_hashes, ranking_context=ranking_context,
            cost_profile=profile, effective_parameters=assumptions,
            parameter_file_sha256=digest(args.model_parameters) if args.model_parameters else None,
            hardware_sha256=digest(args.hardware), cost_table_sha256=digest(args.cost_table), device_executed=False,
            measured_performance_used=False, prepared_shortlist=args.prepared_shortlist,
            empirical_short_batch=args.empirical_short_batch,
            rule_uses_known_development_outcomes=args.empirical_short_batch,
            score=('estimated cycles: max(core critical paths, global memory bandwidth bound)' if args.ranking_model
                   == 'latency' else 'dimensionless unit-weight instruction work on the busiest modeled core'),
            new_numerical_verification=False, performance_hit_rate_established=False))
    rows = []
    for case in manifest['cases']:
        contract = case['contract']
        spec_hash = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        row = dict(id=case['id'], name=case['name'], candidates={})
        for key, item in case['candidates'].items():
            if item['factor'] not in (1, *args.factors):
                continue
            value = dict(resource_checked=False, analysis_available=False, factor=item['factor'], axis=item['axis'])
            if 'files' not in item:
                evidence = item.get('rejection_evidence')
                if evidence:
                    if digest(evidence['path']) != evidence['sha256']:
                        raise ValueError('changed rejection evidence')
                    report = read(evidence['path'])
                    value.update(status=item['status'], error=report.get('error', report.get('reason')),
                                 rejection_evidence=evidence)
                else:
                    value.update(status='NOT_COMPILED', error='no compiled candidate supplied')
                row['candidates'][key] = value
                print(case['id'], key, value['status'], value['error'], flush=True)
                continue
            try:
                paths, metadata, modules, proof = checked_candidate(item)
                value.update(proof)
                trace = analyze_candidate({r: p.read_text()
                                           for r, p in modules.items()}, paths['ttir'].read_text(),
                                          paths['final_ir'].read_text(), metadata, contract, item['axis'],
                                          item['factor'], table, assumptions=assumptions)
                value = features(trace, proof, item['factor'], item['axis'], contract['grid'], spec_hash,
                                 item['files']['compiler']['sha256'], hardware, args.ranking_model, ranking_context)
                value['merge_kind'] = merge_kind(metadata, item['factor'])
                value['status'] = 'STATIC_FEATURES_READY'
                write(args.output / 'traces' / case['id'] / (key + '.json'), trace)
            except Exception as error:
                value.update(status='STATIC_ANALYSIS_UNCONFIRMED', error=f'{type(error).__name__}: {error}')
            row['candidates'][key] = value
            print(case['id'], key, value['status'], value.get(score_field, value.get('error')), flush=True)
        for factor in (() if args.prepared_shortlist else args.factors):
            if not any(c['factor'] == factor for c in row['candidates'].values()):
                row['candidates'][f'a0_f{factor}'] = dict(factor=factor, axis=0, resource_checked=False,
                                                          analysis_available=False, status='NOT_COMPILED',
                                                          error='no compiled candidate supplied')
        row['decision'] = choose_factor(row['candidates'].get('unmerged'),
                                        {k: v
                                         for k, v in row['candidates'].items()
                                         if k != 'unmerged'}, args.factors, args.ranking_model)
        if args.empirical_short_batch:
            tensors = [v for v in contract['arguments'].values() if v['kind'] == 'tensor']
            fp32 = bool(tensors) and all(v['dtype'] == 'torch.float32' for v in tensors)
            row['decision'] = empirical_select(row['candidates'].get('unmerged'),
                                               {k: v
                                                for k, v in row['candidates'].items()
                                                if k != 'unmerged'}, fp32)
        row['coverage'] = coverage(contract['grid'], row['candidates'], args.factors)
        rows.append(row)
        write(args.output / 'decisions.json', rows)
        write(args.output / 'progress.json', dict(completed=len(rows), total=len(manifest['cases'])))
    with (args.output / 'selections.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        score_name = '预计周期评分' if args.ranking_model == 'latency' else '工作量评分'
        writer.writerow(['编号', 'kernel', 'CPU选择', 'axis', '原版' + score_name, '所选' + score_name, '选择原因'])
        for row in rows:
            decision = row['decision']
            writer.writerow([
                row['id'], row['name'], decision['choice'],
                decision.get('axis'),
                decision.get('baseline_score'),
                decision.get('selected_score'), decision['reason']
            ])
    summary = dict(
        model=model, ranking_model=args.ranking_model, cases=len(rows),
        scored_cases=sum('baseline_score' in r['decision']
                         for r in rows), choices=dict(Counter(str(r['decision']['choice']) for r in rows)),
        complete_candidate_coverage=sum(r['coverage']['all_candidates_resolved'] for r in rows),
        candidates=dict(Counter(c['status'] for r in rows for c in r['candidates'].values())),
        forbidden_attempts=forbidden, device_executed=False, device_modules_loaded=[
            n for n in sys.modules if n.split('.')[0] in ('torch', 'torch_npu', 'triton', 'acl', 'pyacl')
        ], measured_performance_used=False, performance_hit_rate_established=False)
    write(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def validate_shortlist(candidates, factors):
    """A shortlist is explicit; missing axes must not be presented as searched."""
    if not 1 <= len(candidates) <= 3 or 'unmerged' not in candidates:
        raise ValueError('shortlist requires the baseline and at most two merged candidates')
    baseline = candidates['unmerged']
    if baseline['factor'] != 1 or baseline['axis'] != -1:
        raise ValueError('invalid unmerged baseline')
    for key, value in candidates.items():
        if key == 'unmerged':
            continue
        factor, axis = value['factor'], value['axis']
        if (type(factor) is not int or factor not in factors or type(axis) is not int or axis not in (0, 1, 2)
                or key != f'a{axis}_f{factor}'):
            raise ValueError('invalid shortlist axis/factor')


if __name__ == '__main__':
    main()
