"""CPU-only, structural factor selection for compiled FP32 CV candidates.

0 means merging disabled (compiler factor=1); positive choices merge that many
programs. The requested set is 0/2/4/6, subject to actual compiler admission.
The default is an inspectable heuristic, not a calibrated latency model. LLVM scalar
steps, synchronization instructions, vector intrinsics and Cube microtiles have
unit weights. Each traced group's work is assigned to a modeled core, and the
largest accumulated core workload determines the score. The same rule handles
short launches, repeated waves and uneven compiled grid-stride loops. It does
not claim instruction counts are hardware cycles. The opt-in latency mode ranks
the dependency graph's estimated duration, including the global bandwidth bound.
Its configurable costs are modelling assumptions until independently calibrated.

No kernel names, timing samples, trained parameters, or device APIs are inputs.
The caller verifies binary/resource identities before passing a trace here.
"""
import ast
from collections import Counter
import heapq
import math

VERSION = 'cv-structural-multi-factor-v4-core-work'
LATENCY_VERSION = 'cv-latency-multi-factor-v1-event-graph'
RANKING_MODELS = {'work': (VERSION, 'work_score'), 'latency': (LATENCY_VERSION, 'estimated_cycles')}
DEFAULT_FACTORS = (2, 4, 6)
CONTROL = ('scalar_steps', 'local_flags', 'cross_flags')
# A data-path fence repeated for each matrix operation belongs to body work.
# Keeping total fences unchanged while grouping two matrices doubles fences per
# group; that is accounted as added body work, not a new synchronization defect.
DMA = ('nd2nz_calls', 'fixpipe_calls', 'load_dma_calls', 'store_dma_calls', 'barriers')


def _instruction_work(counts, tile):
    """Dimensionless work, without timing labels or input-size categories."""
    if any(not isinstance(v, int) or v < 0 for v in counts.values()):
        raise ValueError('instruction counters must be nonnegative integers')
    cube_instructions = 0
    matrix_calls = 0
    for key, count in counts.items():
        if key.startswith('mma_shape:'):
            m, k, n = ast.literal_eval(key.partition(':')[2])
            if min(m, k, n) <= 0:
                raise ValueError('invalid Cube matrix shape')
            cube_instructions += count * math.ceil(m / tile['m']) * math.ceil(k / tile['k']) * math.ceil(n / tile['n'])
            matrix_calls += count
    if matrix_calls != counts.get('mma_calls', 0):
        raise ValueError('missing Cube matrix shape accounting')
    control = {name: counts.get(name, 0) for name in CONTROL}
    vector = sum(value for name, value in counts.items() if name.startswith('llvm.hivm.'))
    transfers = sum(counts.get(name, 0) for name in DMA)
    return dict(control=control, control_total=sum(control.values()), body=cube_instructions + vector + transfers,
                cube_microtiles=cube_instructions, vector_intrinsics=vector, dma_commands=transfers)


def features(trace, proof, factor, axis, original_grid, spec_hash, compiler_hash, hardware, ranking_model='work',
             ranking_context=None):
    """Keep uneven groups separate; compare completed work on the busiest core."""
    if trace['status'] != 'STATIC_COST_ESTIMATED':
        raise ValueError('compiled CV trace is unavailable')
    groups, cores = trace['launched_groups'], trace['cube_core_count']
    if groups <= 0 or cores <= 0:
        raise ValueError('no execution groups or cores')
    traces = sorted(trace['group_traces'], key=lambda g: g['block'])
    if [g['block'] for g in traces] != list(range(groups)):
        raise ValueError('missing or duplicate execution groups')
    counts = trace['counts']
    group_sum = Counter()
    for group in traces:
        if 'counts' not in group:
            raise ValueError('per-group counters missing; regenerate the CPU trace')
        group_sum.update(group['counts'])
    if group_sum != Counter(counts):
        raise ValueError('per-group work does not sum to the candidate total')
    if not counts.get('mma_calls'):
        raise ValueError('no Cube matrix accounting')
    tile = hardware['compute_units']['cube']['fractal_sizes']['fp32']
    if any(not isinstance(tile[k], int) or tile[k] <= 0 for k in ('m', 'k', 'n')):
        raise ValueError('invalid target Cube fractal')
    work = _instruction_work(counts, tile)
    auto = trace['auto_blockify']
    if auto and groups > cores:
        raise ValueError('compiled grid-stride groups exceed core count')
    per_core = [dict(core=i, groups=[], control_total=0, body=0, total=0) for i in range(min(cores, groups))]
    # Without AutoBlockify, hardware assigns queued blocks to available cores.
    # List scheduling is an explicit estimate; it is not a runtime observation.
    ready = [(0, i) for i in range(len(per_core))]
    for group in traces:
        value = _instruction_work(group['counts'], tile)
        slot = group['block'] if auto else heapq.heappop(ready)[1]
        target = per_core[slot]
        target['groups'].append(group['block'])
        target['control_total'] += value['control_total']
        target['body'] += value['body']
        target['total'] += value['control_total'] + value['body']
        if not auto:
            heapq.heappush(ready, (target['total'], slot))
    visits = [g['entry_block_visits'] for g in traces]
    critical = max(per_core, key=lambda c: c['total'])
    result = dict(proof)
    result.update(analysis_available=True, factor=factor, axis=axis, original_grid=list(original_grid),
                  batch_grid=trace['logical_grid'], execution_groups=groups, mode=trace['mode'],
                  specification_sha256=spec_hash, compiler_sha256=compiler_hash, control_per_group={
                      k: v / groups
                      for k, v in work['control'].items()
                  }, control_total_per_group=work['control_total'] / groups, body_per_group=work['body'] / groups,
                  cube_microtiles_per_group=work['cube_microtiles'] / groups,
                  vector_intrinsics_per_group=work['vector_intrinsics'] / groups,
                  dma_commands_per_group=work['dma_commands'] / groups, useful_flops=counts['useful_flops'],
                  padded_flops=counts['padded_flops'], dma_read_bytes=trace['dma_read_bytes'],
                  dma_write_bytes=trace['dma_write_bytes'], l1_to_l0_bytes=counts['l1_to_l0_bytes'],
                  auto_blockify=trace['auto_blockify'], cube_core_count=cores, scoring_model=VERSION,
                  core_workloads=per_core, work_score=critical['total'], critical_core=critical['core'],
                  uniform_execution_groups=all(v == visits[0] for v in visits),
                  scheduling='compiled grid-stride loops' if auto else 'estimated block queue on available cores')
    if ranking_model not in RANKING_MODELS:
        raise ValueError('unknown ranking model')
    if ranking_model == 'latency':
        if not ranking_context:
            raise ValueError('latency ranking requires a shared cost profile identity')
        score = trace['score']
        if not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0:
            raise ValueError('invalid estimated duration')
        result.update(scoring_model=LATENCY_VERSION, estimated_cycles=score, score_unit='estimated_cycles',
                      ranking_context=ranking_context, critical_path_cycles=trace['critical_group_cycles'],
                      global_memory_bound_cycles=trace['global_memory_bound'],
                      estimated_core_cycles=trace['group_cycles'])
    return result


def resource_ok(candidate):
    return bool(candidate and candidate.get('resource_checked')
                and 0 <= candidate.get('ub_bits', -1) <= candidate.get('ub_capacity_bits', -1)
                and candidate.get('ub_capacity_bits', 0) > 0 and candidate.get('binary_sha256'))


def _compare_factor(original, merged, allowed_factors, ranking_model='work'):
    """Return a prediction and reason; unknown baseline resource gives no choice.

The caller must validate the full static input/launch contract for new inputs.
This function does not load, execute, or automatically deploy a selected binary.
"""
    if ranking_model not in RANKING_MODELS:
        raise ValueError('unknown ranking model')
    model, score_field = RANKING_MODELS[ranking_model]

    def result(choice, code, reason, **details):
        return dict(choice=choice, enabled=choice not in (0, None), compiler_factor=1 if choice == 0 else choice,
                    axis=merged.get('axis') if choice not in (0, None) else None,
                    status='STATIC_HEURISTIC_ONLY' if choice is not None else 'RESOURCE_UNCONFIRMED', reason_code=code,
                    reason=reason, **details)

    if not resource_ok(original):
        return result(None, 'BASELINE_RESOURCE_UNCONFIRMED', '原版资源尚未确认，不输出可用配置')
    if not resource_ok(merged):
        return result(0, 'MERGED_RESOURCE_UNAVAILABLE', '合并候选没有通过资源检查，保留原版')
    if not original.get('analysis_available') or not merged.get('analysis_available'):
        return result(0, 'STRUCTURE_UNCONFIRMED', '静态结构尚未覆盖，保留原版')
    factor = merged['factor']
    if original['factor'] != 1 or factor not in allowed_factors:
        return result(0, 'FACTOR_NOT_REQUESTED', '候选不属于本轮请求的合并块数')
    for name in ('specification_sha256', 'compiler_sha256', 'original_grid', 'mode', 'cube_core_count'):
        if original.get(name) != merged.get(name) or original.get(name) is None:
            return result(0, 'SPECIFICATION_MISMATCH', '输入、编译器或执行类型不一致')
    axis = merged['axis']
    expected = list(original['original_grid'])
    if axis not in (0, 1, 2) or expected[axis] < factor or expected[axis] % factor:
        return result(0, 'TAIL_OR_NO_MERGE_OPPORTUNITY', '该轴不能分成所请求块数的完整合并组')
    expected[axis] //= factor
    if merged['batch_grid'] != expected:
        return result(0, 'GRID_MISMATCH', '合并后的调度范围不匹配')
    if original['useful_flops'] != merged['useful_flops']:
        return result(0, 'WORK_MISMATCH', '两版本计算总量不一致')
    if ranking_model == 'work':
        for field in ('padded_flops', 'dma_read_bytes', 'dma_write_bytes', 'l1_to_l0_bytes'):
            if merged[field] > original[field]:
                return result(0, 'EXTRA_WORK_OR_TRAFFIC', '合并增加了填充计算或数据搬运量', increased_field=field)
    elif not original.get('ranking_context') or original['ranking_context'] != merged.get('ranking_context'):
        return result(0, 'COST_PROFILE_MISMATCH', '候选使用的耗时模型或参数不一致')
    # In latency mode, padding and extra traffic are priced by the event graph.
    # They are performance costs, not independent correctness/resource failures.
    if any(
            c.get('scoring_model') != model or not isinstance(c.get(score_field), (int, float))
            or not math.isfinite(c[score_field]) or c[score_field] <= 0 for c in (original, merged)):
        return result(0, 'WORK_SCORE_UNCONFIRMED' if ranking_model == 'work' else 'LATENCY_SCORE_UNCONFIRMED',
                      '候选评分尚未完成，保留原版')
    before, after = original[score_field], merged[score_field]
    if ranking_model == 'latency':
        evidence = dict(baseline_score=before, merged_score=after,
                        predicted_latency_reduction_pct=100 * (before - after) / before, scoring_model=model,
                        score_unit='estimated_cycles')
        if after < before:
            return result(factor, 'LOWER_ESTIMATED_LATENCY', '合并版预计完成时间更短', **evidence)
        return result(0, 'ORIGINAL_ESTIMATED_LATENCY_NOT_HIGHER', '原版预计完成时间不高于合并版', **evidence)
    evidence = dict(baseline_score=before, merged_score=after, instruction_proxy_margin=before - after,
                    predicted_work_reduction_pct=100 * (before - after) / before, scoring_model=VERSION)
    if after < before:
        return result(factor, 'LOWER_CRITICAL_CORE_WORK', '合并后最忙核的预计总工作量更少', **evidence)
    return result(0, 'ORIGINAL_CRITICAL_CORE_WORK_NOT_HIGHER', '原版最忙核的预计总工作量不高于合并版', **evidence)


def choose_zero_or_two(original, merged, ranking_model='work'):
    """Compatibility entry for a 0/2 comparison, using the same scoring rule."""
    return _compare_factor(original, merged, (2, ), ranking_model)


def choose_factor(original, candidates, factors=DEFAULT_FACTORS, ranking_model='work'):
    """Rank admitted candidates; failed compilation is never a numeric score.

The model itself does not assume a backend supports factor=6. The caller must
compile each requested candidate and provide its own resource proof/features.
Ties choose unmerged, then the smaller factor and axis. No timing inputs.
"""
    if ranking_model not in RANKING_MODELS:
        raise ValueError('unknown ranking model')
    model, score_field = RANKING_MODELS[ranking_model]
    factors = tuple(sorted(set(factors)))
    if not factors or any(type(f) is not int or f not in DEFAULT_FACTORS for f in factors):
        raise ValueError('requested factors must be a subset of 2/4/6')
    comparisons, rejected, ranked = {}, {}, []
    for key, candidate in candidates.items():
        decision = _compare_factor(original, candidate, factors, ranking_model)
        comparisons[key] = decision
        if 'baseline_score' in decision:
            ranked.append(
                dict(candidate=key, factor=candidate['factor'], axis=candidate['axis'], score=candidate[score_field]))
        else:
            rejected[key] = dict(reason_code=decision['reason_code'], reason=decision['reason'],
                                 candidate_status=candidate.get('status'),
                                 compiler_or_analysis_error=candidate.get('error'))
    if not ranked:
        decision = dict(next(iter(comparisons.values()), _compare_factor(original, None, factors, ranking_model)))
        decision.update(selected_candidate='unmerged' if decision['choice'] == 0 else None,
                        requested_choices=[0, *factors], rankings=[], excluded_candidates=rejected,
                        comparisons=comparisons)
        return decision
    ranked.append(dict(candidate='unmerged', factor=0, axis=None, score=original[score_field]))
    ranked.sort(key=lambda r: (r['score'], r['factor'], -1 if r['axis'] is None else r['axis']))
    selected = ranked[0]
    if selected['factor']:
        decision = dict(comparisons[selected['candidate']])
        decision['reason'] = ('通过检查的候选中，预计完成时间最短' if ranking_model == 'latency' else '通过检查的候选中，最忙核预计工作量最低')
    else:
        decision = dict(choice=0, compiler_factor=1, enabled=False, axis=None, status='STATIC_HEURISTIC_ONLY',
                        reason_code='ORIGINAL_LOWEST_LATENCY' if ranking_model == 'latency' else 'ORIGINAL_LOWEST_WORK',
                        reason='原版预计完成时间最短或并列最短' if ranking_model == 'latency' else '原版预计工作量最低或并列最低',
                        baseline_score=original[score_field], scoring_model=model)
    decision.update(selected_candidate=selected['candidate'], selected_score=selected['score'],
                    requested_choices=[0, *factors], rankings=ranked, excluded_candidates=rejected,
                    comparisons=comparisons,
                    score_unit='estimated_cycles' if ranking_model == 'latency' else 'instruction_work')
    return decision
