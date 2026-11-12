"""Development heuristic, evaluated retrospectively on 12 FP32 CV cases.

Inputs are CPU features of already compiled candidates. No timings or kernel
names are selection inputs. This module does not compile, load, or launch code.
The short-batch preference was designed using known development measurements;
it is not an independently calibrated latency model.
"""
import math

from cv_binary_selector import choose_factor


def select(original, candidates, fp32_scope=True):
    """Use latency ranking, with one empirical short-batch preference.

    Keep the compiler/resource/identity/work-equivalence gates of the latency
    selector. If M/N are single tiles, the original programs fit in one core
    wave, and batch x2 passes those gates, prefer x2. In other situations keep
    the latency decision. Generalization beyond the development set is untested.
    """
    decision = choose_factor(original, candidates, (2, 4), 'latency')
    if not fp32_scope or decision['choice'] is None:
        return decision
    grid = original['original_grid']
    eligible = {
        key: candidates[key]
        for key, comparison in decision['comparisons'].items()
        if 'baseline_score' in comparison
    }
    if any(item.get('merge_kind') in ('M', 'N') for item in candidates.values()):
        return decision
    if math.prod(grid) > original['cube_core_count']:
        return decision
    preferred = [
        key for key, item in eligible.items()
        if item.get('merge_kind') == 'independent_batch' and item['factor'] == 2 and all(
            size == 1 for axis, size in enumerate(grid) if axis != item['axis'])
    ]
    if not preferred:
        return decision
    key = min(preferred, key=lambda k: (eligible[k]['axis'], k))
    item = eligible[key]
    return dict(decision, choice=2, compiler_factor=2, enabled=True, axis=item['axis'], selected_candidate=key,
                selected_score=None, score_unit=None, reason_code='EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2',
                reason='单个空间tile且原program不超过核数：优先资源已确认的batch×2',
                decision_method='empirical preference; not lowest estimated latency',
                prior_latency_choice=decision['selected_candidate'], model_estimated_cycles=item['estimated_cycles'])
