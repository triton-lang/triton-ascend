"""Consistency checks and statistics for the final paired attention timings.

The caller must verify the hardware collector and correctness evidence. These
functions validate reported inputs and compute statistics; they cannot prove
that the reported measurements were collected on hardware.
"""
import copy
import math
from statistics import fmean, median
from op_statistic import validate_op_statistic_block

ARMS = ("cvpipeline", "ascendc")
TARGET_FRACTION = 0.9
PRODUCTION_SHAPES = tuple(
    (128, 8, n, hd)
    for n, dimensions in [(32768, (64,)), (16384, (64, 128)), (8192, (64, 128)),
                          (4096, (64, 128)), (2048, (64, 128)), (1024, (64, 128))]
    for hd in dimensions
)
ACCEPTED_EXCEPTION_SHAPE = (128, 8, 2048, 128)


def _validate_case(case):
    shape = tuple(case.get("shape", ()))
    if any(type(value) is not int for value in shape) or shape not in PRODUCTION_SHAPES:
        raise ValueError("Final comparison requires one of the eleven BS128/H8 production shapes")
    if case.get("block_m") != 128 or type(case.get("block_m")) is not int:
        raise ValueError("The declared original-kernel comparison uses BM128")
    bn = case.get("block_n")
    if type(bn) is not int or bn <= 0 or bn % 16 or shape[2] % bn:
        raise ValueError("BN must be a positive multiple of 16 dividing N")
    if (case.get("dtype") != "float16" or case.get("layout") != "BNSD"
            or case.get("causal") is not False or case.get("scale") != 0.5):
        raise ValueError("Expected FP16 BNSD, noncausal attention at scale 0.5")
    return shape


def _task_mean(samples):
    if not isinstance(samples, list) or len(samples) != 10:
        raise ValueError("Each arm needs exactly ten selected hardware task durations")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("Task durations must be finite positive numbers in microseconds")
    return fmean(samples)


def summarize_paired_rounds(case, rounds):
    """Require six alternating AB/BA rounds, using arithmetic task means."""
    shape = _validate_case(case)
    if len(rounds) != 6:
        raise ValueError("Exactly six fresh paired rounds are required")
    first_order = tuple(rounds[0].get("order", ()))
    if first_order not in (ARMS, ARMS[::-1]):
        raise ValueError("Each order must name CVPipeline and AscendC once")
    fingerprints = rounds[0].get("input_fingerprint")
    identities = rounds[0].get("arm_identities")
    if not fingerprints or not isinstance(identities, dict) or set(identities) != set(ARMS):
        raise ValueError("Record the common input fingerprint and both arm identities")
    if not all(identities.values()):
        raise ValueError("Arm identities cannot be empty")
    means = {arm: [] for arm in ARMS}
    representation = None
    used_sources = set()
    selected_operators = {}
    selected_device = None
    for index, row in enumerate(rounds):
        expected_order = first_order if index % 2 == 0 else first_order[::-1]
        if type(row.get("round_index")) is not int or row["round_index"] != index:
            raise ValueError("Rounds must have unique consecutive indices 0 through 5")
        if tuple(row.get("order", ())) != expected_order:
            raise ValueError("Alternate AB and BA for three rounds of each order")
        if row.get("measurement_mode") != "hardware-operation-duration" or row.get("timing_unit") != "us":
            raise ValueError("Simulator, host elapsed and mixed-unit timings are not hardware task samples")
        if row.get("case") != case:
            raise ValueError("Shapes, geometry and numerical settings must match across the cohort")
        if row.get("input_fingerprint") != fingerprints or row.get("arm_identities") != identities:
            raise ValueError("Do not pool changed inputs, compiler images or reference identities")
        for arm in ARMS:
            raw_key, aggregate_key = arm + '_durations_us', arm + '_op_statistic'
            if (raw_key in row) == (aggregate_key in row):
                raise ValueError('Supply exactly one raw-task or CSV-aggregate measurement per arm')
            kind = 'raw-task-durations' if raw_key in row else 'op-statistic-aggregate'
            if representation is not None and representation != kind:
                raise ValueError('Do not mix measurement representations within one cohort')
            representation = kind
            if raw_key in row:
                mean = _task_mean(row[raw_key])
            else:
                block = row[aggregate_key]
                mean = validate_op_statistic_block(block)
                expected_operator = '_attn_fwd' if arm == 'cvpipeline' else 'FlashAttentionScore'
                if block['op_type'] != expected_operator:
                    raise ValueError('CSV row is not the expected attention operator')
                operator = (block['op_type'], block['core_type'])
                if arm in selected_operators and selected_operators[arm] != operator:
                    raise ValueError('Operator/core type changed across rounds')
                selected_operators[arm] = operator
                if selected_device is not None and selected_device != block['device_id']:
                    raise ValueError('Do not pool operations from different physical devices')
                selected_device = block['device_id']
                source = (block['source_path'], block['source_record_index'])
                if source in used_sources:
                    raise ValueError('The same CSV record cannot be counted twice')
                used_sources.add(source)
            means[arm].append(mean)
    medians = {arm: median(values) for arm, values in means.items()}
    paired = [cvp < ascend / TARGET_FRACTION for cvp, ascend in zip(means["cvpipeline"], means["ascendc"])]
    median_met = medians["cvpipeline"] < medians["ascendc"] / TARGET_FRACTION
    strict_met = median_met and all(paired)
    headroom = 100 * (1 - medians["cvpipeline"] / (medians["ascendc"] / TARGET_FRACTION))
    return {
        "case": copy.deepcopy(case), "timing_unit": "us", "round_count": 6,
        "sample_representation": representation,
        "selected_tasks_per_arm_per_round": 10, "block_means_us": means,
        "median_us": medians, "ascendc_over_cvp_percent": 100 * medians["ascendc"] / medians["cvpipeline"],
        "paired_target_met": paired, "paired_target_wins": sum(paired),
        "median_target_met": median_met, "strict_target_met": strict_met,
        "target_headroom_percent": headroom,
        "independent_repeat_needed": strict_met and headroom < 2,
        "accepted_exception_case": shape == ACCEPTED_EXCEPTION_SHAPE,
        "exception_caveat": "Exception label does not turn a failed strict target into a pass; report current-parent regression separately.",
        "scope": "Reported cohort consistency and statistics only; collector/correctness evidence must be audited separately.",
    }


def summarize_matrix(cohorts):
    """Reject missing/duplicate shapes before producing an eleven-case report."""
    if len(cohorts) != len(PRODUCTION_SHAPES):
        raise ValueError("A complete report requires all eleven production cases")
    shapes = [_validate_case(cohort["case"]) for cohort in cohorts]
    if len(set(shapes)) != len(PRODUCTION_SHAPES) or set(shapes) != set(PRODUCTION_SHAPES):
        raise ValueError("Missing or duplicate production shape")
    rows = [summarize_paired_rounds(cohort["case"], cohort["rounds"]) for cohort in cohorts]
    rows.sort(key=lambda row: PRODUCTION_SHAPES.index(tuple(row["case"]["shape"])))
    return {"cases": rows, "strict_target_count": sum(row["strict_target_met"] for row in rows),
            "all_strict_targets_met": all(row["strict_target_met"] for row in rows),
            "repeat_required_cases": [row["case"] for row in rows if row["independent_repeat_needed"]]}
