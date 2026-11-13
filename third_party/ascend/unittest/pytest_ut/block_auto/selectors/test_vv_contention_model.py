import copy
from dataclasses import asdict

import pytest

from backend_cost_model import ASSUMPTIONS, Analyzer, Unsupported
from vv_contention_model import ContentionProfile, DependencySchedule, choose, schedule


def trace(compute=0, requests=1, size=1024):
    s = DependencySchedule(ASSUMPTIONS)
    for _ in range(requests):
        s.scalar_step()
        s.dma("load", size)
        s.flag("llvm.hivm.SET.FLAG.IMM", [4, 1, 0])
        s.flag("llvm.hivm.WAIT.FLAG.IMM", [4, 1, 0])
        s.vf({"cycles": compute, "counts": {}})
        s.flag("llvm.hivm.SET.FLAG.IMM", [1, 5, 0])
        s.flag("llvm.hivm.WAIT.FLAG.IMM", [1, 5, 0])
        s.dma("store", size)
        s.barrier([])
    return s.result()


def compute_trace(cycles):
    s = DependencySchedule(ASSUMPTIONS)
    s.vf({"cycles": cycles, "counts": {}})
    return s.result()


def test_no_contention_preserves_local_dependencies_and_barriers():
    t = trace(compute=100, requests=3)
    before = copy.deepcopy(t)
    result = schedule([t], 1, ContentionProfile(1e30, 0, 0))
    assert result["score"] == t["cycles"]
    assert t == before
    assert result["dma_requests"] == 6
    assert result["dma_bytes"] == 6 * 1024


def test_shared_requests_can_hurt_identical_byte_counts():
    profile = ContentionProfile(512, 64, 0)
    fragmented = schedule([trace(requests=8, size=128)] * 4, 4, profile)
    combined = schedule([trace(requests=1, size=1024)] * 4, 4, profile)
    assert fragmented["dma_bytes"] == combined["dma_bytes"]
    assert fragmented["dma_requests"] == combined["dma_requests"] * 8
    assert fragmented["score"] > combined["score"]
    assert fragmented["dma_issue_wait_cycles"] > 0


def test_command_rate_and_bandwidth_are_independent_constraints():
    work = [trace(requests=3, size=8192)] * 8
    fast = schedule(work, 8, ContentionProfile(512, 1, 0))
    slow_issue = schedule(work, 8, ContentionProfile(512, 128, 0))
    slow_memory = schedule(work, 8, ContentionProfile(1, 1, 0))
    assert slow_issue["score"] > fast["score"]
    assert slow_memory["score"] > fast["score"]
    assert slow_memory["score"] >= slow_memory["dma_bytes"]


def test_merging_compute_work_below_core_count_loses_parallelism():
    profile = ContentionProfile(512, 8, 32)
    baseline = schedule([compute_trace(1000)] * 4, 4, profile)
    merged = schedule([compute_trace(2000)] * 2, 4, profile)
    assert baseline["score"] < merged["score"]


def test_small_blocks_amortize_dispatch_without_a_kernel_name_rule():
    profile = ContentionProfile(512, 8, 32)
    baseline = schedule([compute_trace(1)] * 8, 8, profile)
    merged = schedule([compute_trace(4)] * 2, 8, profile)
    assert merged["score"] < baseline["score"]


def test_nonpersistent_blocks_wait_for_a_free_core():
    result = schedule([compute_trace(92)] * 5, 2, ContentionProfile(512, 0, 0))
    assert result["block_release"] == [0, 0, 100, 100, 200]
    assert result["score"] == 300


def test_readiness_after_a_barrier_controls_shared_transfer_issue():
    delayed = DependencySchedule(ASSUMPTIONS)
    delayed.vf({"cycles": 1000, "counts": {}})
    delayed.barrier([])
    delayed.dma("load", 32)
    fast = DependencySchedule(ASSUMPTIONS)
    fast.dma("load", 32)
    result = schedule([delayed.result(), fast.result()], 2, ContentionProfile(512, 8, 0))
    assert result["block_completion"][1] == 33
    assert result["block_completion"][0] == 1041


def test_zero_resources_and_bad_event_graphs_are_rejected():
    with pytest.raises(Unsupported, match="empty launch"):
        schedule([], 0)
    t = compute_trace(10)
    t["events"][0]["dependencies"] = [(0, 0)]
    with pytest.raises(Unsupported, match="dependency graph"):
        schedule([t], 1)


@pytest.mark.parametrize("field,value", [("dma_issue_interval", -1), ("block_dispatch_interval", float("nan")),
                                         ("global_bytes_per_cycle", 0), ("global_bytes_per_cycle", float("inf")),
                                         ("dma_issue_interval", True)])
def test_invalid_profiles_fail_closed(field, value):
    args = asdict(ContentionProfile())
    args[field] = value
    with pytest.raises(ValueError):
        ContentionProfile(**args)


def test_no_finite_safe_score_means_no_executable_selection():
    candidates = {
        "unmerged": {"resource_checked": False, "status": "STATIC_COST_ESTIMATED", "score":
                     0}, "a0_f2": {"resource_checked": True, "status": "STATIC_COST_ESTIMATED", "score": float("nan")},
        "a0_f4": {"resource_checked": True, "status": "STATIC_ANALYSIS_UNSUPPORTED", "score": 0}
    }
    assert choose(candidates) is None
    candidates["a0_f2"]["score"] = 10
    assert choose(candidates) == "a0_f2"
    candidates["a0_f2"]["historical_us"] = float("inf")
    candidates["unmerged"]["historical_us"] = 0
    assert choose(candidates) == "a0_f2"


def test_unknown_device_data_is_not_interpreted_as_a_loop_bound():
    text = '''
define void @main(ptr addrspace(1) %0) {
  %2 = load i32, ptr addrspace(1) %0
  %3 = icmp sgt i32 %2, 0
  br i1 %3, label %4, label %5
4:
  ret void
5:
  ret void
}
'''
    with pytest.raises(Unsupported, match="data-dependent branch"):
        Analyzer(text, {}, kernel_factory=DependencySchedule).trace("main", [None], 0)
