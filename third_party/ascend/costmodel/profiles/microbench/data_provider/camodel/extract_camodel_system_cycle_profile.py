#!/usr/bin/env python3
"""Convert parsed SIMT CAMODEL counts into SYS_CNT-cycle effective rates.

This does not claim isolated instruction latency.  It reports workload-level
issue rates from active veccores and keeps the source workload in the output.
"""

import argparse
import json
import statistics
from pathlib import Path

GROUPS = ("memory", "float_alu", "shuffle", "predicate", "control", "int_alu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--simulator-clock-mhz", type=float, default=1650.0)
    parser.add_argument("--sys-cnt-mhz", type=float, default=988.9)
    parser.add_argument("--scope", default="unknown")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    active = []
    for name, unit in data.get("per_unit", {}).items():
        span = unit.get("span")
        if not span or span.get("delta", 0) <= 0:
            continue
        active.append((name, float(span["delta"]), unit.get("group_counts", {})))
    if not active:
        raise RuntimeError("no active CAMODEL units with a positive timestamp span")

    wall_simulator_cycles = max(row[1] for row in active)
    clock_ratio = args.simulator_clock_mhz / args.sys_cnt_mhz
    rates = {}
    for group in GROUPS:
        total = sum(float(row[2].get(group, 0)) for row in active)
        aggregate_per_simulator_cycle = total / wall_simulator_cycles
        per_unit_rates = [float(row[2].get(group, 0)) / row[1] for row in active]
        rates[group] = {
            "ops": total,
            "aggregate_ops_per_simulator_cycle": aggregate_per_simulator_cycle,
            "warp_instructions_per_system_cycle": aggregate_per_simulator_cycle * clock_ratio,
            "median_per_unit_ops_per_simulator_cycle": statistics.median(per_unit_rates),
        }

    print(
        json.dumps(
            {
                "unit": "system_cycles",
                "scope": args.scope,
                "source": str(args.input),
                "active_units": len(active),
                "wall_simulator_cycles": wall_simulator_cycles,
                "simulator_clock_mhz": args.simulator_clock_mhz,
                "sys_cnt_mhz": args.sys_cnt_mhz,
                "rates": rates,
                "confidence": "low",
                "note": "Workload-effective CAMODEL issue rates; not isolated peak throughput or latency.",
            }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
