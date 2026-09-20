#!/usr/bin/env python3
"""Matched-only, per-stage-union comparison for the 6 scalar-dominated kernels.

Costmodel side: only the stages that actually executed in the sampled CAModel
program are summed (`matched`).  The costmodel profile stores SYS_CNT-domain
cycles (988.9 MHz by default); pass --sim-mhz/--sys-mhz to convert them back to
CAModel core cycles before comparing with the CAModel union window.
CAModel side: every matched logical stage contributes the union window of its
instructions (first issue -> last retire).  The category total is the sum over
matched stages, so overlapping instructions inside one stage are counted once,
while separate stages remain separate (no cross-stage gap inflation).

Usage:
  python3 summarize_all_scalar.py --base ~/scalar_dominate_eval_seeded \
      --report-dir ~/scalar_dominate_eval_round4/out \
      --csv out/all_scalar_eval.csv
"""
import argparse
import csv
import json
import pathlib
import re
import sys

KERNELS = [
    "padded_copy_gather",
    "padded_copy_scatter",
    "padded_copy_wgrad",
    "binned_copy_gather",
    "binned_copy_scatter",
    "binned_copy_wgrad",
]
CAT_DIRECT = "direct_scalar_load"
CAT_INDIRECT = "indirect_scalar_load"
CAT_STORE = "scalar_store"


def read(path):
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def find_dump(base, kernel, mode):
    log = base / "out" / f"{kernel}_{mode}.run.log"
    hits = re.findall(r"Profiling results saved in (\S+)", read(log))
    if not hits:
        return None
    dump = pathlib.Path(hits[-1]) / "dump"
    return dump if dump.exists() else None


def line_addr(addr):
    return int(addr, 16) & ~0x7F


def line_off(addr):
    # 128B-line-normalised low 12 bits; enough for the allocator layout used
    # by the six-kernel evaluation (indices/bin_ids ...e200/e400, bins/padded
    # bins/weights ...e5xx/...e7xx/...ea00, binned bins ...0200, indices ...fe00).
    return line_addr(addr) & 0xFFF


# ---------------------------------------------------------------------------
# CAModel dump parsers
# ---------------------------------------------------------------------------
def simd_loads(dump):
    instr = read(dump / "core0.veccore0.instr_log.dump")
    issue, retire = {}, {}
    for line in read(dump / "core0.veccore0.ccu.scalar_issque.dump").splitlines():
        m = re.search(r"\[info\]\s+(\d+)\s+\[Push Instr\] name=(\S+) pc=(\S+) id=(\d+)", line)
        if m:
            issue[int(m.group(4))] = int(m.group(1))
            continue
        m = re.search(
            r"\[info\]\s+(\d+)\s+\[RETIRE INSTR\] instr\.name=(\S+)\s+instr\.pc=(\S+)\s+instr\.id=(\d+)",
            line,
        )
        if m:
            retire[int(m.group(4))] = int(m.group(1))
    rows = []
    for line in instr.splitlines():
        m = re.search(
            r"\[(\d+)\] \(PC: (0x[0-9a-fA-F]+)\)\s+(\S+)\s+:\s+"
            r"\(Binary: 0x[0-9a-fA-F]+\)\s+\(ID: (\d+)\)\s+(\S+)(.*)",
            line,
        )
        if not m:
            continue
        iid = int(m.group(4))
        name = m.group(5)
        if not name.startswith(("LD_", "LDP_")):
            continue
        rest = m.group(6)
        am = re.search(r"XN:\w+=(0x[0-9a-fA-F]+)", rest)
        ub = re.search(r"accessUb:(\d+)", rest)
        ddr = re.search(r"accessDdr:(\d+)", rest)
        addr = am.group(1).lower() if am else None
        if not addr or int(addr, 16) < 0x100000000:
            continue
        if (ub and ub.group(1) == "1") or (ddr and ddr.group(1) != "1"):
            continue
        if iid not in issue or iid not in retire:
            continue
        rows.append({
            "id": iid, "issue": issue[iid], "retire": retire[iid], "win": retire[iid] - issue[iid], "name": name,
            "addr": addr
        })
    return sorted(rows, key=lambda r: r["issue"])


def simt_loads(dump):
    events = {}
    for line in read(dump / "core0.veccore0.rvec.simt.lsu.dump").splitlines():
        m = re.search(r"\[(\d+)\] (RECV_INSTR|ISSUE_INSTR|RETIRE_INSTR) name=(\S+) id=(\d+)", line)
        if not m or m.group(3) != "SIMT_LDG":
            continue
        iid = int(m.group(4))
        rec = events.setdefault(iid, {"id": iid, "issue": None, "retire": None})
        if m.group(2) == "ISSUE_INSTR":
            rec["issue"] = int(m.group(1))
        elif m.group(2) == "RETIRE_INSTR":
            rec["retire"] = int(m.group(1))
    dc = {}
    for line in read(dump / "core0.veccore0.rvec.simt.dc.dump").splitlines():
        m = re.search(r"\[(\d+)\]\[TagRam\] C6,.*addr:(0x[0-9a-fA-F]+), size:(\d+),.*isaId:(\d+)", line)
        if m:
            dc[int(m.group(4))] = {"addr": m.group(2).lower(), "size": int(m.group(3))}
    rows = []
    for iid, ev in events.items():
        if ev["issue"] is None or ev["retire"] is None:
            continue
        d = dc.get(iid)
        if not d or d["size"] >= 128:  # scalar only; exclude vector tile LDG
            continue
        rows.append({
            "id": iid, "issue": ev["issue"], "retire": ev["retire"], "win": ev["retire"] - ev["issue"], "addr":
            d["addr"], "size": d["size"]
        })
    return sorted(rows, key=lambda r: r["issue"])


def simd_store(dump):
    pushes, retires = {}, {}
    for line in read(dump / "core0.veccore0.ccu.mte3_issque.dump").splitlines():
        m = re.search(r"\[info\]\s+(\d+)\s+\[Push Instr\] name=(\S+) pc=(\S+) id=(\d+)", line)
        if m:
            pushes[int(m.group(4))] = (int(m.group(1)), m.group(2), m.group(3))
            continue
        m = re.search(
            r"\[info\]\s+(\d+)\s+\[RETIRE INSTR\] instr\.name=(\S+)\s+instr\.pc=(\S+)\s+instr\.id=(\d+)",
            line,
        )
        if m:
            retires[int(m.group(4))] = int(m.group(1))
    rows = []
    for iid, (t, name, pc) in sorted(pushes.items(), key=lambda kv: kv[1][0]):
        if name != "MOV_SRC_TO_DST_ALIGNv2" or iid not in retires:
            continue
        rows.append({"id": iid, "issue": t, "retire": retires[iid], "win": retires[iid] - t, "name": name, "pc": pc})
    return rows


def simt_store(dump):
    events = {}
    for line in read(dump / "core0.veccore0.rvec.simt.lsu.dump").splitlines():
        m = re.search(r"\[(\d+)\] (RECV_INSTR|ISSUE_INSTR|RETIRE_INSTR) name=(\S+) id=(\d+)", line)
        if not m or m.group(3) != "SIMT_STG":
            continue
        iid = int(m.group(4))
        rec = events.setdefault(iid, {"id": iid})
        rec[m.group(2)] = int(m.group(1))
    rows = []
    for iid, rec in events.items():
        if "ISSUE_INSTR" in rec and "RETIRE_INSTR" in rec:
            rows.append({
                "id": iid, "issue": rec["ISSUE_INSTR"], "retire": rec["RETIRE_INSTR"], "win":
                rec["RETIRE_INSTR"] - rec["ISSUE_INSTR"], "name": "SIMT_STG"
            })
    return sorted(rows, key=lambda r: r["issue"])


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def report_stages(report_path, mode):
    d = json.load(open(report_path))
    direct, indirect, stores = [], [], []
    for st in d["stage_model"]["logical_stages"]:
        model = st.get("model")
        if model not in ("scalar_load", "scalar_store"):
            continue
        w = st.get("workload", {})
        cycles = None
        for impl in st["implementations"]:
            im = impl["implementation"]
            if im["mode"] == mode and im["superblock_factor"] == 1:
                rc = impl["resource_system_cycles"]
                cycles = rc.get("load_per_iteration", 0.0) if model == "scalar_load" else rc.get(
                    "store_per_iteration", 0.0)
                break
        if cycles is None:
            continue
        item = {
            "id": st["id"], "cycles": float(cycles), "K": int(w.get("scalar_load_count_per_iteration", 0)), "indirect":
            int(w.get("indirect_scalar_load_count_per_iteration", 0))
        }
        if model == "scalar_store":
            stores.append(item)
        elif item["K"] == 1 and item["indirect"] == 0:
            direct.append(item)
        else:
            indirect.append(item)
    return direct, indirect, stores, d.get("decision_kind", "")


def union(rows):
    if not rows:
        return 0.0
    return float(max(r["retire"] for r in rows) - min(r["issue"] for r in rows))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="CAModel eval dir containing out/<kernel>_<mode>.run.log")
    ap.add_argument("--report-dir", default=None, help="dir containing costmodel_<kernel>.json (default: <base>/out)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--mode", default="auto", choices=["auto", "simd", "simt"],
                    help="force implementation mode; default follows decision_kind")
    ap.add_argument("--sim-mhz", type=float, default=1800.0,
                    help="CAModel/simulator core clock for converting profile SYS_CNT cycles back to CAModel cycles")
    ap.add_argument("--sys-mhz", type=float, default=988.9, help="SYS_CNT clock of the costmodel profile")
    args = ap.parse_args()
    base = pathlib.Path(args.base).expanduser()
    report_dir = pathlib.Path(args.report_dir).expanduser() if args.report_dir else base / "out"
    cycle_scale = args.sim_mhz / args.sys_mhz if args.sim_mhz > 0.0 and args.sys_mhz > 0.0 else 1.0
    rows = []

    for k in KERNELS:
        rp = report_dir / f"costmodel_{k}.json"
        if not rp.exists():
            print("missing report", rp, file=sys.stderr)
            continue
        direct_stages, indirect_stages, store_stages, decision = report_stages(rp, None)
        # Determine mode from decision unless explicitly forced.
        mode = args.mode if args.mode != "auto" else ("simd" if decision.startswith("all_simd") else "simt")
        direct_stages, indirect_stages, store_stages, decision = report_stages(rp, mode)
        camodel_mode = "simd" if mode == "simd" else "simt_only"
        dump = find_dump(base, k, camodel_mode)
        if dump is None:
            print("missing CAModel dump", k, camodel_mode, file=sys.stderr)
            continue
        loads = simd_loads(dump) if mode == "simd" else simt_loads(dump)
        if k.startswith("padded_"):
            direct_loads = [r for r in loads if line_off(r["addr"]) in (0x200, 0x400)]
            other_loads = [r for r in loads if line_off(r["addr"]) not in (0x200, 0x400)]
            if k == "padded_copy_scatter":
                stage7_loads = [r for r in other_loads if line_off(r["addr"]) != 0xA00]
                stage11_loads = [r for r in other_loads if line_off(r["addr"]) == 0xA00]
                matched = []
                if direct_loads:
                    matched.append(("direct", direct_stages[:2], direct_loads[:2]))
                if stage7_loads and len(indirect_stages) >= 1:
                    matched.append(("indirect", indirect_stages[:1], stage7_loads))
                if stage11_loads and len(indirect_stages) >= 2:
                    matched.append(("indirect", indirect_stages[1:2], stage11_loads))
            else:
                matched = []
                if direct_loads:
                    matched.append(("direct", direct_stages[:2], direct_loads[:2]))
                if other_loads and indirect_stages:
                    matched.append(("indirect", indirect_stages[:1], other_loads))
        else:
            first_line = line_addr(loads[0]["addr"]) if loads else None
            direct_loads = [r for r in loads if first_line is not None and line_addr(r["addr"]) == first_line]
            indirect_loads = [r for r in loads if first_line is None or line_addr(r["addr"]) != first_line]
            matched = []
            if len(direct_loads) >= 2:
                matched.append(("direct", direct_stages[:2], direct_loads[:2]))
            elif len(direct_loads) == 1:
                matched.append(("direct", direct_stages[-1:], direct_loads))
            if indirect_loads and indirect_stages:
                matched.append(("indirect", indirect_stages[:1], indirect_loads))

        # Group matched entries by category, preserving stage identity.
        cat_rows = {CAT_DIRECT: [], CAT_INDIRECT: []}
        for cat, stages, mrows in matched:
            key = CAT_DIRECT if cat == "direct" else CAT_INDIRECT
            # one measured union per stage
            per_stage = []
            if len(stages) == len(mrows):
                per_stage = [[r] for r in mrows]
            else:
                per_stage = [mrows]
            for st, srows in zip(stages, per_stage):
                cat_rows[key].append((st, srows))

        def add(cat, entries, total_stages, note=""):
            pred_sys = sum(st["cycles"] for st, _ in entries)
            pred_cam = pred_sys * cycle_scale
            meas = sum(union(srows) for _, srows in entries)
            err = round((pred_cam - meas) / meas * 100.0, 1) if meas else ""
            rows.append({
                "kernel": k,
                "mode": mode,
                "category": cat,
                "costmodel_sys_cycles": round(pred_sys, 2),
                "costmodel_camodel_cycles": round(pred_cam, 2),
                "camodel_stage_union": round(meas, 2) if meas else "",
                "error_pct": err,
                "matched_stages": f"{len(entries)}/{total_stages}",
                "note": note,
            })

        direct_note = "" if len(cat_rows[CAT_DIRECT]) == len(direct_stages) else "unmatched stage (control-flow)"
        add(CAT_DIRECT, cat_rows[CAT_DIRECT], len(direct_stages), direct_note)
        add(CAT_INDIRECT, cat_rows[CAT_INDIRECT], len(indirect_stages),
            "matched indirect stages; per-stage union" if cat_rows[CAT_INDIRECT] else "")

        if store_stages:
            store_rows = simd_store(dump) if mode == "simd" else simt_store(dump)
            add(CAT_STORE, [(store_stages[0], store_rows[:1])], len(store_stages),
                "CAModel " + ("MTE3 MOV" if mode == "simd" else "SIMT_STG"))

    out = csv.DictWriter(sys.stdout, fieldnames=list(rows[0].keys()), lineterminator="\n")
    out.writeheader()
    out.writerows(rows)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
