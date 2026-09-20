#!/usr/bin/env python3
"""Parse CAModel dumps for the 4-op scalar-load whitebox cases.

Input: one flat OPPROF dump directory per case (--launch-count=1 layout):
  simd cases -> core0.veccore0.{instr_log,ccu.scalar_issque,dcache_log}.dump
  simt cases -> core0.veccore0.rvec.simt.{lsu,dc,ubitf}.dump
Common -> core0.biu.brif.log.dump
Output: JSON to stdout.
"""
import json
import pathlib
import re
import sys


def ts(line):
    m = re.match(r"\[info\]\s+(\d+)", line)
    return int(m.group(1)) if m else None


def parse_simd(root: pathlib.Path):
    iss = (root / "core0.veccore0.ccu.scalar_issque.dump").read_text().splitlines()
    pushes = []
    for line in iss:
        m = re.search(
            r"\[info\]\s+(\d+)\s+\[Push Instr\] name=LD_XD_XN_IMM pc=(0x[0-9A-Fa-f]+) id=(\d+)",
            line,
        )
        if m:
            pushes.append({"push": int(m.group(1)), "pc": m.group(2).lower(), "id": int(m.group(3))})
    retires = {}
    for line in iss:
        m = re.search(
            r"\[info\]\s+(\d+)\s+\[RETIRE INSTR\] instr.name=LD_XD_XN_IMM\s+instr.pc=(0x[0-9A-Fa-f]+)\s+instr.id=(\d+)",
            line,
        )
        if m:
            retires[int(m.group(3))] = int(m.group(1))
    instr = {}
    for line in (root / "core0.veccore0.instr_log.dump").read_text().splitlines():
        m = re.search(
            r"\[info\]\s+\[(\d+)\].*\(ID: (\d+)\) LD_XD_XN_IMM.*execTime:(0x[0-9a-fA-F]+), dcacheHit:(\d+)",
            line,
        )
        if m:
            instr[int(m.group(2))] = {
                "instr_log_time": int(m.group(1)),
                "exec_time": int(m.group(3), 16),
                "dcache_hit": int(m.group(4)),
            }
    ops = []
    for p in pushes:
        op = {**p, "retire": retires.get(p["id"]), **instr.get(p["id"], {})}
        if op["retire"] is not None:
            op["latency"] = op["retire"] - op["push"]
        ops.append(op)

    dc = (root / "core0.veccore0.dcache_log.dump").read_text().splitlines()
    events = []
    for line in dc:
        t = ts(line)
        if t is None:
            continue
        if "calc_req_addr" in line:
            m = re.search(r"req pc:(0x[0-9a-fA-F]+).*aligned_addr_:(0x[0-9a-fA-F]+)", line)
            if m:
                events.append({"time": t, "pc": m.group(1).lower(), "event": "calc", "addr": m.group(2).lower()})
        elif "lookup_tag is" in line:
            m = re.search(r"lookup_tag is (MISS|HIT)\. req pc :(0x[0-9a-fA-F]+)", line)
            if m:
                events.append({"time": t, "pc": m.group(2).lower(), "event": m.group(1)})
        elif "push req to mshr" in line:
            m = re.search(r"req_addr:(0x[0-9a-fA-F]+),mshr_main_entry_id:(\d+)", line)
            if m:
                events.append({"time": t, "event": "mshr_push", "addr": m.group(1).lower(), "entry": int(m.group(2))})
        elif "send_rd_biu_req" in line:
            m = re.search(r"req_id:(\d+),req_rd_addr:(0x[0-9a-fA-F]+)", line)
            if m:
                events.append({"time": t, "event": "biu_req", "gid": int(m.group(1)), "addr": m.group(2).lower()})
        elif "recv_rd_biu_rsp" in line:
            m = re.search(r"rsp_id:(\d+), aligned_addr: (0x[0-9a-fA-F]+)", line)
            if m:
                events.append({"time": t, "event": "biu_rsp", "gid": int(m.group(1)), "addr": m.group(2).lower()})
        elif "refill data" in line:
            m = re.search(r"tag_addr:(0x[0-9a-fA-F]+),idx_addr: (0x[0-9a-fA-F]+)", line)
            if m:
                events.append({"time": t, "event": "refill", "tag": m.group(1), "idx": m.group(2)})
        elif "retire_mshr_instr" in line:
            m = re.search(r"isa_pc:(0x[0-9a-fA-F]+)", line)
            if m:
                events.append({"time": t, "pc": m.group(1).lower(), "event": "mshr_retire"})
    ld_pcs = {p["pc"] for p in pushes}
    ld_addrs = {e["addr"] for e in events if e["event"] == "calc" and e.get("pc") in ld_pcs}
    return {
        "case": root.name,
        "mode": "simd",
        "first_issue": ops[0]["push"] if ops else None,
        "last_retire": max(o["retire"] for o in ops if o["retire"] is not None) if ops else None,
        "ops": ops,
        "dc_events": events,
        "summary": {
            "line_count": len(ld_addrs),
            "miss_count": sum(1 for o in ops if o.get("dcache_hit") == 0),
            "hit_count": sum(1 for o in ops if o.get("dcache_hit") == 1),
            "biu_read_count": sum(1 for e in events if e["event"] == "biu_req" and e["addr"] in ld_addrs),
            "biu_line_count": len({e["addr"]
                                   for e in events
                                   if e["event"] == "biu_req" and e["addr"] in ld_addrs}),
        },
    }


def parse_simt(root: pathlib.Path):
    lsu_lines = (root / "core0.veccore0.rvec.simt.lsu.dump").read_text().splitlines()
    ldgs = {}
    for line in lsu_lines:
        m = re.search(r"\[info\]\s+\[(\d+)\]\s+(RECV|ISSUE|RETIRE)_INSTR name=SIMT_LDG id=(\d+)", line)
        if m:
            ldg = ldgs.setdefault(int(m.group(3)), {})
            ldg[m.group(2).lower()] = int(m.group(1))
    order = sorted(ldgs, key=lambda i: ldgs[i].get("issue", 1 << 60))

    dc_lines = (root / "core0.veccore0.rvec.simt.dc.dump").read_text().splitlines()
    tag_by_id = {}
    for line in dc_lines:
        m = re.search(
            r"\[(\d+)\]\[TagRam\] C6,.*addr:(0x[0-9a-fA-F]+), size:(\d+),.*tagRst:(\w+), isaId:(\d+)",
            line,
        )
        if m:
            tag_by_id[int(m.group(5))] = {
                "tag": int(m.group(1)),
                "addr": m.group(2).lower(),
                "size": int(m.group(3)),
                "tag_result": m.group(4),
            }

    ub_lines = (root / "core0.veccore0.rvec.simt.ubitf.dump").read_text().splitlines()
    ub_by_id = {}
    for line in ub_lines:
        m = re.search(
            r"\[(\d+)\] \[UBITF_RCEV_REQ\] \[DC_MROB_RD\] SIMT_LDG .*thread_count (\d+).*isa_id (\d+)",
            line,
        )
        if m:
            ub_by_id.setdefault(int(m.group(3)), {})["mrob_recv"] = int(m.group(1))
            ub_by_id[int(m.group(3))]["thread_count"] = int(m.group(2))
        m = re.search(r"\[(\d+)\] \[GSU_I2_OUT\] \[DC_MROB_RD\].*size (\d+), target_size (\d+)", line)
        if m:
            # assign to the LDG whose mrob_recv is the latest seen before this line
            candidates = [i for i, v in ub_by_id.items() if v.get("mrob_recv", 1 << 60) <= int(m.group(1))]
            if candidates:
                i = max(candidates, key=lambda x: ub_by_id[x].get("mrob_recv", -1))
                ub_by_id[i]["gsu_out"] = int(m.group(1))
                ub_by_id[i]["gsu_size"] = int(m.group(2))
                ub_by_id[i]["target_size"] = int(m.group(3))
        m = re.search(r"\[(\d+)\] \[UBITF_SEND_RSP\] \[DC_MROB_RD\].*isa_id (\d+)", line)
        if m:
            ub_by_id.setdefault(int(m.group(2)), {})["ubitf_rsp"] = int(m.group(1))

    biu_lines = (root / "core0.biu.brif.log.dump").read_text().splitlines()
    sends = []
    recvs = {}
    for line in biu_lines:
        m = re.search(r"\[info\]\s+(\d+): send_rd_cmd, port: (\w+).*addr: (0x[0-9a-fA-F]+).*gid: (\d+)", line)
        if m:
            sends.append({
                "time": int(m.group(1)),
                "port": m.group(2),
                "addr": m.group(3).lower(),
                "gid": int(m.group(4)),
            })
        m = re.search(r"\[info\]\s+(\d+): recv_biu_data.*gid[:=]\s*(\d+)", line)
        if m:
            recvs[int(m.group(2))] = int(m.group(1))

    used_gids = set()
    ops = []
    for i in order:
        ldg = ldgs[i]
        tag = tag_by_id.get(i, {})
        # match the earliest unused BIU send for this address after the tag
        cand = [
            s for s in sends if s.get("port", "").startswith("vicache0") and s["gid"] not in used_gids
            and s["addr"] == tag.get("addr") and s["time"] >= tag.get("tag", 0)
        ]
        send = min(cand, key=lambda s: s["time"]) if cand else None
        if send:
            used_gids.add(send["gid"])
        op = {
            "id": i,
            "issue": ldg.get("issue"),
            "retire": ldg.get("retire"),
            "latency": (ldg.get("retire") - ldg.get("issue")) if "retire" in ldg and "issue" in ldg else None,
            **tag,
            **ub_by_id.get(i, {}),
        }
        if send:
            op["biu_send"] = send["time"]
            op["biu_gid"] = send["gid"]
            op["biu_recv"] = recvs.get(send["gid"])
            op["biu_fill"] = (op["biu_recv"] - send["time"]) if op.get("biu_recv") is not None else None
        ops.append(op)

    first_issue = min(ldgs[i]["issue"] for i in ldgs if "issue" in ldgs[i])
    last_retire = max(ldgs[i]["retire"] for i in ldgs if "retire" in ldgs[i])
    return {
        "case": root.name,
        "mode": "simt",
        "first_issue": first_issue,
        "last_retire": last_retire,
        "ops": ops,
        "summary": {
            "line_count": len({o["addr"]
                               for o in ops
                               if o.get("addr")}),
            "tag_invalid": sum(1 for o in ops if o.get("tag_result") == "INVALID"),
            "biu_read_count": len([o for o in ops if "biu_gid" in o]),
            "biu_fill_values": [o.get("biu_fill") for o in ops],
        },
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <dump-dir> [dump-dir ...]")
    out = []
    for arg in sys.argv[1:]:
        root = pathlib.Path(arg)
        lsu = root / "core0.veccore0.rvec.simt.lsu.dump"
        if lsu.exists() and "SIMT_LDG" in lsu.read_text():
            out.append(parse_simt(root))
        else:
            out.append(parse_simd(root))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
