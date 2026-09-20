#!/usr/bin/env python3
"""Extract 4-op scalar-store event lists / windows from CAModel dumps.

Usage:
  python3 parse_store_scalar_o4.py camodel_results/simd_same camodel_results/simd_diff \
      camodel_results/simt_same camodel_results/simt_diff
Output: JSON list, one object per case.
"""
import json
import pathlib
import re
import sys

ADDR_MAIN = "0x164b6da00"
TARGET_ADDRS = {
    "0x164b6da00", "0x164b6da40", "0x164b6da80", "0x164b6dac0",  # SIMD diff 64B lines
    "0x164b6da00", "0x164b6da80", "0x164b6db00", "0x164b6db80",  # SIMT diff 128B lines
}


def ts(line):
    m = re.match(r"\[info\]\s+\[?(\d+)", line)
    return int(m.group(1)) if m else None


def parse_simd(root: pathlib.Path):
    ins = (root / "core0.veccore0.instr_log.dump").read_text().splitlines()
    sq = (root / "core0.veccore0.ccu.scalar_issque.dump").read_text().splitlines()
    dc = (root / "core0.veccore0.dcache_log.dump").read_text().splitlines()
    ops = []
    for line in ins:
        m = re.search(
            r"\(PC: (0x[0-9a-fA-F]+)\).*\(ID: (\d+)\).*ST_XD_XN_IMM.*XN:X0=(0x[0-9a-fA-F]+).*execTime:(0x[0-9a-fA-F]+)",
            line,
        )
        if m and m.group(3).lower() == ADDR_MAIN:
            ops.append({"pc": m.group(1).lower(), "id": int(m.group(2)), "execTime": int(m.group(4), 16)})
    for op in ops:
        for line in sq:
            m = re.search(r"\[info\]\s+(\d+) \[Push Instr\] name=ST_XD_XN_IMM pc=(0x[0-9a-fA-F]+) id=(\d+)", line)
            if m and m.group(2).lower() == op["pc"] and int(m.group(3)) == op["id"]:
                op["issue"] = int(m.group(1))
            m = re.search(
                r"\[info\]\s+(\d+) \[RETIRE INSTR\] instr.name=ST_XD_XN_IMM\s+instr.pc=(0x[0-9a-fA-F]+)\s+instr.id=(\d+)",
                line)
            if m and m.group(2).lower() == op["pc"] and int(m.group(3)) == op["id"]:
                op["retire"] = int(m.group(1))
    for op in ops:
        ev = {}
        for line in dc:
            t = ts(line)
            if t is None:
                continue
            if "calc_req_addr" in line and f"req pc:{op['pc']}" in line:
                ev["calc"] = t
                m = re.search(r"aligned_addr_:(0x[0-9a-fA-F]+)", line)
                if m:
                    op["line_addr"] = m.group(1).lower()
            if "push req to stb" in line and f"req_pc:{op['pc']}" in line:
                ev["stb_push"] = t
            if "stb_create_rd_req" in line and f"req_pc:{op['pc']}" in line:
                ev["stb_create_rd"] = t
        if "stb_create_rd" in ev:
            for line in dc:
                t = ts(line)
                if t is not None and t >= ev["stb_create_rd"] and "send_rd_biu_req" in line:
                    m = re.search(r"req_id:(\d+)", line)
                    ev["biu_req"] = t
                    ev["read_id"] = int(m.group(1))
                    break
        if "read_id" in ev:
            for line in dc:
                if f"rsp_id:{ev['read_id']}" in line and "recv_rd_biu_rsp(stb)" in line:
                    ev["biu_rsp"] = ts(line)
                if f"isa_pc:{op['pc']}" in line and "retire_stb_instr" in line:
                    ev["stb_retire"] = ts(line)
        op["events"] = ev
    return {
        "case": root.name,
        "window": (max(o["retire"] for o in ops) - min(o["issue"] for o in ops)) if ops else None,
        "ops": ops,
    }


def parse_simt(root: pathlib.Path):
    lsu = (root / "core0.veccore0.rvec.simt.lsu.dump").read_text().splitlines()
    dc = (root / "core0.veccore0.rvec.simt.dc.dump").read_text().splitlines()
    ub = (root / "core0.veccore0.rvec.simt.ubitf.dump").read_text().splitlines()
    bw = (root / "core0.biu.bwif.log.dump").read_text().splitlines()
    ops = []
    cur = None
    for line in lsu:
        m = re.search(r"\[(\d+)\] (RECV|ISSUE|RETIRE)_INSTR name=SIMT_STG id=(\d+)", line)
        if m:
            if m.group(2) == "RECV":
                cur = {"id": int(m.group(3)), "recv": int(m.group(1))}
                ops.append(cur)
            else:
                for op in ops:
                    if op["id"] == int(m.group(3)):
                        op[m.group(2).lower()] = int(m.group(1))
                        break
    for op in ops:
        for line in dc:
            m = re.search(r"\[(\d+)\]\[TagRam\] C6,.*cmd:WRITE.*addr:(0x[0-9a-fA-F]+).*isaId:(\d+)", line)
            if m and int(m.group(3)) == op["id"]:
                op["tag"] = int(m.group(1))
                op["addr"] = m.group(2).lower()
                break
        for line in ub:
            m = re.search(
                r"\[(\d+)\] \[UBITF_RCEV_REQ\] \[DC_BPSQ_WR\] SIMT_STG .*addr (0x[0-9a-fA-F]+).*thread_count (\d+)",
                line)
            if m and int(m.group(1)) >= op.get("tag", 0) and int(m.group(3)) == 32 and "ub_addr" not in op:
                # note: ubitf lines carry isa_id, check it explicitly
                mi = re.search(r"isa_id:(\d+)", line)
                if mi and int(mi.group(1)) == op["id"]:
                    op["ub_addr"] = m.group(2)
                    op["thread_count"] = int(m.group(3))
                    op["ub_recv"] = int(m.group(1))
    writes = []
    for line in bw:
        m = re.search(r"\[(\d+)\]\s*:?\s*recv_simt_wr_cmd, port: (vdcache\d+), addr: (0x[0-9a-fA-F]+).*gid:\s*(\d+)",
                      line)
        if m and m.group(3).lower() in TARGET_ADDRS:
            writes.append(
                {"time": int(m.group(1)), "port": m.group(2), "addr": m.group(3).lower(), "gid": int(m.group(4))})
    return {
        "case": root.name,
        "window": (max(o["retire"] for o in ops) - min(o["issue"] for o in ops)) if ops else None,
        "ops": ops,
        "target_biu_writes": writes,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <case_dir> [case_dir ...]")
    out = []
    for arg in sys.argv[1:]:
        root = pathlib.Path(arg)
        out.append(parse_simd(root) if (root / "core0.veccore0.ccu.scalar_issque.dump").exists() else parse_simt(root))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
