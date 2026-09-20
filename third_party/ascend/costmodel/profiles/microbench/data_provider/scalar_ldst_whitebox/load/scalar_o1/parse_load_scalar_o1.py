#!/usr/bin/env python3
"""Parse the selected CAModel dumps for the single scalar-load whitebox cases.

Input: one directory per case with the selected dump files:
  simd/   -> core0.veccore0.instr_log.dump, core0.veccore0.ccu.scalar_issque.dump,
             core0.veccore0.dcache_log.dump, core0.biu.brif.log.dump
  simtNN/ -> core0.veccore0.rvec.simt.{lsu,dc,ubitf}.dump, core0.biu.brif.log.dump
Output: JSON to stdout (list of per-case stage timelines).
"""
import json
import pathlib
import re
import sys

ADDR = "0x164b6da00"


def ts(line):
    m = re.match(r"\[info\]\s+(\d+)", line)
    return int(m.group(1)) if m else None


def parse_simd(root: pathlib.Path):
    iss = (root / "core0.veccore0.ccu.scalar_issque.dump").read_text().splitlines()
    ld = None
    for line in iss:
        m = re.search(
            r"\[info\]\s+(\d+)\s+\[Push Instr\] name=LD_XD_XN_IMM pc=(0x[0-9A-Fa-f]+) id=(\d+)",
            line,
        )
        if m:
            ld = {"push": int(m.group(1)), "pc": m.group(2).lower(), "id": int(m.group(3))}
            break
    if ld is None:
        raise SystemExit("SIMD: no LD_XD_XN_IMM push found")
    for line in iss:
        m = re.search(
            r"\[info\]\s+(\d+)\s+\[RETIRE INSTR\] instr.name=LD_XD_XN_IMM\s+instr.pc=(0x[0-9A-Fa-f]+)\s+instr.id=(\d+)",
            line,
        )
        if m and int(m.group(3)) == ld["id"]:
            ld["retire"] = int(m.group(1))
            break
    if "retire" not in ld:
        raise SystemExit("SIMD: no LD retire found")

    dc = (root / "core0.veccore0.dcache_log.dump").read_text().splitlines()
    start = None
    for i, line in enumerate(dc):
        if "calc_req_addr" in line and f"req pc:{ld['pc']}" in line:
            start = i
            break
    if start is None:
        raise SystemExit("SIMD: no calc_req_addr for LD")
    end = len(dc)
    for j in range(start + 1, len(dc)):
        if "calc_req_addr" in dc[j]:
            end = j
            break
    ev = {}
    for line in dc[start:end]:
        t = ts(line)
        if t is None:
            continue
        if "calc_req_addr" in line:
            ev["dc_calc"] = t
        if "lookup_tag is MISS" in line:
            ev["dc_miss"] = t
        if "push req to mshr" in line:
            ev["dc_mshr_push"] = t
        if "mshr_req_create_rd_req" in line:
            ev["dc_mshr_rd"] = t
        if "send_rd_biu_req" in line:
            ev["dc_biu_send"] = t
        if "recv_rd_biu_rsp" in line:
            ev["dc_biu_recv"] = t
        if "refill data to cache_ram" in line:
            ev["dc_refill"] = t
        if "retire_mshr_instr" in line and ld["pc"] in line:
            ev["dc_mshr_retire"] = t

    biu = (root / "core0.biu.brif.log.dump").read_text().splitlines()
    sends = []
    for line in biu:
        m = re.search(r"\[info\]\s+(\d+): send_rd_cmd.*addr: (0x[0-9a-fA-F]+).*gid: (\d+)", line)
        if m and m.group(2).lower() == ADDR:
            sends.append((int(m.group(1)), int(m.group(3))))
    biu_ev = {}
    if sends:
        send_ts, gid = min(sends, key=lambda x: abs(x[0] - ev.get("dc_biu_send", x[0])))
        biu_ev["biu_send"] = send_ts
        biu_ev["gid"] = gid
        for line in biu:
            m = re.search(r"\[info\]\s+(\d+): (recv_biu_data|push_biu_data).*gid[:=]\s*(\d+)", line)
            if m and int(m.group(3)) == gid:
                if m.group(2) == "recv_biu_data" and "biu_recv" not in biu_ev:
                    biu_ev["biu_recv"] = int(m.group(1))
                if m.group(2) == "push_biu_data":
                    biu_ev["biu_push"] = int(m.group(1))
    stages = {
        "issue_to_tag": None,
        "tag_to_biu_send": None,
        "biu_fill": None,
        "return_to_retire": None,
    }
    if "dc_calc" in ev and "dc_biu_send" in ev:
        stages["issue_to_tag"] = ev["dc_calc"] - ld["push"]
        stages["tag_to_biu_send"] = ev["dc_biu_send"] - ev["dc_calc"]
    if "dc_biu_send" in ev and "dc_biu_recv" in ev:
        stages["biu_fill"] = ev["dc_biu_recv"] - ev["dc_biu_send"]
    if "dc_biu_recv" in ev and "retire" in ld:
        stages["return_to_retire"] = ld["retire"] - ev["dc_biu_recv"]
    return {
        "case": "simd",
        "total_cycles": ld["retire"] - ld["push"],
        "events": {**ld, **ev, **biu_ev},
        "stages": stages,
    }


def parse_simt(root: pathlib.Path):
    lsu = (root / "core0.veccore0.rvec.simt.lsu.dump").read_text().splitlines()
    ld = None
    for line in lsu:
        m = re.search(r"\[info\]\s+\[(\d+)\]\s+ISSUE_INSTR name=SIMT_LDG id=(\d+)", line)
        if m:
            ld = {"issue": int(m.group(1)), "id": int(m.group(2))}
            break
    if ld is None:
        raise SystemExit("SIMT: no SIMT_LDG issue")
    for line in lsu:
        m = re.search(r"\[info\]\s+\[(\d+)\]\s+RETIRE_INSTR name=SIMT_LDG id=(\d+)", line)
        if m and int(m.group(2)) == ld["id"]:
            ld["retire"] = int(m.group(1))
            break

    dc = (root / "core0.veccore0.rvec.simt.dc.dump").read_text().splitlines()
    dc_ev = {}
    for line in dc:
        m = re.search(
            r"\[(\d+)\]\[TagRam\] C6,.*addr:(0x[0-9a-fA-F]+), size:(\d+),.*tagRst:(\w+), isaId:(\d+)",
            line,
        )
        if m and int(m.group(5)) == ld["id"]:
            dc_ev = {
                "tag": int(m.group(1)),
                "addr": m.group(2).lower(),
                "req_size": int(m.group(3)),
                "tag_result": m.group(4),
            }
            break

    ubitf = (root / "core0.veccore0.rvec.simt.ubitf.dump").read_text().splitlines()
    ub_ev = {}
    for line in ubitf:
        m = re.search(
            r"\[(\d+)\] \[UBITF_RCEV_REQ\] \[DC_MROB_RD\] SIMT_LDG .*size (\d+), thread_count (\d+)",
            line,
        )
        if m:
            ub_ev.update(
                mrob_recv=int(m.group(1)),
                request_size=int(m.group(2)),
                thread_count=int(m.group(3)),
            )
        m = re.search(r"\[(\d+)\] \[GSU_I2_OUT\] \[DC_MROB_RD\].*size (\d+), target_size (\d+)", line)
        if m:
            ub_ev.update(gsu_out=int(m.group(1)), gsu_size=int(m.group(2)), target_size=int(m.group(3)))
        m = re.search(r"\[(\d+)\] \[UBITF_SEND_RSP\] \[DC_MROB_RD\]", line)
        if m and "ubitf_rsp" not in ub_ev:
            ub_ev["ubitf_rsp"] = int(m.group(1))

    addr = dc_ev.get("addr", ADDR)
    biu = (root / "core0.biu.brif.log.dump").read_text().splitlines()
    sends = []
    for line in biu:
        m = re.search(r"\[info\]\s+(\d+): send_rd_cmd.*addr: (0x[0-9a-fA-F]+).*gid: (\d+)", line)
        if m and m.group(2).lower() == addr:
            sends.append((int(m.group(1)), int(m.group(3))))
    biu_ev = {}
    if sends and "tag" in dc_ev:
        send_ts, gid = min([s for s in sends if s[0] >= dc_ev["tag"]], key=lambda x: x[0], default=(None, None))
        if send_ts is not None:
            biu_ev = {"biu_send": send_ts, "gid": gid}
            for line in biu:
                m = re.search(r"\[info\]\s+(\d+): (recv_biu_data|push_biu_data).*gid[:=]\s*(\d+)", line)
                if m and int(m.group(3)) == gid:
                    if m.group(2) == "recv_biu_data" and "biu_recv" not in biu_ev:
                        biu_ev["biu_recv"] = int(m.group(1))
                    if m.group(2) == "push_biu_data" and "biu_push" not in biu_ev:
                        biu_ev["biu_push"] = int(m.group(1))

    ev = {**ld, **dc_ev, **ub_ev, **biu_ev}
    stages = {
        "issue_to_tag": None,
        "tag_to_biu_send": None,
        "biu_fill": None,
        "biu_return_to_mrob": None,
        "mrob_to_ubitf_rsp": None,
        "ubitf_rsp_to_retire": None,
    }
    if "tag" in ev:
        stages["issue_to_tag"] = ev["tag"] - ev["issue"]
    if "tag" in ev and "biu_send" in ev:
        stages["tag_to_biu_send"] = ev["biu_send"] - ev["tag"]
    if "biu_send" in ev and "biu_recv" in ev:
        stages["biu_fill"] = ev["biu_recv"] - ev["biu_send"]
    if "biu_recv" in ev and "mrob_recv" in ev:
        stages["biu_return_to_mrob"] = ev["mrob_recv"] - ev["biu_recv"]
    if "mrob_recv" in ev and "ubitf_rsp" in ev:
        stages["mrob_to_ubitf_rsp"] = ev["ubitf_rsp"] - ev["mrob_recv"]
    if "ubitf_rsp" in ev and "retire" in ev:
        stages["ubitf_rsp_to_retire"] = ev["retire"] - ev["ubitf_rsp"]
    return {
        "case": root.name,
        "total_cycles": ev["retire"] - ev["issue"],
        "events": ev,
        "stages": stages,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <case-dir> [case-dir ...]")
    out = []
    for arg in sys.argv[1:]:
        root = pathlib.Path(arg)
        if (root / "core0.veccore0.ccu.scalar_issque.dump").exists():
            out.append(parse_simd(root))
        else:
            out.append(parse_simt(root))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
