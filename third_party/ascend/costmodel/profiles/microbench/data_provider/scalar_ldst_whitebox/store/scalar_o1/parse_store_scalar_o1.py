#!/usr/bin/env python3
"""Extract the single-op scalar-store event timeline / stage cycles from CAModel dumps.

Usage:
  python3 parse_store_scalar_o1.py camodel_results/simd camodel_results/simt32
Output: JSON list, one object per case.
"""
import json
import pathlib
import re
import sys

ADDR_MAIN = "0x164b6da00"


def ts(line):
    m = re.match(r"\[info\]\s+\[?(\d+)", line)
    return int(m.group(1)) if m else None


def parse_simd(root: pathlib.Path):
    ins = (root / "core0.veccore0.instr_log.dump").read_text().splitlines()
    sq = (root / "core0.veccore0.ccu.scalar_issque.dump").read_text().splitlines()
    dc = (root / "core0.veccore0.dcache_log.dump").read_text().splitlines()
    brif = (root / "core0.biu.brif.log.dump").read_text().splitlines()
    bwif = (root / "core0.biu.bwif.log.dump").read_text().splitlines()

    st = None
    for line in ins:
        m = re.search(
            r"\(PC: (0x[0-9a-fA-F]+)\).*\(ID: (\d+)\).*ST_XD_XN_IMM.*XN:X0=(0x[0-9a-fA-F]+).*execTime:(0x[0-9a-fA-F]+)",
            line,
        )
        if m and m.group(3).lower() == ADDR_MAIN:
            st = {"pc": m.group(1).lower(), "id": int(m.group(2)), "execTime": int(m.group(4), 16)}
            break
    if st is None:
        raise SystemExit("no target ST found")
    for line in sq:
        m = re.search(r"\[info\]\s+(\d+) \[Push Instr\] name=ST_XD_XN_IMM pc=(0x[0-9a-fA-F]+) id=(\d+)", line)
        if m and m.group(2).lower() == st["pc"] and int(m.group(3)) == st["id"]:
            st["issue"] = int(m.group(1))
        m = re.search(
            r"\[info\]\s+(\d+) \[RETIRE INSTR\] instr.name=ST_XD_XN_IMM\s+instr.pc=(0x[0-9a-fA-F]+)\s+instr.id=(\d+)",
            line)
        if m and m.group(2).lower() == st["pc"] and int(m.group(3)) == st["id"]:
            st["retire"] = int(m.group(1))

    ev = {}
    for line in dc:
        t = ts(line)
        if t is None:
            continue
        if "calc_req_addr" in line and f"req pc:{st['pc']}" in line:
            ev["dc_calc"] = t
        if "lookup_tag is MISS" in line and f"req pc :{st['pc']}" in line:
            ev["dc_miss"] = t
        if "push req to stb" in line and f"req_pc:{st['pc']}" in line:
            ev["stb_push"] = t
        if "stb_create_rd_req" in line and f"req_pc:{st['pc']}" in line:
            ev["stb_create_rd"] = t
    # first read req after target STB create
    if "stb_create_rd" in ev:
        for line in dc:
            t = ts(line)
            if t is not None and t >= ev["stb_create_rd"] and "send_rd_biu_req" in line:
                m = re.search(r"req_id:(\d+)", line)
                ev["dc_biu_req"] = t
                ev["read_id"] = int(m.group(1))
                break
    if "read_id" in ev:
        rid = ev["read_id"]
        for line in dc:
            t = ts(line)
            if f"rsp_id:{rid}" in line and "recv_rd_biu_rsp(stb)" in line:
                ev["dc_biu_rsp"] = t
            if f"isa_pc:{st['pc']}" in line and "retire_stb_instr" in line:
                ev["stb_retire"] = t
        for line in brif:
            m = re.search(r"\[info\]\s+(\d+): (send_rd_cmd|recv_biu_data|push_biu_data).*gid:?\s*(\d+)", line)
            if m and int(m.group(3)) == rid:
                ev["biu_" + m.group(2)] = int(m.group(1))
    # writeback (after retire)
    for line in dc:
        t = ts(line)
        if t is None:
            continue
        if "create_write_data_to_ddr_req" in line and ADDR_MAIN in line:
            ev["create_wr_ddr"] = t
        if "send_wr_biu_req" in line and ADDR_MAIN in line:
            m = re.search(r"ll_req_id:(\d+)", line)
            ev["dc_wr_req"] = t
            ev["wr_id"] = int(m.group(1))
        if "send_wr_biu_data_req" in line and "wr_id" in ev and f"ll_req_id:{ev['wr_id']}" in line:
            ev["dc_wr_data"] = t
    if "wr_id" in ev:
        for line in bwif:
            m = re.search(
                r"\[info\] \[(\d+)\]: (recv_su_wr_cmd|send_aw_cmd|send_wr_cmd_rsp|recv_wr_data|write_store_buf|recv_wack|send_wr_data|recv_brsp|send_data_rsp).*gid:\s*(\d+)",
                line)
            if m and int(m.group(3)) == ev["wr_id"]:
                ev["wr_" + m.group(2)] = int(m.group(1))

    stages = {
        "issue_to_tag": ev["dc_calc"] - st["issue"],
        "tag_to_stb_push": ev["stb_push"] - ev["dc_calc"],
        "stb_push_to_read_req": ev["stb_create_rd"] - ev["stb_push"],
        "read_req_to_biu_req": ev["dc_biu_req"] - ev["stb_create_rd"],
        "biu_req_to_dc_rsp": ev["dc_biu_rsp"] - ev["dc_biu_req"],
        "dc_rsp_to_retire": st["retire"] - ev["dc_biu_rsp"],
    }
    return {
        "case": root.name,
        "total_cycles": st["retire"] - st["issue"],
        "execTime": st["execTime"],
        "events": {"issue": st["issue"], "retire": st["retire"], **ev},
        "stages": stages,
    }


def parse_simt(root: pathlib.Path):
    lsu = (root / "core0.veccore0.rvec.simt.lsu.dump").read_text().splitlines()
    dc = (root / "core0.veccore0.rvec.simt.dc.dump").read_text().splitlines()
    ub = (root / "core0.veccore0.rvec.simt.ubitf.dump").read_text().splitlines()
    bwif = (root / "core0.biu.bwif.log.dump").read_text().splitlines()

    ev = {}
    st_id = None
    for line in lsu:
        m = re.search(r"\[(\d+)\] (RECV|ISSUE|RETIRE)_INSTR name=SIMT_STG id=(\d+)", line)
        if m:
            d = m.group(2).lower()
            if st_id is None:
                st_id = int(m.group(3))
            if int(m.group(3)) == st_id:
                ev[d] = int(m.group(1))
    if st_id is None:
        raise SystemExit("no SIMT_STG found")
    for line in dc:
        m = re.search(r"\[(\d+)\]\[TagRam\] C6,.*cmd:WRITE.*addr:(0x[0-9a-fA-F]+).*isaId:(\d+)", line)
        if m and int(m.group(3)) == st_id:
            ev["tag"] = int(m.group(1))
            ev["addr"] = m.group(2).lower()
            break
    for line in ub:
        m = re.search(r"\[(\d+)\] \[UBITF_RCEV_REQ\] \[DC_BPSQ_WR\] SIMT_STG .*thread_count (\d+)", line)
        if m and "mrob_recv" not in ev:
            ev["ub_recv"] = int(m.group(1))
            ev["thread_count"] = int(m.group(2))
        m = re.search(r"\[(\d+)\] \[GSU_I2_OUT\] \[DC_BPSQ_WR\].*target_size (\d+)", line)
        if m and "gsu_out" not in ev:
            ev["gsu_out"] = int(m.group(1))
            ev["target_size"] = int(m.group(2))
        m = re.search(r"\[(\d+)\] \[UBITF_SEND_RSP\] \[DC_BPSQ_WR\]", line)
        if m and "ubitf_rsp" not in ev:
            ev["ubitf_rsp"] = int(m.group(1))
    # pick first vdcache0 write for target addr after UBITF rsp
    gid = None
    for line in bwif:
        m = re.search(r"\[(\d+)\]: recv_simt_wr_cmd, port: (vdcache\d+), addr: (0x[0-9a-fA-F]+).*gid:\s*(\d+)", line)
        if m and m.group(3).lower() == ADDR_MAIN and int(m.group(1)) >= ev.get("ubitf_rsp", 0):
            if m.group(2) == "vdcache0" or gid is None:
                gid = int(m.group(4))
                ev["biu_recv_cmd"] = int(m.group(1))
                if m.group(2) == "vdcache0":
                    break
    if gid is not None:
        for line in bwif:
            m = re.search(
                r"\[(\d+)\]\s*:?\s*(send_aw_cmd|send_wr_cmd_rsp|recv_wr_data|write_store_buf|recv_wack|send_wr_data|recv_brsp|send_data_rsp).*gid:\s*(\d+)",
                line)
            if m and int(m.group(3)) == gid:
                ev["biu_" + m.group(2)] = int(m.group(1))
    stages = {
        "issue_to_tag": ev["tag"] - ev["issue"],
        "tag_to_ubitf_rsp": ev["ubitf_rsp"] - ev["tag"],
        "ubitf_rsp_to_biu_cmd": ev["biu_recv_cmd"] - ev["ubitf_rsp"],
        "biu_cmd_to_data_buf": ev["biu_write_store_buf"] - ev["biu_recv_cmd"],
        "data_buf_to_wack": ev["biu_recv_wack"] - ev["biu_write_store_buf"],
        "wack_to_data_rsp": ev["biu_send_data_rsp"] - ev["biu_recv_wack"],
        "data_rsp_to_retire": ev["retire"] - ev["biu_send_data_rsp"],
    }
    return {
        "case": root.name,
        "total_cycles": ev["retire"] - ev["issue"],
        "thread_count": ev.get("thread_count"),
        "target_size": ev.get("target_size"),
        "events": ev,
        "stages": stages,
    }


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <simd_dir> [simt_dir ...]")
    out = []
    for arg in sys.argv[1:]:
        root = pathlib.Path(arg)
        out.append(parse_simd(root) if (root / "core0.veccore0.ccu.scalar_issque.dump").exists() else parse_simt(root))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
