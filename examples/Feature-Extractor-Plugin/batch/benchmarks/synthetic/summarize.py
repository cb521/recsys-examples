#!/usr/bin/env python3
"""Combine run_bench.py JSON (wall-clock medians) with the event log (operator-level op time) into the RESULTS tables.
  python3 bench/summarize.py --json bench/out/bench1.json --eventlog bench/out/evlog1 [--md out.md]
"""
import sys, os, json, re, argparse, statistics, collections
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "advisor"))
import fusion_advisor as A

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--json", required=True); ap.add_argument("--eventlog", required=True); ap.add_argument("--md", default="")
    a = ap.parse_args()
    res = json.load(open(a.json)); execs, acc = A.parse([a.eventlog])
    per = collections.defaultdict(list)   # (query,cfg) -> list of dict(rep metrics)
    for ex in execs.values():
        m = re.match(r"^(\w+)\|([\w@]+)\|rep\d+$", ex["desc"])
        if not m or ex["end"] is None: continue
        cfg, q = m.group(1), m.group(2)
        r = A.analyse_exec(ex, acc, 10, 864, 0.6, 8)
        per[(q, cfg)].append(r)
    cfgs = res["args"]["configs"].split(",")
    L = ["| 查询 | 类型 | " + " | ".join(f"{c} 中位(s)" for c in cfgs) + " | " + " | ".join(f"{c} vs gpu" for c in cfgs if c != "gpu") +
         " | gpu: P/F op time(s) | fused: P/F op time(s) | gpu: P/F 占 GPU op time | gpu: P/F 占总时长 | 融合进计划(Project/Filter) |", "|" + "---|" * (2 + len(cfgs) + len(cfgs) - 1 + 5)]
    for q, rec in res["queries"].items():
        if "error" in rec: L.append(f"| {q} | {rec['kind']} | 出错: {rec['error'][:80]} |"); continue
        med = {c: rec["cfg"][c]["median"] for c in cfgs}
        def pf(cfg, key):
            v = [r[key] for r in per.get((q, cfg), [])]; return statistics.median(v) if v else float("nan")
        row = [q, rec["kind"]] + [f"{med[c]:.2f}" for c in cfgs]
        row += [f"{(med['gpu']/med[c]-1)*100:+.1f}%" for c in cfgs if c != "gpu"]
        row += [f"{pf('gpu','pf_op_time_s'):.2f}", f"{pf('fused','pf_op_time_s'):.2f}" if "fused" in cfgs else "-",
                f"{pf('gpu','pf_share_of_gpu_op_time_pct'):.0f}%", f"{pf('gpu','pf_share_of_wall_pct'):.0f}%"]
        fl = rec["cfg"].get("fused", {}).get("plan_flags", {})
        row.append(("P" if fl.get("fused_project") else "-") + ("/F" if fl.get("fused_filter") else "/-"))
        L.append("| " + " | ".join(row) + " |")
    out = "\n".join(L) + "\n"
    if a.md: open(a.md, "w").write(out)
    print(out)

if __name__ == "__main__":
    main()
