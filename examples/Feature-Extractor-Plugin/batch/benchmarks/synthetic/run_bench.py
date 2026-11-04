#!/usr/bin/env python3
"""Benchmark runner: same query under several configs (one Spark session; configs are runtime conf toggles).

  gpu-run.sh -- python3 bench/run_bench.py --data DIR --out OUT.json --configs gpu,ast,fused --reps 5 --warmup 2 [--queries E1_wide_project,...]
Configs: cpu | gpu (spark-rapids, fusion off) | ast (projectAstEnabled) | fused (fusion plugin on)
Timing = wall clock of a noop-sink action, median of --reps (reps are interleaved across configs to cancel drift).
Each run carries job description "cfg|query|repN" and an event log is written (--eventlog DIR) for advisor/summarise.
"""
import sys, os, json, time, argparse, statistics
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests", "fz")); sys.path.insert(0, HERE)
from common import *
import queries as Q

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--configs", default="gpu,ast,fused"); ap.add_argument("--queries", default="all")
    ap.add_argument("--reps", type=int, default=5); ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--eventlog", default=""); ap.add_argument("--pool-frac", default="0.7")
    ap.add_argument("--no-aqe", action="store_true"); ap.add_argument("--local-n", type=int, default=4)
    ap.add_argument("--write-variants", action="store_true", help="also time parquet-write sink for E1/E2")
    a = ap.parse_args()
    cfgs = a.configs.split(",")
    os.environ["RAPIDS_POOL_FRAC"] = a.pool_frac
    extra = {"spark.rapids.sql.concurrentGpuTasks": "3", "spark.sql.files.maxPartitionBytes": "256m"}
    if a.eventlog:
        os.makedirs(a.eventlog, exist_ok=True)
        extra.update({"spark.eventLog.enabled": "true", "spark.eventLog.dir": "file://" + os.path.abspath(a.eventlog)})
    sp = build_session(use_gpu=True, local_n=a.local_n, driver_mem="40g", extra=extra, aqe=not a.no_aqe)
    sp.conf.set("spark.sql.shuffle.partitions", "32")
    for t in ["wide", "lineitem", "orders", "customer", "part", "supplier", "nation", "region"]:
        p = os.path.join(a.data, t)
        if os.path.exists(p): sp.read.parquet(p).createOrReplaceTempView(t)
    names = list(Q.ALL) if a.queries == "all" else a.queries.split(",")
    res = {"args": vars(a), "queries": {}}
    sc = sp.sparkContext
    def run(cfg, name, tag, sink):
        set_mode(sp, cfg)
        sc.setJobDescription(f"{cfg}|{name}|{tag}")
        df = sp.sql(Q.ALL[name])
        t0 = time.perf_counter()
        if sink == "noop": df.write.format("noop").mode("overwrite").save()
        else: df.write.mode("overwrite").parquet(os.path.join(a.out + ".tmp_out"))
        return time.perf_counter() - t0, df
    todo = [(n, "noop") for n in names] + ([(n, "write") for n in Q.WRITE_VARIANTS if n in names] if a.write_variants else [])
    for name, sink in todo:
        key = name if sink == "noop" else name + "@write"
        rec = {"kind": Q.KIND[name], "sink": sink, "cfg": {}}
        try:
            for c in cfgs:
                for w in range(a.warmup): run(c, name, f"warm{w}", sink)
                df = sp.sql(Q.ALL[name]); set_mode(sp, c); fl = plan_flags(df)
                rec["cfg"][c] = {"times": [], "plan_flags": fl}
            for r in range(a.reps):
                for c in cfgs:
                    t, _ = run(c, name, f"rep{r}", sink); rec["cfg"][c]["times"].append(t)
            for c in cfgs:
                ts = rec["cfg"][c]["times"]; rec["cfg"][c].update(median=statistics.median(ts), min=min(ts), max=max(ts))
            if "gpu" in rec["cfg"]:
                g = rec["cfg"]["gpu"]["median"]
                for c in cfgs:
                    if c != "gpu": rec["cfg"][c]["speedup_vs_gpu_pct"] = round((g / rec["cfg"][c]["median"] - 1) * 100, 1)
        except Exception as e:  # noqa
            rec["error"] = str(e)[:500]
        res["queries"][key] = rec
        line = " ".join(f"{c}={rec['cfg'][c].get('median', float('nan')):.2f}s" for c in rec["cfg"])
        print(f"[{now()}] {key} {line} {rec.get('error','')}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    sp.stop()

if __name__ == "__main__":
    main()
