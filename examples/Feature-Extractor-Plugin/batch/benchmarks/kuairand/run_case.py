# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one KuaiRand feature case three ways and save every intermediate artifact.

    cpu   : plain Spark (spark.rapids.sql.enabled=false)
    gpu   : spark-rapids only (the baseline)
    fused : spark-rapids + expression-fusion plugin (spark.rapids.sql.fusion.mode=SAFE)

Writes to <out>/<case>/:
    plan_cpu.txt, plan_gpu.txt, plan_fused.txt   executed physical plans
    fusion_decisions.txt                         what the plugin fused / did not fuse and why
    kernels/*.cu                                 generated CUDA source (when FUSION_DUMP_DIR points there)
    timings.json                                 per-round wall times and medians (modes interleaved per round)

Sinks (what happens to the feature rows, i.e. what is timed):
    parquet  : write the features as Parquet (offline sample generation), the default
    gpu_agg  : reduce every feature column to a sum on the device and collect one row, so the rows never leave the
               GPU (stands in for "features go straight into GPU training")
    noop     : Spark's noop sink (forces a columnar-to-row conversion of every row on the host)
    check.txt                                    row-level equality of the three outputs

Must be started with FUSION_DUMP_DIR=<out>/<case>/kernels so the plugin dumps the kernels it compiles.
"""
import argparse
import json
import os
import statistics
import time

from pyspark.sql import SparkSession

MODES = ["cpu", "gpu", "fused"]


def configure(spark, mode):
    spark.conf.set("spark.rapids.sql.enabled", "false" if mode == "cpu" else "true")
    spark.conf.set("spark.rapids.sql.fusion.mode", "SAFE" if mode.startswith("fused") else "OFF")
    # fused_nosync: do not synchronize the stream after every fused kernel (default is to synchronize)
    spark.conf.set("spark.rapids.sql.fusion.syncAfterLaunch", "false" if mode == "fused_nosync" else "true")


def register_tables(spark, data):
    for t in ["log", "user", "video_basic"]:
        p = os.path.join(data, t)
        if os.path.exists(p):
            spark.read.parquet(p).createOrReplaceTempView(t)


def gpu_agg(df):
    cols = [c for c, t in df.dtypes if t != "string"]
    return df.selectExpr("count(*)", *[f"sum(CAST({c} AS DOUBLE))" for c in cols])


def run(df, sink, path):
    t0 = time.perf_counter()
    if sink == "parquet":
        df.write.mode("overwrite").parquet(path)
    elif sink == "gpu_agg":
        gpu_agg(df).collect()
    else:
        df.write.format("noop").mode("overwrite").save()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, help="path to a .sql file holding one SELECT")
    ap.add_argument("--data", required=True, help="directory with the Parquet tables made by prepare_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--sink", default="parquet", choices=["parquet", "gpu_agg", "noop"])
    ap.add_argument("--nosync", action="store_true", help="also time fused without the per-kernel stream synchronize")
    a = ap.parse_args()
    modes = MODES + (["fused_nosync"] if a.nosync else [])

    name = os.path.splitext(os.path.basename(a.case))[0] + ("" if a.sink == "parquet" else "_" + a.sink.replace("_", ""))
    out = os.path.join(a.out, name)
    os.makedirs(out, exist_ok=True)
    sql = open(a.case).read()

    spark = SparkSession.builder.appName(f"kuairand-{name}").getOrCreate()
    spark.conf.set("spark.session.timeZone", "UTC")
    jvm = spark.sparkContext._jvm
    register_tables(spark, a.data)

    # 1. plans and fusion decisions
    for mode in MODES:
        configure(spark, mode)
        df = spark.sql(sql)
        if a.sink == "gpu_agg":
            df = gpu_agg(df)
        with open(os.path.join(out, f"plan_{mode}.txt"), "w") as f:
            f.write(df._jdf.queryExecution().executedPlan().toString())
        if mode == "fused":
            with open(os.path.join(out, "fusion_decisions.txt"), "w") as f:
                f.write(jvm.com.nvidia.spark.rapids.fusion.Fusion.explain(df._jdf))

    # 2. timings: warm-up, then modes interleaved inside every round
    times = {m: [] for m in modes}
    for r in range(a.warmup + a.reps):
        for mode in modes:
            configure(spark, mode)
            t = run(spark.sql(sql), a.sink, os.path.join(out, f"sink_{mode}"))
            if r >= a.warmup:
                times[mode].append(t)
    med = {m: statistics.median(v) for m, v in times.items()}
    with open(os.path.join(out, "timings.json"), "w") as f:
        json.dump({"reps": a.reps, "warmup": a.warmup, "sink": a.sink,
                   "seconds": times, "median": med,
                   "fused_vs_gpu_speedup": med["gpu"] / med["fused"],
                   "gpu_vs_cpu_speedup": med["cpu"] / med["gpu"]}, f, indent=1)

    # 3. correctness: write each result (dictionary encoding off: it merges 0.0 and -0.0), compare on the CPU
    for mode in MODES:
        configure(spark, mode)
        spark.sql(sql).write.mode("overwrite").option("parquet.enable.dictionary", "false") \
            .parquet(os.path.join(out, f"res_{mode}"))
    configure(spark, "cpu")
    res = {m: spark.read.parquet(os.path.join(out, f"res_{m}")) for m in MODES}
    lines = [f"rows: {res['cpu'].count()}"]
    for x, y in [("gpu", "fused"), ("cpu", "gpu"), ("cpu", "fused")]:
        d1 = res[x].exceptAll(res[y]).count()
        d2 = res[y].exceptAll(res[x]).count()
        lines.append(f"rows only in {x}: {d1}   rows only in {y}: {d2}")
    with open(os.path.join(out, "check.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps(med))
    spark.stop()


if __name__ == "__main__":
    main()
