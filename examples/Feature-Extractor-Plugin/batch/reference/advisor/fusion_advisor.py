#!/usr/bin/env python3
"""Fusion advisor: read Spark event logs (JSON lines; spark-rapids GPU plans) and say which queries are worth turning the
expression-fusion plugin on for.  It changes nothing and needs no GPU.

  python3 advisor/fusion_advisor.py EVENTLOG_OR_DIR [more ...] [--threshold 10] [--bw-gbs 864] [--eff 0.6] [--json out.json] [--md out.md]
                                    [--filter-desc REGEX]

What it measures (all from the event log):
  * per SQL execution: wall time, every Project/Filter operator (GpuProject/GpuFilter/Fused*), expression count (parsed from the plan text),
    the operators' "op time" metric and its share of all GPU operator time and of the query wall time,
  * writing style of the computation (SQL/DataFrame/translated UDF, Scala/Java UDF through CPU bridge, RapidsUDF, Python UDF, typed Dataset),
  * blockers (opaque expressions, strings, unsupported functions) that would stop fusion,
  * an ESTIMATE of saved intermediate memory traffic: unfused ~ (2.5 col accesses per operator), fused ~ (distinct inputs + outputs),
    converted to time with the GPU memory bandwidth, and capped by the operator's measured op time.
The estimate is a model, not a measurement: validate with bench/run_bench.py on a few queries before rolling out.
"""
import sys, os, re, json, gzip, argparse, collections, glob

PROJ = re.compile(r"^(Gpu)?(Fused)?(Gpu)?Project(Exec)?$"); FILT = re.compile(r"^(Gpu)?(Fused)?(Gpu)?Filter(Exec)?$")
PY_NODES = ("BatchEvalPython", "ArrowEvalPython", "GpuArrowEvalPython", "FlatMapGroupsInPandas", "FlatMapCoGroupsInPandas", "AggregateInPandas", "MapInPandas", "GpuMapInPandas", "WindowInPandas")
TYPED_NODES = ("MapElements", "TypedFilter", "DeserializeToObject", "SerializeFromObject", "MapPartitions", "AppendColumns", "MapGroups", "FlatMapGroups", "MapInBatch")
RDD_NODES = ("Scan ExistingRDD", "RDDScan", "ExistingRDD")
BIN_OPS = [" + ", " - ", " * ", " / ", " % ", " div ", " > ", " < ", " >= ", " <= ", " = ", " <=> ", " AND ", " OR ", " & ", " | ", " ^ "]
# SQL names (as printed in GPU plans) that the plugin's v1 whitelist handles; anything else printed as name( is a potential blocker
V1_FUNCS = set("""abs gpuabs sqrt floor ceil ceiling signum rint degrees radians isnan gpuisnan nanvl coalesce gpucoalesce greatest least pmod cast isnull gpuisnull isnotnull gpuisnotnull
 shiftleft shiftright shiftrightunsigned bit_count year month dayofmonth dayofweek quarter weekday last_day date_add date_sub datediff hour minute second not if case when and or in""".split())
V2_FUNCS = set("sin cos tan asin acos atan sinh cosh tanh exp expm1 log log2 log10 log1p cbrt pow power atan2 hypot round bround".split())
STRING_FUNCS = set("length substring substr concat upper lower trim ltrim rtrim like rlike regexp_replace regexp_extract split lpad rpad replace instr locate startswith endswith contains".split())

def open_log(p):
    if p.endswith(".gz"): return gzip.open(p, "rt", errors="replace")
    return open(p, errors="replace")

def iter_logs(paths):
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(glob.glob(os.path.join(p, "**", "*"), recursive=True)):
                if os.path.isfile(f) and not f.endswith((".crc", ".inprogress.crc", ".zstd", ".lz4", ".snappy")): yield f
        else: yield p

def parse(paths):
    execs = {}; runtime = collections.defaultdict(float); stage_acc = collections.defaultdict(dict); stage_exec = {}; job_stage = {}; app = {}
    for f in iter_logs(paths):
        try: fh = open_log(f)
        except Exception: continue
        with fh:
            for line in fh:
                if not line.startswith("{"): continue
                try: e = json.loads(line)
                except Exception: continue
                ev = e.get("Event", "")
                if ev.endswith("SparkListenerSQLExecutionStart"):
                    execs[e["executionId"]] = {"id": e["executionId"], "desc": e.get("description", ""), "start": e["time"], "plan": e["sparkPlanInfo"], "end": None, "file": f}
                elif ev.endswith("SparkListenerSQLExecutionEnd"):
                    if e["executionId"] in execs: execs[e["executionId"]]["end"] = e["time"]
                elif ev == "SparkListenerJobStart":
                    eid = (e.get("Properties") or {}).get("spark.sql.execution.id")
                    for s in e.get("Stage Infos", []):
                        if eid is not None: stage_exec[s["Stage ID"]] = int(eid)
                elif ev == "SparkListenerStageCompleted":
                    for a in e["Stage Info"].get("Accumulables", []):
                        stage_acc[e["Stage Info"]["Stage ID"]][a["ID"]] = a.get("Value")
                        if a.get("Name") == "internal.metrics.executorRunTime" and num(a.get("Value")) is not None:
                            runtime[e["Stage Info"]["Stage ID"]] += num(a.get("Value")) / 1000.0   # ms -> s (summed over tasks)
                elif ev == "SparkListenerApplicationStart": app = {"name": e.get("App Name")}
    # accumulator id -> value (max over stages: values are cumulative)
    acc = {}
    for st, d in stage_acc.items():
        for i, v in d.items():
            n = num(v)
            if n is not None: acc[(stage_exec.get(st), i)] = max(acc.get((stage_exec.get(st), i), 0), n)
    exec_runtime = collections.defaultdict(float)
    for st, v in runtime.items(): exec_runtime[stage_exec.get(st)] += v
    acc[('runtime', None)] = dict(exec_runtime)
    return execs, acc

def num(v):
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        m = re.match(r"^\s*-?\d+(\.\d+)?\s*$", v)
        if m: return float(v)
        m = re.search(r"\((\d+) bytes\)", v)
        if m: return float(m.group(1))
    return None

def flatten(n, out=None, depth=0):
    out = [] if out is None else out
    out.append((n, depth))
    for c in n.get("children", []): flatten(c, out, depth + 1)
    return out

def bracket_list(s):
    """expressions printed as 'GpuProject [e1, e2, ...], true' -> list of top-level expressions"""
    i = s.find("[")
    if i < 0: return [s]
    depth, cur, outl, j = 0, "", [], i
    for ch in s[i:]:
        if ch in "([": depth += 1
        if ch in ")]": depth -= 1
        if depth == 0 and ch == "]": break
        if depth == 1 and ch == "," : outl.append(cur.strip()); cur = ""; continue
        if not (depth == 1 and ch == "[" and cur == ""): cur += ch
    outl.append(cur.strip().lstrip("["))
    return [x for x in outl if x]

def analyse_exprs(simple):
    exprs = bracket_list(simple)
    text = " ".join(exprs)
    attrs = set(re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*#\d+L?", text))
    ops = sum(text.count(o) for o in BIN_OPS) + len(re.findall(r"\bCASE WHEN\b", text)) + len(re.findall(r"\bif \(", text)) + len(re.findall(r"\bNOT\b", text))
    funcs = re.findall(r"\b([A-Za-z_][A-Za-z_0-9]*)\(", text)
    ops += len([f for f in funcs if f.lower() not in ("if",)])
    low = [f.lower() for f in funcs]
    blockers = collections.Counter()
    for f in low:
        base = f[3:] if f.startswith("gpu") and f[3:] in V1_FUNCS | V2_FUNCS | STRING_FUNCS else f
        if f in V1_FUNCS or base in V1_FUNCS: continue
        if base in V2_FUNCS: blockers["非精确/后续批次函数:" + base] += 1
        elif base in STRING_FUNCS: blockers["字符串函数:" + base] += 1
        elif f == "gpucpubridge": blockers["CPU bridge(UDF 被翻译失败，留在 CPU 算)"] += 1
        elif f.startswith("gpu_"): blockers["RapidsUDF(手写 GPU UDF，黑盒)"] += 1
        elif "udf" in f: blockers["UDF 黑盒:" + f] += 1
        else: blockers["未识别函数:" + f] += 1
    if "pythonUDF" in text: blockers["Python UDF 结果被引用"] += 1
    return {"n_exprs": len(exprs), "n_ops": ops, "n_inputs": len(attrs), "blockers": dict(blockers), "text": text[:300]}

def analyse_exec(ex, acc, threshold, bw, eff, avg_w):
    nodes = flatten(ex["plan"])
    eid = ex["id"]
    dur = ((ex["end"] or ex["start"]) - ex["start"]) / 1000.0
    run_s = acc.get(("runtime", None), {}).get(eid, 0.0)
    par = min(max(run_s / dur, 1.0), 64.0) if dur > 0 and run_s else 1.0     # average task parallelism: op times are summed over concurrent tasks
    scan_op = 0.0
    optime = collections.defaultdict(float); total_op = 0.0; rows = 0.0
    pf = []
    styles = set(); pyn = []; typed = []; scan_rows = 0.0
    for n, d in nodes:
        name = n["nodeName"]; ms = {m["name"]: m["accumulatorId"] for m in n.get("metrics", [])}
        ot = None
        for k in ("op time", "total time", "duration"):
            if k in ms: ot = acc.get((eid, ms[k])) ;
            if ot is not None: break
        ot = (ot or 0.0) / 1e9 if ("op time" in ms or "total time" in ms) else 0.0
        if name.startswith("Gpu") or name.startswith("Fused"): total_op += ot
        orows = acc.get((eid, ms["output rows"])) if "output rows" in ms else None
        if orows and ("Scan" in name): scan_rows += orows
        if "Scan" in name and (name.startswith("Gpu") or "Scan" in name): scan_op += ot
        if orows: rows = max(rows, orows)
        if any(name.startswith(p) or name == p for p in PY_NODES): styles.add("Python UDF"); pyn.append(name)
        if any(name.startswith(p) for p in TYPED_NODES): styles.add("Dataset 带类型写法(ds.map/filter)"); typed.append(name)
        if any(name.startswith(p) for p in RDD_NODES): styles.add("RDD")
        base = name.replace("Gpu", "").replace("Fused", "").replace("Exec", "")
        if base in ("Project", "Filter") and (name.startswith("Gpu") or name.startswith("Fused")):
            info = analyse_exprs(n.get("simpleString", ""))
            info.update(node=name, kind=base, fused=name.startswith("Fused"), op_time_s=ot)
            pf.append(info)
    # style detection from expression text
    for p in pf:
        for b in p["blockers"]:
            if "CPU bridge" in b: styles.add("Scala/Java UDF(未被翻译，走 CPU bridge)")
            if "RapidsUDF" in b: styles.add("手写 RapidsUDF")
    if not styles or pf: styles.add("SQL/DataFrame(含已被 udf-compiler 翻译的 Scala UDF)")
    n_pf_cpu = sum(1 for n, d in nodes if n["nodeName"] in ("Project", "Filter") or re.match(r"^\*?\(\d+\) (Project|Filter)", n["nodeName"]))
    r = rows or scan_rows
    pf_op = sum(p["op_time_s"] for p in pf)
    saved_s = 0.0; saved_bytes = 0.0
    for p in pf:
        if p["fused"] or p["n_ops"] < 2: continue
        unfused = 2.5 * p["n_ops"] * r * avg_w
        fused = (p["n_inputs"] + max(p["n_exprs"], 1)) * r * avg_w
        sv = max(unfused - fused, 0.0)
        t = sv / (bw * 1e9 * eff)
        p["est_saved_bytes"] = sv; p["est_saved_s"] = min(t, 0.9 * p["op_time_s"] / par) if p["op_time_s"] else t
        saved_bytes += sv; saved_s += p["est_saved_s"]
    scan_wall = scan_op / par
    headroom = max(dur - scan_wall, 0.0)          # time that is not (parallel-overlapped) scan: an upper bound on what any operator-level change can win
    saved_s = min(saved_s, headroom)
    gain = 100.0 * saved_s / dur if dur > 0 else 0.0
    share_total = 100.0 * pf_op / total_op if total_op else 0.0
    share_wall = 100.0 * (pf_op / par) / dur if dur > 0 else 0.0
    n_ops = sum(p["n_ops"] for p in pf)
    bl = collections.Counter()
    for p in pf:
        if not p["fused"]:
            for b, c in p["blockers"].items(): bl[b] += c
    already = bool(pf) and all(p["fused"] for p in pf)
    # verdict
    if not pf: verdict, why = "无关", "计划里没有 GPU Project/Filter（全 CPU 或纯聚合/扫描），融合无从下手。"
    elif already: verdict, why = "已融合", "该查询已经跑在融合算子上。"
    else:
        reasons = []
        if bl: reasons.append("可能被挡住：" + "；".join(list(bl)[:3]))
        if n_ops < 3: reasons.append("表达式很少(%d 个运算)，没有可省的中间结果" % n_ops)
        if share_total < 15: reasons.append("Project/Filter 只占 GPU 算子时间的 %.0f%%，其余时间在扫描/聚合/join/shuffle，融合碰不到" % share_total)
        if gain >= threshold and not bl: verdict = "值得融合"
        elif gain >= threshold and bl: verdict = "部分值得"; reasons.append("先清理上面的阻塞项，否则区域会被切碎")
        elif gain >= threshold / 3: verdict = "边际"
        else: verdict = "不值得"
        if verdict in ("值得融合", "部分值得"): reasons.append("估算可省 %.1f%% 端到端(模型值)" % gain)
        why = "；".join(reasons) or "无明显阻塞。"
    return {"execution_id": eid, "description": ex["desc"][:100], "duration_s": round(dur, 2), "styles": sorted(styles), "project_filter_nodes": len(pf),
            "n_expr_ops": n_ops, "pf_op_time_s": round(pf_op, 3), "pf_share_of_gpu_op_time_pct": round(share_total, 1), "pf_share_of_wall_pct": round(share_wall, 1), "parallelism": round(par, 2), "scan_wall_s": round(scan_wall, 2),
            "rows_est": r, "est_saved_traffic_gb": round(saved_bytes / 1e9, 2), "est_saved_s": round(saved_s, 3), "est_gain_pct": round(gain, 1),
            "blockers": dict(bl), "cpu_project_filter_nodes": n_pf_cpu, "verdict": verdict, "why": why, "nodes": pf}

def render_md(rows, a):
    L = ["# 融合评估报告（Fusion advisor）", "", f"输入：{', '.join(a.paths)}；判定门槛：估算端到端 ≥ {a.threshold}%；显存带宽按 {a.bw_gbs} GB/s x {a.eff} 效率；平均列宽 {a.avg_width} 字节。",
         "估算是**模型值**（中间结果读写量 / 带宽，且不超过该算子实测 op time 的 90%），用来排序和筛选，上线前用 bench 实测确认。", ""]
    cnt = collections.Counter(r["verdict"] for r in rows)
    L.append("## 总览：" + "，".join(f"{k} {v} 个" for k, v in cnt.items()) + f"（共 {len(rows)} 个 SQL 执行）")
    L += ["", "| # | 查询 | 时长(s) | P/F 个数 | 运算数 | P/F 占 GPU 算子时间 | P/F 占总时长 | 估算端到端 | 结论 | 原因 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -r["est_gain_pct"]):
        L.append(f"| {r['execution_id']} | {r['description'][:40].replace('|', '/')} | {r['duration_s']} | {r['project_filter_nodes']} | {r['n_expr_ops']} | {r['pf_share_of_gpu_op_time_pct']}% | {r['pf_share_of_wall_pct']}% | {r['est_gain_pct']}% | **{r['verdict']}** | {r['why'][:160]} |")
    st = collections.defaultdict(lambda: [0, 0.0, 0.0])
    for r in rows:
        for s in r["styles"]: st[s][0] += 1; st[s][1] += r["duration_s"]; st[s][2] += r["est_saved_s"]
    L += ["", "## 按写法统计", "", "| 写法 | 查询数 | 总时长(s) | 估算可省(s) | 说明 |", "|---|---|---|---|---|"]
    notes = {"SQL/DataFrame(含已被 udf-compiler 翻译的 Scala UDF)": "可融合（SQL 与 DataFrame 同一种计划；翻译成功的 Scala UDF 在计划里与普通表达式无异，事件日志无法区分）",
             "Scala/Java UDF(未被翻译，走 CPU bridge)": "翻译失败，整个 Project 留在原样，不能融合；改写成 DataFrame 表达式最划算",
             "手写 RapidsUDF": "黑盒，只能融合它前后的部分", "Python UDF": "黑盒且要出 GPU，融合管不到；改写成表达式或 pandas UDF",
             "Dataset 带类型写法(ds.map/filter)": "lambda 在 CPU 上逐行执行，无法融合；改写成 DataFrame/SQL 表达式", "RDD": "没有计划可改"}
    for s, (n, d, sv) in sorted(st.items(), key=lambda kv: -kv[1][1]): L.append(f"| {s} | {n} | {d:.1f} | {sv:.2f} | {notes.get(s, '')} |")
    top = [r for r in rows if r["verdict"] in ("值得融合", "部分值得")]
    L += ["", "## 建议", ""]
    if top: L += [f"- 值得开融合的查询：" + "、".join(f"#{r['execution_id']}" for r in top[:10]) + f"（共 {len(top)} 个）；开启方式见 README，先 `spark.rapids.sql.fusion.shadow=true` 观察一轮。"]
    else: L += ["- 这批日志里没有估算 ≥ 门槛的查询：Project/Filter 占比低或表达式很少，融合预期收益小于门槛。"]
    blk = collections.Counter()
    for r in rows:
        for b, c in r["blockers"].items(): blk[b] += c
    if blk: L += ["- 最常见的阻塞项：" + "；".join(f"{b} x{c}" for b, c in blk.most_common(5))]
    return "\n".join(L) + "\n"

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+"); ap.add_argument("--threshold", type=float, default=10.0)
    ap.add_argument("--bw-gbs", type=float, default=864.0); ap.add_argument("--eff", type=float, default=0.6); ap.add_argument("--avg-width", type=float, default=8.0)
    ap.add_argument("--json", default=""); ap.add_argument("--md", default=""); ap.add_argument("--filter-desc", default="")
    a = ap.parse_args()
    execs, acc = parse(a.paths)
    rows = []
    for ex in execs.values():
        if a.filter_desc and not re.search(a.filter_desc, ex["desc"]): continue
        if ex["end"] is None: continue
        rows.append(analyse_exec(ex, acc, a.threshold, a.bw_gbs, a.eff, a.avg_width))
    md = render_md(rows, a)
    if a.md: open(a.md, "w").write(md)
    if a.json: json.dump(rows, open(a.json, "w"), indent=1, ensure_ascii=False, default=str)
    print(md)

if __name__ == "__main__":
    main()
