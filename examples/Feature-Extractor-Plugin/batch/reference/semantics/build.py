#!/usr/bin/env python3
"""Merge parts/*.json into contracts.json and EXPRESSIONS.md.  Run: python3 semantics/build.py"""
import json, os, re, datetime
D = os.path.dirname(os.path.abspath(__file__)); P = os.path.join(D, "parts")
entries = []
for f in ["core.json", "math_bitwise_datetime.json"]:
    entries += json.load(open(os.path.join(P, f)))
cast = json.load(open(os.path.join(P, "cast.json")))
ce = cast["expression"]; ce.setdefault("category", "cast"); ce.setdefault("sql", "CAST(a AS type)")
ce["spark_class"] = "Cast"; ce["gpu_class"] = "GpuCast"; ce.setdefault("device_function_hint", "see cast_matrix")
entries.append(ce)
extra = os.path.join(P, "extra.json")
if os.path.exists(extra): entries += json.load(open(extra))
for e in entries:
    w = str(e.get("whitelist", "no"))
    e["tier"] = w.split()[0]            # v1 | v2 | no
    e["tier_condition"] = w[len(e["tier"]):].strip(" ()")
GLOBAL = {
 "spark_version_reference": "3.5.x (spark-rapids 26.08.1 jar; source tree 26.10-dev read-only)",
 "assumptions_v1": [
  "ANSI mode off (spark.sql.ansi.enabled=false); expressions with failOnError=true or hasSideEffects=true are NOT fusible",
  "session time zone UTC for any timestamp<->date/field extraction (non-UTC needs tz DB: v2)",
  "types: boolean, byte, short, int, long, float, double, date, timestamp. No string/decimal/nested/interval.",
  "all expressions deterministic and row-wise; no input_file_name/rand/monotonically_increasing_id/partition id",
 ],
 "float_rules": [
  "Compare with Spark total order: NaN==NaN true, NaN greater than everything, -0.0 == 0.0 true.",
  "Arithmetic is IEEE; nvrtc must use --fmad=false, no fast-math, no -use_fast_math, ftz off (denormals preserved).",
  "Output NaN payload may differ; tests normalise any NaN to one value but compare -0.0/0.0 bitwise.",
  "x / 0 and x % 0 (int AND float) yield NULL in non-ANSI mode, not Inf/NaN.",
 ],
 "int_rules": [
  "+,-,* wrap (two's complement); byte/short results truncated. abs/neg of MIN_VALUE = MIN_VALUE.",
  "Integral div / % : b==0 -> NULL; b==-1 must be special-cased (MIN div -1 = MIN, MIN % -1 = 0) - C++ would be UB.",
 ],
 "null_rules": [
  "NullIntolerant expressions: valid_out = AND(valid_in). Exceptions listed per expression (And/Or/If/CaseWhen/Coalesce/IsNull/IsNotNull/EqualNullSafe/IsNaN/NaNvl/Greatest/Least).",
  "Some expressions create NULL from non-NULL inputs (division/mod by zero, log(x<=0), float->timestamp NaN): output validity must be computed, and output nullability must match the unfused GPU plan.",
 ],
 "region_rules": [
  "Side-effect-free (hasSideEffects=false) subtree required; ANSI-checked ops are barriers.",
  "GpuFloor/GpuCeil report hasSideEffects=true even for double/long input (spark-rapids quirk): fusion should decide purity from input type, not from the flag, but must keep result identical.",
  "Do not fuse across GpuNondeterministic / non-retryable expressions.",
 ],
}
wl = sorted({e["spark_class"] for e in entries if e["tier"] == "v1" and e["category"] not in ("leaf","leaf-wrapper")})
out = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "global_rules": GLOBAL,
       "v1_whitelist_spark_classes": wl, "expressions": entries, "cast_matrix": cast["matrix"],
       "cast_excluded_types_note": cast.get("excluded_types_note", "")}
json.dump(out, open(os.path.join(D, "contracts.json"), "w"), indent=1, ensure_ascii=False)

def short(s, n=400):
    s = str(s).replace("\n", " ").replace("|", "\\|")
    return s if len(s) <= n else s[:n] + "..."
L = ["# 表达式融合合同（语义研究）", "",
     "机器可读版：`contracts.json`（由 `build.py` 从 `parts/` 合并生成）。**来源**：spark-rapids 源码（只读）+ Spark 3.5 语义。",
     "带“待差分测试确认”字样的条目是读源码得出的推断，已列入 `tests/` 的定向用例；以 `KNOWN-DIFFS.md` 与测试结果为准。", "",
     "## 全局规则", ""]
for k, v in GLOBAL.items():
    L.append(f"### {k}"); L += ([f"- {x}" for x in v] if isinstance(v, list) else [f"- {v}"]); L.append("")
L += ["## 总览（tier: v1=第一版白名单，v2=后续/仅近似，no=不融合）", "",
      "| 表达式 | GPU 类 | 类别 | tier | 条件/理由 |", "|---|---|---|---|---|"]
for e in entries:
    L.append(f"| {short(e['spark_class'],60)} | {short(e.get('gpu_class',''),40)} | {e['category']} | {e['tier']} | {short(e.get('tier_condition') or e.get('whitelist_reason',''),160)} |")
L += ["", "## 明细", ""]
for e in entries:
    L.append(f"### {e['spark_class']}  (`{e.get('gpu_class','')}`)  tier={e['tier']}")
    for k, lab in [("sql","SQL"),("types","类型"),("null_semantics","NULL 语义"),("non_ansi_behavior","非 ANSI 行为"),("ansi_behavior","ANSI 行为"),
                   ("rapids_notes","spark-rapids 实现要点"),("whitelist_reason","白名单理由"),("device_function_hint","设备函数要点")]:
        v = e.get(k)
        if v not in (None, "", {}): L.append(f"- **{lab}**：{v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False)}")
    if e.get("source"): L.append(f"- 源码：{e['source']}")
    L.append("")
L += ["## Cast 矩阵", "", "| from | to | CPU | GPU | tier | 设备函数 |", "|---|---|---|---|---|---|"]
for m in cast["matrix"]:
    L.append(f"| {m['from']} | {m['to']} | {short(m['cpu'],120)} | {short(m['gpu'],120)} | {m['whitelist']} | {short(m.get('device_function_hint',''),120)} |")
L += ["", "Cast 其它要点见 `parts/cast_notes.md`；数学/位运算/日期要点见 `parts/math_bitwise_datetime_notes.md`。", ""]
open(os.path.join(D, "EXPRESSIONS.md"), "w").write("\n".join(L))
print("entries", len(entries), "v1 classes", len(wl))
