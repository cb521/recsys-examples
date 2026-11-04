# tests/ 差分与模糊测试

一键：`tests/run-all.sh [-c 随机用例数] [-s 种子] [-p 分区] [-o 输出目录] [-n 不测融合]`（1 张卡，约 20～30 分钟；设 `-p` 时用 srun 申请）。

| 文件 | 作用 |
|---|---|
| `fz/gen_data.py` | 生成 100 万行确定性测试表：NULL 约 12%，约 45% 为极值（MIN/MAX/0/-1/NaN/±Inf/-0.0/非规格数…），含 bool/byte/short/int/long/float/double/date/timestamp |
| `fz/exprgen.py` | 带类型的随机表达式树生成器（算术/比较/逻辑/CASE/COALESCE/greatest/least/Cast/位运算/数学/日期时间），可渲染成 SQL 与 DataFrame API |
| `fz/directed.py` | 定向网格：每个算子 x 全部特殊值笛卡尔积，三方（cpu / gpu 不融合 / fused）按位比较 |
| `fz/fuzz.py` `fz/fuzz_loop.sh` | 随机差分；失败自动最小化并存 `OUT/fuzz/failures/*.json`；卡死自动检测并续跑 |
| `fz/diff.py` | 按位比较（整数/布尔/日期完全一致，浮点按位且 NaN 归一，-0.0 与 0.0 区分） |
| `fz/report.py` | 汇总表；融合 != 不融合 GPU 时退出码 1 |
| `udf/` | Scala/Java/RapidsUDF/Python/typed Dataset 写法覆盖实验（`udf/build.sh` 用 Spark 自带的 scalac 编译，无需联网） |
| `gpu-run.sh` | GPU 运行封装：设了 `SLURM_PARTITION`（或 `-p`）就走 srun，否则本机直接跑（集群上脚本与数据要放共享文件系统，不能放 /tmp） |

比较对：`gpu-vs-cpu`（已知差异候选，见 `semantics/KNOWN-DIFFS.md`）、`FUSION-vs-unfused`（**必须为 0**）、`fused-vs-cpu`（参考）。
几个坑（已处理）：Parquet 字典编码会合并 0.0/-0.0（已关）；plugin jar 只能放 `--jars`/`spark.jars`，不能同时放 `--driver-class-path`。
