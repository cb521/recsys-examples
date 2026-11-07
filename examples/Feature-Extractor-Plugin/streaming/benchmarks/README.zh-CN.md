# 流式特征计算基准：Flink CPU 与 GPU（Mega-Flink）

本文档说明 `streaming` 部分的六组性能用例（c1–c6）如何定义、如何测量、结果是多少、有哪些局限，以及复现需要什么。读者入口见上层 README 的"第五部分"。

> 状态说明：本文数字来自内部实测报告的最新改写版。该版本在补测和改写之后**尚未经过独立复核**；改写前的版本曾经独立复核，复核意见已逐条处理。标"未核实"的内容表示没有找到直接证据。

## 目录

- [1. 总览](#overview)
- [2. 测试环境与数据集](#env)
- [3. 测量方法](#method)
- [4. 用例与结果](#cases)：[c1](#c1-filter)、[c2](#c2-window)、[c3](#c3-topn)、[c4](#c4-dedup)、[c5](#c5-join)、[c6](#c6-lookup)
- [5. 多并行度与大维表](#parallelism)
- [6. 时间花在哪里（breakdown 与 nsys）](#breakdown)
- [7. 已知局限](#limits)
- [8. 复现](#repro)

<a id="overview"></a>
## 1. 总览

打满档吞吐（条/秒），单卡、并行度 1、独占运行，两边都开 `pipeline.object-reuse`（"调优后"）：

| 用例 | CPU | GPU | 谁更快 |
| --- | --- | --- | --- |
| c1 过滤清洗 | 2,472,421 | 2,059,561 | CPU +20% |
| c2 HOP 两阶段 | 1,475,619 | 1,861,833 | GPU +26% |
| c2 TUMBLE | 1,613,516 | 2,020,259 | GPU +25% |
| c3 TopN | 1,363,629 | 1,231,075（batch） | CPU +11% |
| c4 去重 | 1,033,538 | 850,292（batch） | CPU +22% |
| c5 join | 291,797 | 253,233（融合） | CPU +15% |
| c5 join-filter | 307,921 | 262,297（融合） | CPU +17% |
| c5 join-agg | 293,385 | 236,056（融合） | CPU +24%（GPU p50 延迟低约 23%） |
| c6 Lookup（INNER） | 1,299,187 | 2,284,322 | GPU +76% |

百分比 = 较快一方吞吐 / 较慢一方吞吐 − 1。每个数字都是 3 轮中位数。

**配置口径很重要。** 默认配置（`pipeline.object-reuse=false`）下，c1 是 GPU 更快（1.03M 对 0.83M，GPU +23%）；两边同开 object-reuse 后翻转为 CPU 更快。object-reuse 对 CPU 的提升大于对 GPU（c1：CPU 约 ×2.95，GPU 约 ×2.0）。因此选型前应先把 CPU 侧调到位。下文每个用例同时给出默认配置与调优后的结果。

<a id="env"></a>
## 2. 测试环境与数据集

**硬件与软件**

- GPU：NVIDIA L20 单卡（驱动对应 CUDA 12.4，原生代码按 sm_89 编译）。
- Flink 2.2.1；Mega-Flink 为与之配套的同一次构建。JVM 堆 16 GB、G1，默认 heap 状态后端，watermark 延迟 20 ms，其余为默认（含默认 buffer-timeout）。
- 单并行度为主，本地盘读数据。宿主机 CPU 型号与核数：未核实，发布前补充。
- GPU 算子参数：攒批 4,096 行、最长等待 5 ms；c6 维表显存哈希表上限 4,194,304 行；c6 每批少于 1,024 行时回落到 JVM 查表。

**数据集**

- [KuaiRand](https://kuairand.com/)（快手公开的推荐数据集，含随机曝光视频），使用 KuaiRand-27K 的用户行为日志，共 92,919,789 行，按 `time_ms` 排序，转成 int64 列文件，进程内按时间顺序回放。引用格式与许可请以其公开页面为准（页面链接发布前请再核对）。
- 视频表共 32,038,725 行。c6 的维表只装入回放切片里出现过的视频（中/高档 1,039,412 行，低档 413,436 行；只取 video_id 与 author_id 两个 BIGINT 列），CPU 的 HashMap 与 GPU 显存哈希表使用同一份数据。

<a id="method"></a>
## 3. 测量方法

**三档负载**

| 档 | 输入速率 | 记录数 |
| --- | --- | --- |
| 低 | 20,000 条/秒 | 60 万 |
| 中 | 100,000 条/秒 | 300 万 |
| 打满 | 不限速 | 300 万 |

每档先以同样速率预热 30 万条再计时。低、中档受限速，两边吞吐相同（都跑满），用来比延迟；打满档用来比吞吐。

**轮数与统计**：每个"用例 × 档 × 模式"跑 3 轮，各模式轮换顺序；吞吐报 3 轮中位数；p50 / p95 / p99 是各轮各自的分位数再取中位数（不是吞吐中位数那一轮的值）。

**延迟定义**：从该条记录的计划发送时刻到 sink 收到的时间。定速档按计划时刻算，避免协调遗漏。打满档源头被反压，延迟是积压下的值，只能做相对比较。表中 "<0.01" 或 "0.00" 表示小于 0.01 ms（c1、c6 的 source 到 sink 在同一个算子链内）。c2 有少量窗口输出没有触发引用（打满/中档 24,354 行，低档 19,205 行），不计入延迟。

**正确性对拍**：每个用例的 CPU 与各 GPU 模式的最终表计算 SHA-256，逐"用例 × 档"比较，所有已发布结果均为 MATCH（0 次不一致）。c5 的 join 对数另用独立 Java 程序核对（正式轮 1,284,869 对）。注意：对拍对象是**最终表**，CPU 与 GPU 的中间 changelog 行数并不相同（GPU 批内合并会少发或多发中间更新），例如 c3 打满 CPU 965,196 行对 GPU batch 876,508 行，c4 为 3,361,901 对 3,302,645，c5-join-agg（默认配置）CPU 765,548 对融合 825,672。所以延迟样本数也略有不同。

**独占运行**：每个作业独占整个节点（一次只跑一个作业），1 张 GPU。另有一轮"并发初筛"（4 个作业同节点同时跑），仅用于筛选，不用于下表；同型号节点间差异小（多数 ±5% 内），并发初筛的吞吐多数比独占偏高 3–6%，个别用例（c3 GPU batch）有离群值，因此下表只用独占结果。

**CPU 基线**：Flink 自带的最优配置，两边相同的并行度与 mini-batch 设置；c2 同时测了默认两阶段（Local+Global）与一阶段（`1p`），c4 测了开 mini-batch 的 CPU（`cpu-mb`），取各自最优对照。"调优后"指两边都加 `pipeline.object-reuse=true`（c3 另加 `table.exec.rank.topn-cache-size=100000`，因为打满档约有 20 万个不同用户）。除此之外未对 CPU 或 GPU 做逐用例的更多调参。

**耗时分解（breakdown）的两类数据，口径不同**：

- JFR 采样（CPU 侧各段占比）是**采样估算**：按线程栈归类，带采样开销，只用于看大致分布。
- Mega-Flink 算子自带的 CUDA event 计时（行→列、H2D、kernel、D2H、列→行）是**实测**。其中 H2D/D2H 含排队与同步，比 nsys 看到的拷贝时间高 1.5–2.5 倍。
- 另用 nsys 对少数用例做了独立核对：kernel 总时间与 Mega-Flink 计时的 kernel 段一致（相差约 10%）。
- breakdown 作业每组只跑 1 轮且开了 profiling，吞吐略低，不用于比速度。

<a id="cases"></a>
## 4. 用例与结果

下列表格中，"默认"指 Flink 默认配置，"调优"指两边同开 object-reuse（c3 加 topn-cache）。吞吐单位为条/秒，延迟单位为 ms。"对拍"一列对该行所在的"用例 × 档"整体而言。

<a id="c1-filter"></a>
### c1 过滤清洗

**做什么 / 场景**：推荐样本清洗与基础特征：过滤无效或随机曝光记录，保留播放超过 3 秒或有点击的行，并计算观看比例、时长分桶、互动次数。执行计划为 `Source → Calc → Sink`，CPU 与 GPU 的逻辑与物理计划相同，GPU 模式只是把 Calc 换成 GPU Calc。

```sql
SELECT seq, user_id, video_id, time_ms,
       CAST(play_time_ms AS DOUBLE) / CAST(duration_ms + 1 AS DOUBLE) AS watch_ratio,
       CASE WHEN duration_ms < 15000 THEN 0 WHEN duration_ms < 60000 THEN 1 ELSE 2 END AS duration_bucket,
       is_like + is_follow + is_comment + is_forward AS interact_cnt,
       is_click
FROM events
WHERE duration_ms > 0 AND play_time_ms >= 0 AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
```

**默认配置**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 低 | 20,000 | <0.01 / <0.01 | 19,999 | 2.74 / 6.16 | MATCH |
| 中 | 100,000 | <0.01 / 0.21 | 99,989 | 2.96 / 7.60 | MATCH |
| 打满 | 831,997 | <0.01 / <0.01 | 1,025,360 | 1.86 / 3.38 | MATCH |

**调优（两边同开 object-reuse）**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 低 | 未测 | 未测 | 未测 | 未测 | 未测 |
| 中 | 100,000 | <0.01 / 0.01 | 99,978 | 2.80 / 6.23 | MATCH |
| 打满 | 2,472,421 | <0.01 / <0.01 | 2,059,561 | 0.92 / 1.77 | MATCH |

**结论**：默认配置下 GPU +23%，调优后 CPU +20%。GPU 一侧的 Mega-Flink 实测计时（默认配置，打满，927 批）：行→列 254 ms、H2D 20 ms、kernel 25 ms、D2H 14 ms、列→行 186 ms；这五段合计 499 ms，kernel 约占 5%，行列转换约占 88%。nsys 核对：kernel 13.2 ms、H2D 15.9 ms、D2H 8.1 ms。

<a id="c2-window"></a>
### c2 每用户最近 30 分钟点击数的窗口聚合（HOP 与 TUMBLE）

**做什么 / 场景**：实时统计特征，如"用户最近 N 分钟的曝光 / 点击 / 播放时长"。事件时间窗口，按 user_id 聚合。HOP 窗口长 30 分钟、步长 5 分钟（每条记录落入多个窗口）；TUMBLE 为 5 分钟滚动窗口。CPU 默认把窗口聚合拆为 Local + Global 两阶段；另测一阶段（`1p`，直接 WindowAggregate）。GPU 两阶段对应实验性的 `LocalWindowAggregate` 路径，一阶段对应 Mega-Flink 较成熟的路径。

```sql
-- HOP
SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt,
       SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms
FROM TABLE(HOP(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE, INTERVAL '30' MINUTE))
GROUP BY user_id, window_start, window_end;

-- TUMBLE：把 HOP(...) 换成
TABLE(TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE))
```

**HOP，默认配置**（"1p" = 一阶段）

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU 两阶段 | 19,951 | 64.08 / 123.64 | MATCH |
| 低 | CPU 1p | 19,951 | 30.69 / 67.84 | MATCH |
| 低 | GPU 两阶段 | 19,971 | 54.76 / 105.96 | MATCH |
| 低 | GPU 1p | 19,973 | 26.36 / 52.98 | MATCH |
| 中 | CPU 两阶段 | 99,752 | 68.30 / 132.81 | MATCH |
| 中 | CPU 1p | 99,747 | 25.19 / 64.30 | MATCH |
| 中 | GPU 两阶段 | 99,821 | 38.24 / 69.66 | MATCH |
| 中 | GPU 1p | 99,870 | 17.28 / 40.35 | MATCH |
| 打满 | CPU 两阶段 | 744,721 | 53.80 / 117.34 | MATCH |
| 打满 | CPU 1p | 628,892 | 26.11 / 78.29 | MATCH |
| 打满 | GPU 两阶段 | 898,493 | 37.33 / 83.32 | MATCH |
| 打满 | GPU 1p | 892,390 | 16.62 / 57.67 | MATCH |

**HOP，调优**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU 两阶段 | 99,824 | 62.89 / 124.43 | MATCH |
| 中 | CPU 1p | 99,828 | 22.46 / 65.07 | MATCH |
| 中 | GPU 两阶段 | 99,924 | 37.16 / 64.68 | MATCH |
| 中 | GPU 1p | 99,920 | 16.06 / 34.12 | MATCH |
| 打满 | CPU 两阶段 | 1,475,619 | 48.34 / 97.40 | MATCH |
| 打满 | CPU 1p | 1,079,916 | 25.01 / 86.26 | MATCH |
| 打满 | GPU 两阶段 | 1,861,833 | 36.44 / 99.76 | MATCH |
| 打满 | GPU 1p | 1,184,784 | 16.02 / 51.10 | MATCH |

低档调优：未测。

**TUMBLE，默认配置**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 低 | 19,992 | 73.37 / 119.64 | 19,993 | 52.67 / 105.76 | MATCH |
| 中 | 99,961 | 59.86 / 122.12 | 99,967 | 34.96 / 77.36 | MATCH |
| 打满 | 724,616 | 43.42 / 105.17 | 965,671 | 32.37 / 94.38 | MATCH |

**TUMBLE，调优**

| 档 | CPU 吞吐 | CPU p50 / p99 | GPU 吞吐 | GPU p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| 中 | 99,972 | 61.94 / 121.82 | 99,976 | 34.66 / 63.30 | MATCH |
| 打满 | 1,613,516 | 37.31 / 107.71 | 2,020,259 | 33.91 / 86.85 | MATCH |

**结论**：GPU 吞吐与延迟都更好。调优后 HOP 两阶段 GPU +26%，TUMBLE +25%；一阶段对一阶段 GPU +10%（CPU 一阶段本身比两阶段慢，所以"最优对最优"为 +26%）。默认配置下的"GPU +42%（对 CPU 一阶段）"是非最优对比，不建议引用。CPU 一阶段在默认配置下轮间波动较大（0.59–0.69M）。GPU 一侧的 breakdown 里状态读写（stateRead / merge）并未消失，"GPU 批量处理省掉状态读写"没有实测支持。

<a id="c3-topn"></a>
### c3 每用户观看最久的前 10 个视频（TopN）

**做什么 / 场景**：用户行为序列上的 Top-K 特征，如"最近观看最久的 10 个视频"。append-only 的 `ROW_NUMBER` TopN。计划为 `Calc → Exchange(hash user_id) → Rank(AppendFast)`，GPU 把 Rank 换成 GPU TopN，有三种形态：batch（攒批）、persistent（常驻）、adaptive（自适应）。

```sql
SELECT seq, user_id, video_id, play_time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn
  FROM events) WHERE rn <= 10
```

**默认配置**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU | 19,995 | 14.70 / 33.03 | MATCH |
| 低 | GPU batch | 19,995 | 16.81 / 33.23 | MATCH |
| 低 | GPU persistent | 19,995 | 14.73 / 33.14 | MATCH |
| 低 | GPU adaptive | 19,995 | 14.99 / 33.04 | MATCH |
| 中 | CPU | 99,977 | 3.58 / 6.69 | MATCH |
| 中 | GPU batch | 99,976 | 8.59 / 16.11 | MATCH |
| 中 | GPU persistent | 99,976 | 4.14 / 8.92 | MATCH |
| 中 | GPU adaptive | 99,979 | 4.15 / 7.40 | MATCH |
| 打满 | CPU | 1,024,400 | 4.69 / 20.25 | MATCH |
| 打满 | GPU batch | 930,760 | 5.61 / 25.01 | MATCH |
| 打满 | GPU persistent | 489,815 | 11.20 / 33.67 | MATCH |
| 打满 | GPU adaptive | 690,440 | 10.28 / 32.89 | MATCH |

GPU batch 打满档轮间波动大：8 轮复测 0.90–1.16M，呈双峰（5 轮约 0.93M，3 轮 1.13–1.16M）；CPU 8 轮 1.02–1.12M，中位数 1.06M。因此不能说 GPU 稳定胜出或稳定落后。

**调优（两边开 object-reuse，另加 topn-cache-size=100000）**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU | 99,978 | 3.53 / 6.70 | MATCH |
| 中 | GPU batch | 99,972 | 8.64 / 15.12 | MATCH |
| 中 | GPU persistent | 99,977 | 4.16 / 13.48 | MATCH |
| 中 | GPU adaptive | 99,978 | 4.05 / 10.77 | MATCH |
| 打满 | CPU | 1,363,629 | 3.59 / 22.79 | MATCH |
| 打满 | GPU batch | 1,231,075 | 5.22 / 29.97 | MATCH |
| 打满 | GPU persistent | 430,154 | 12.87 / 38.05 | MATCH |
| 打满 | GPU adaptive | 718,364 | 9.48 / 31.29 | MATCH |

低档调优：未测。

**结论**：CPU 不输：调优后 CPU +11%（对 GPU batch），并行度 2/4 下 CPU +38% / +21%（见第 5 节）。persistent 仅在低/中档延迟优于 batch，但不优于 CPU，且打满吞吐很低；adaptive 在低/中档 0 行走 GPU（延迟与 CPU 相同，不是 GPU 的结果），打满档约 36% 的行走 CPU 路径。

**为什么 GPU 没赢**（默认配置，独占，nsys 与参数扫描）：910 次 `select_top_n_by_group` kernel 合计 108 ms，加 H2D 5 ms、D2H 2 ms，约占约 3.2 s 作业时间的 3.6%，GPU 设备 95% 以上空闲；批大小（1,024 到 65,536）与最长等待（1/5/20 ms）扫描下，打满吞吐都在 0.87–1.11M，无稳定趋势；只有 source + 交换 + sink、不做任何计算的上限探针，CPU 0.856M、GPU 0.851M，说明瓶颈在 JVM 流水线。

<a id="c4-dedup"></a>
### c4 keep-last 去重

**做什么 / 场景**：样本去重、取最新状态。按 (user_id, video_id) 保留时间最新的一条。计划为 `Calc → Exchange → Rank(AppendFast, keep-last) → Calc`；GPU 把 Rank 换成 GPU Deduplicate（batch / adaptive）。另测开 mini-batch 的 CPU（`cpu-mb`）。

```sql
SELECT seq, user_id, video_id, play_time_ms, time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms, time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY ts DESC) AS rn
  FROM events) WHERE rn = 1
```

**默认配置**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU | 19,995 | 11.49 / 24.98 | MATCH |
| 低 | CPU mini-batch | 19,995 | 22.50 / 43.65 | MATCH |
| 低 | GPU batch | 19,995 | 14.54 / 25.23 | MATCH |
| 低 | GPU adaptive | 19,995 | 11.52 / 24.95 | MATCH |
| 中 | CPU | 99,962 | 2.88 / 5.06 | MATCH |
| 中 | CPU mini-batch | 99,950 | 14.93 / 30.94 | MATCH |
| 中 | GPU batch | 99,967 | 5.99 / 16.08 | MATCH |
| 中 | GPU adaptive | 99,967 | 2.86 / 5.06 | MATCH |
| 打满 | CPU | 685,882 | 5.23 / 19.63 | MATCH |
| 打满 | CPU mini-batch | 580,305 | 7.74 / 26.90 | MATCH |
| 打满 | GPU batch | 660,054 | 7.22 / 27.85 | MATCH |
| 打满 | GPU adaptive | 675,786 | 6.98 / 28.34 | MATCH |

**调优**

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU | 99,970 | 2.74 / 5.06 | MATCH |
| 中 | GPU batch | 99,964 | 5.93 / 13.95 | MATCH |
| 中 | GPU adaptive | 99,970 | 2.86 / 7.19 | MATCH |
| 打满 | CPU | 1,033,538 | 3.30 / 17.82 | MATCH |
| 打满 | GPU batch | 850,292 | 5.96 / 21.96 | MATCH |
| 打满 | GPU adaptive | 852,773 | 6.03 / 22.28 | MATCH |

低档调优：未测。

**结论**：默认配置下 CPU 略好（0.686M 对 0.660M），调优后 CPU +22%。去重的主要成本是序列化与 JVM 状态读写，不在 GPU 能替换的部分（JFR 估算：CPU 状态约 14%，GPU 模式状态约 17%；GPU 实测 kernel 13 ms、H2D 8 ms、D2H 8 ms，983 批）。adaptive 在低/中档 0 行走 GPU；打满档约 4.5% 的行走 CPU 路径。

<a id="c5-join"></a>
### c5 曝光 × 点击 Interval Join，及后接 Filter / 聚合的融合

**做什么 / 场景**：曝光-点击归因、训练样本拼接。曝光流 `imp` 与点击流 `clk`（`is_click=1`）取自同一份日志，join 键为 `user_id*10^8+video_id`（单 BIGINT），条件为点击时间落在曝光后 30 分钟内。三个变体：

- `c5-join`：只做 join。
- `c5-join-filter`：join 后加 `c.play_time_ms >= i.duration_ms`（Flink 将其下推为 join 的残差条件；GPU 融合模式会把残差条件放到设备端执行，因此"融合 vs 不融合"之差 = 融合 + 残差下放）。
- `c5-join-agg`：join 后按 user_id 聚合（开 mini-batch 5 ms / 4096，两边相同）。

GPU 融合模式下，join 后的 Calc / LocalAggregate 与 join 在同一个 GPU 阶段内完成，少一次列→行与行→列。`c5-join` 的 GPU batch 默认即为融合（Join→投影）。

```sql
CREATE TEMPORARY VIEW imp AS
  SELECT seq, user_id * 100000000 + video_id AS uv, user_id, duration_ms, ts FROM events;
CREATE TEMPORARY VIEW clk AS
  SELECT seq, user_id * 100000000 + video_id AS uv, play_time_ms, ts FROM events WHERE is_click = 1;

-- c5-join
SELECT i.seq, c.seq, i.uv, c.play_time_ms, i.duration_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE;

-- c5-join-filter：在上面基础上加
WHERE c.play_time_ms >= i.duration_ms      -- 此时只选 i.seq, c.seq, i.uv, c.play_time_ms

-- c5-join-agg：在 join 条件之后
SELECT i.user_id, COUNT(*) AS clicks, SUM(c.play_time_ms) AS play_ms
FROM imp i JOIN clk c ON ... GROUP BY i.user_id;
```

**默认配置**（同一作业内显式 fuse / nofuse，3 轮中位数；"nres-nofuse"= 不融合但设备端残差开）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join | 低 | 19,994 / 29.91 / 79.26 | 19,993 / 29.88 / 80.09 | 19,993 / 29.98 / 79.91 | MATCH |
| join | 中 | 99,928 / 10.57 / 24.84 | 99,921 / 14.08 / 27.56 | 99,933 / 14.23 / 26.58 | MATCH |
| join | 打满 | 259,430 / 20.62 / 55.11 | 232,091 / 24.97 / 56.24 | 221,821 / 26.10 / 56.35 | MATCH |
| join-filter | 低 | 19,993 / 29.64 / 79.08 | 19,990 / 30.00 / 80.25 | 19,989 / 29.97 / 80.03 | MATCH |
| join-filter | 中 | 99,928 / 10.36 / 26.64 | 99,953 / 13.95 / 26.62 | 99,904 / 13.95 / 25.87 | MATCH |
| join-filter | 打满 | 272,760 / 20.17 / 56.22 | 254,153 / 22.89 / 54.92 | 250,751 / 23.30 / 54.38 | MATCH |
| join-agg | 低 | 19,861 / 107.58 / 195.47 | 19,958 / 92.68 / 180.70 | 19,859 / 96.89 / 183.59 | MATCH |
| join-agg | 中 | 99,252 / 55.91 / 102.08 | 99,774 / 37.97 / 76.68 | 99,397 / 43.59 / 86.78 | MATCH |
| join-agg | 打满 | 268,556 / 60.56 / 121.23 | 233,707 / 44.49 / 83.84 | 232,152 / 50.97 / 98.09 | MATCH |

join-filter 的"不融合 + 设备端残差"（打满）为 251,131，与"不融合"251k 基本相同，说明残差下放单独几乎没有收益。GPU persistent 形态（默认融合）：join 打满 166,494（p99 199.94 ms）、join-filter 打满 171,651（p99 197.54 ms），各档尾延迟约 200 ms，均明显差于 CPU 与 batch。

**调优**（两边同开 object-reuse；低档：未测）

| 用例 | 档 | CPU 吞吐 / p50 / p99 | GPU 融合 | GPU 不融合 | 对拍 |
| --- | --- | --- | --- | --- | --- |
| join | 中 | 99,928 / 10.38 / 37.56 | 99,931 / 13.89 / 26.79 | 99,938 / 13.84 / 25.97 | MATCH |
| join | 打满 | 291,797 / 19.63 / 60.01 | 253,233 / 24.07 / 54.08 | 251,270 / 24.24 / 52.71 | MATCH |
| join-filter | 中 | 99,927 / 10.09 / 30.54 | 99,903 / 13.82 / 25.54 | 99,921 / 13.71 / 25.68 | MATCH |
| join-filter | 打满 | 307,921 / 18.80 / 63.22 | 262,297 / 23.49 / 51.55 | 259,502 / 23.67 / 52.45 | MATCH |
| join-agg | 中 | 99,141 / 55.20 / 102.33 | 99,798 / 37.57 / 76.84 | 99,377 / 42.76 / 86.52 | MATCH |
| join-agg | 打满 | 293,385 / 58.96 / 127.08 | 236,056 / 45.31 / 84.37 | 253,569 / 48.46 / 98.14 | MATCH |

join-agg 打满档 CPU 三轮为 293k、257k、300k，波动较大。

**结论**：吞吐上 CPU 更快：调优后 join +15%、join-filter +17%、join-agg +24%（对 GPU 融合）。接聚合时 GPU 延迟更低（p50 约低 23%）。融合：调优后吞吐无收益（join +0.8%、join-filter +1.1%，噪声内；join-agg 融合反而低约 7%），只有 join-agg 的 p50 延迟低约 6.5%（打满）/ 12%（中档）；join、join-filter 延迟无差别。两边最大的成本都是 JVM 状态读写（join 两侧的缓冲，JFR 估算：CPU join 约 52%、GPU join 约 43%），GPU 没有替换它。本数据上 join 放大倍数约为 1，融合能省的搬运量有限。

<a id="c6-lookup"></a>
### c6 查特征维表 Lookup Join

**做什么 / 场景**：实时拼接物品侧特征：事件流按 video_id 查维表取 author_id。仅测 INNER。CPU 用全量内存 HashMap（相当于 Flink 全缓存的较好情形），GPU 把维表装入显存哈希表。注意：每批少于 1,024 行时 GPU 路径回落到 JVM 查表，所以低/中档的 `gpu` 模式多半没有真正上 GPU；另测 `gpu-min1`（强制每批上 GPU）作对照。

```sql
SELECT e.seq, e.video_id, d.author_id
FROM events AS e
JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d
  ON e.video_id = d.video_id
```

维表为 1,039,412 行（低档 413,436 行），不是全表 3,204 万行。

**GPU 是否真正参与（nsys 统计 CUDA kernel 实例数，默认配置，各 1 次）**

| 档 | `gpu` kernel 实例 | `gpu-min1` kernel 实例 |
| --- | --- | --- |
| 低 | 3（几乎没上 GPU） | 8,484 |
| 中 | 22（几乎没上 GPU） | 6,151 |
| 打满 | 878 | 848 |

因此低/中档 `gpu` 的约 3 ms 延迟不是 GPU 带来的；只有打满档的 GPU 结果有运行时证据。c6 的正式计时轮次没有运行时计时指标（Lookup Join 无此类指标），证据来自规划期 `shim_status`（三档均为 gpu-replaced）和 nsys。

**默认配置**（同一作业内三种模式轮换）

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 低 | CPU | 20,000 | <0.01 / <0.01 | MATCH |
| 低 | GPU（实际走 JVM） | 19,998 | 2.70 / 6.10 | MATCH |
| 低 | GPU min1 | 19,998 | 2.71 / 6.14 | MATCH |
| 中 | CPU | 100,000 | <0.01 / 14.76 | MATCH |
| 中 | GPU（实际走 JVM） | 99,990 | 3.18 / 9.48 | MATCH |
| 中 | GPU min1 | 99,991 | 2.96 / 8.97 | MATCH |
| 打满 | CPU | 601,930 | <0.01 / <0.01 | MATCH |
| 打满 | GPU | 838,326 | 2.33 / 4.70 | MATCH |
| 打满 | GPU min1 | 834,406 | 2.33 / 4.76 | MATCH |

**调优**（低档：未测）

| 档 | 模式 | 吞吐 | p50 / p99 | 对拍 |
| --- | --- | --- | --- | --- |
| 中 | CPU | 100,000 | <0.01 / 19.20 | MATCH |
| 中 | GPU | 99,990 | 2.93 / 6.28 | MATCH |
| 中 | GPU min1 | 99,994 | 2.78 / 6.20 | MATCH |
| 打满 | CPU | 1,299,187 | <0.01 / <0.01 | MATCH |
| 打满 | GPU | 2,284,322 | 0.88 / 1.75 | MATCH |
| 打满 | GPU min1 | 2,237,176 | 0.89 / 1.76 | MATCH |

**结论**：打满档 GPU 更快：默认 +39%（838k 对 602k），调优后 +76%；低/中档受限速，吞吐相同，GPU 因攒批多约 3 ms 延迟。只验证了单 BIGINT 键加少量 BIGINT 列这一最简情形。

**LEFT lookup**：在本构建下不能上 GPU，会回落到 CPU，所以没有性能对比。规划期状态：维表有主键时报输出维表列与维表 schema 不匹配（LEFT 把维表列变为可空，而主键列非空，类型不相等）；去掉主键则报 lookup key 未证明包含维表主键。

**JFR 占比不采用**：c6 的 JFR 归类把 Mega-Flink 包名下的类一律计为 "GPU 算子 JVM 侧"，而 CPU 查表用的维表类恰好在同一个包下，导致 CPU 作业里也出现约 24% 的 "GPU 相关"，确认是误归类，因此 c6 的 JFR 占比不作结论，其他用到同一个包的用例（含驱动类本身）的 "GPU 相关" 占比也应复查。

<a id="parallelism"></a>
## 5. 多并行度与大维表

做法：同一份回放，单个轻量 Long 序列源经 `k mod p` 分发给 p 个回放 map（每个 subtask 按时间序回放），后续算子并行度 = p，CPU 与 GPU 并行度相同，p 个 subtask 共用 1 张 L20。仅测 c2、c3、c6。默认配置下的并行度 2/4 对拍 18 组（用例 × 档 × 并行度）全部 MATCH；调优后并行度 2/4 的对拍状态：未核实。

**调优后，打满档吞吐（条/秒）**（以下并行度 2/4 的数字取自内部报告，本草稿未逐项回源核对）

| 用例 | 并行度 2（CPU / GPU） | 并行度 4（CPU / GPU） | 谁更快 |
| --- | --- | --- | --- |
| c2 HOP 两阶段 | 2,015k / 2,532k | 2,907k / 3,365k | GPU +26% / +16% |
| c2 HOP 一阶段 | 1,846k / 2,017k | 3,028k / 3,340k | GPU +9% / +10% |
| c3 TopN（GPU batch） | 2,517k / 1,824k | 3,827k / 3,154k | CPU +38% / +21% |
| c6 Lookup | 1,592k / 2,991k | 3,168k / 3,809k | GPU +88% / +20% |

c3 GPU adaptive：并行度 2 为 980k，并行度 4 为 1,545k。

**c6 大维表**（调优，并行度 1，维表垫到 400 万行，GPU lookup 上限 4,194,304 行）：打满 CPU 1,292k，GPU 2,360k（+83%），gpu-min1 2,370k；中档两边跑满，GPU p50 3.1 ms，CPU <0.01 ms。

**延迟注意**：并行度大于 1 时，回放与算子之间多一次网络交换（Flink 默认 buffer-timeout 100 ms，低流量时缓冲块要等超时才发），所以低/中档 p50 比并行度 1 高很多（例如 c3 低档 83–126 ms 对并行度 1 的 14.7 ms）。这是测试驱动引入的、CPU 与 GPU 共有的加成，并行度大于 1 的低/中档延迟只用于两边相对比较。

<a id="breakdown"></a>
## 6. 时间花在哪里（breakdown 与 nsys）

- 单并行度下，Flink 自身的序列化 / 网络交换、source 回放、状态读写、DataStream 与 Table 的行转换占大头；GPU kernel 本身只占很小一块。例如 c3 GPU batch 打满：kernel 约 0.12 s，整个作业约 3.3M 行约 3.2 s（按正式中位数）。
- GPU 一侧新增的成本是行→列、H2D / D2H、列→行，以及 JVM 侧的批打包与解包；状态读写、timer、changelog 合并、Global 聚合仍在 JVM，GPU 没有替换。
- 采样估算与实测的口径区别见第 3 节。JFR 只采样线程名含 "(1/1)" 的任务线程，persistent 形态若有独立常驻线程则未被统计，其 GPU 占比仅供参考。
- CPU 与 GPU 的逻辑计划、物理计划（Calc / Rank / WindowAggregate / IntervalJoin / LookupJoin 的结构）完全相同，EXPLAIN 里看不出区别；替换发生在 ExecNode 翻译为 Transformation 的阶段。

<a id="limits"></a>
## 7. 已知局限

1. **单卡、并行度主要为 1。** 并行度 2、4 只测了 c2、c3、c6；c4、c5 在并行度 2/4 未测；多卡、更大并行度未测。
2. **c6 全量维表未测。** 全表 3,204 万行超过 GPU lookup 上限（4,194,304 行），只测了回放切片内的 1,039,412 行和垫到 400 万行的情形。
3. **报告改写后尚未独立复核。** 补测（object-reuse 调优、并行度、TUMBLE、nsys）之后的改写版没有经过独立复核。
4. **LEFT lookup 本构建不能上 GPU**，没有 LEFT 的性能对比。
5. **c6 的 JFR 占比因误归类不采用。** 凡用到同一包的用例，"GPU 相关"占比都需复查。
6. **JFR 是采样估算，**不是精确耗时；Mega-Flink 计时与 nsys 才是实测，其中 H2D / D2H 含排队。
7. **调优档位不全。** 调优后的低档（20k/s）未测；c4、c5、c6 的调优只测了中、打满两档。调优仅限 `pipeline.object-reuse`（及 c3 的 topn-cache），未做更多逐用例调参，也没有测试 mini-batch 等其他 CPU 侧选项（c4 的 `cpu-mb` 与 c5-join-agg 的 mini-batch 除外）。
8. **默认配置不是最优配置。** 默认配置下的结论（如 c1 GPU 赢）在调优后会变化，引用时请注明口径。
9. **对拍对象是最终表**，中间 changelog 在 CPU 与 GPU 之间不同（见第 3 节）。
10. **GPU persistent 与 adaptive：** 本次未得到"选 persistent"的支持；adaptive 在低/中档 0 行走 GPU。c5 persistent 只在 breakdown 轮记录了命令计数，正式轮为 0，原因未核实。
11. **未做：** 其他 GPU 型号（如 H20）、多卡、更大规模的维表、其他数据集；只验证了一种数据集与一种 GPU。
12. **c3 GPU batch 打满档有双峰波动**（见 c3 一节），机制没有更细的证据，不作断言。

<a id="repro"></a>
## 8. 复现

**目前脚本和驱动程序尚未放入本仓库，也还没有完成路径参数化**（现有脚本里的数据与输出路径是写死的本机路径，需要改成参数后才能发布）。因此下面只列出复现需要什么，还不能"一条命令复现"。

需要的东西：

- 1 张 NVIDIA L20（或同代 sm_89 的 GPU），驱动支持 CUDA 12.4；独占节点；宿主机内存足够 16 GB JVM 堆加数据文件。
- Flink 2.2.1 与与之配套构建的 Mega-Flink（GPU 算子与 shim）。获取方式与构建说明：未核实，待补。
- KuaiRand-27K 数据集（从其公开页面获取），以及把日志转成按 `time_ms` 排序的 int64 列文件的转换程序（内部已有，尚未放入仓库）。
- 测试驱动：负责进程内回放、SQL 定义、sink 收 changelog、延迟统计与 SHA-256 对拍（内部已有，尚未放入仓库）。
- 运行脚本：按"用例 × 模式 × 档位 × 轮数"轮换执行，汇总中位数并做对拍（内部已有，尚未放入仓库）。
- 可选：JFR 与 nsys，用于 breakdown 与 kernel 核对。

模式名（沿用内部命名，仅供对照结果表）：`cpu`、`cpu-mb`、`cpu-1p`、`gpu`、`gpu-1p`、`gpu-batch`、`gpu-persistent`、`gpu-adaptive`、`*-fuse` / `*-nofuse`、`gpu-min1`；调优模式在后面加 `-tuned`。

注意事项：

- 运行期间不要修改正在被读取的 shell 脚本；内部曾因此出现个别轮次被误判为失败（数据本身有效，已计入）。
- 定速档的延迟按计划发送时刻计算；打满档延迟只作相对比较。
- 并发运行会改变吞吐（初筛里并发数字多数偏高 3–6%，个别用例有离群值），对比请在独占节点上进行。
