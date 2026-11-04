<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 第二类：能融合，但占整体时间太小（Case C、Case D）

> 本文所有执行计划、融合判断、kernel 源码都是实跑时自动导出的原文，只做了换行、省略（标 `...`）和路径替换（数据目录写成 `<data>`）。
> 数据：KuaiRand-27K，转成 Parquet 后 **92,919,789 行**。
> 环境：1 张 NVIDIA L20，Spark 3.5.9 local[8]，spark-rapids 26.08.1，插件 `gpu-fusion-plugin-0.1.0`。
> 性能数字：预热 10 轮、各 20 个计时轮、两种配置每轮先后顺序交替；时间拆分来自另一次带 DEBUG 指标的 20 轮运行（预热 2 轮），所以总墙钟与性能表略有出入。

**这一类的特点**：查询里确实有可以融合的表达式（C：过滤 + 5 个特征；D：一个 3 运算的过滤），插件也真的融合了，融合算子自己的耗时也降了；但它们在整条查询里占的时间很小——C 的大头是读、写 Parquet，D 的大头是拼表后的 shuffle。

**一句话结论**：这一类里插件**既不变快、也不变慢**：C 中位数 0.722 s vs 0.711 s（10/20 轮插件更快），D 中位数 2.346 s vs 2.349 s（10/20），差距都在噪声以内；结果与 spark-rapids 逐位相同。

## Case C：先过滤样本，再算特征（写 Parquet）

### 0. 这个 case 在算什么

先丢掉无效和过短的曝光，只保留信息流流量，再对留下的行算 5 个特征。过滤和特征都是逐行运算，是样本生成里最常见的"先筛后算"。

```sql
SELECT
  user_id,
  video_id,
  least(play_time_ms / (duration_ms + 1D), 5D)                AS watch_ratio,
  log(1D + play_time_ms / 1000D)                              AS log_play_sec,
  CASE WHEN duration_ms < 15000 THEN 0
       WHEN duration_ms < 60000 THEN 1 ELSE 2 END             AS duration_bucket,
  is_like + is_follow + is_comment + is_forward               AS interact_cnt,
  is_click
FROM log
WHERE duration_ms > 0
  AND play_time_ms >= 0
  AND is_rand = 0
  AND (play_time_ms > 3000 OR is_click = 1)
```

过滤后剩 **45,611,516 行**（原 92,919,789 行）。

### 1. 原始 Spark（CPU）物理执行计划

```
*(1) Project [user_id#0, video_id#1, least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#124, LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#125, CASE WHEN (duration_ms#13 < 15000) THEN 0 WHEN (duration_ms#13 < 60000) THEN 1 ELSE 2 END AS duration_bucket#126, (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#127, is_click#5]
+- *(1) Filter ((((((isnotnull(duration_ms#13) AND isnotnull(play_time_ms#12)) AND isnotnull(is_rand#17)) AND (duration_ms#13 > 0)) AND (play_time_ms#12 >= 0)) AND (is_rand#17 = 0)) AND ((play_time_ms#12 > 3000) OR (is_click#5 = 1)))
   +- *(1) ColumnarToRow
      +- FileScan parquet [user_id#0,video_id#1,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,play_time_ms#12,duration_ms#13,is_rand#17] Batched: true, DataFilters: [isnotnull(duration_ms#13), isnotnull(play_time_ms#12), isnotnull(is_rand#17), (duration_ms#13 > ..., Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(duration_ms), IsNotNull(play_time_ms), IsNotNull(is_rand), GreaterThan(duration_ms,0),..., ReadSchem ...

```

读法：`Filter` 和 `Project` 都在 `*(1)` 里，即同一段 Java 代码生成；过滤条件里多出的 `isnotnull(...)` 是 Spark 自动补的（`>`、`=` 遇到 NULL 本来就是不满足）。

### 2. spark-rapids 翻译后的 GPU 物理执行计划（基线）

基线计划就是第 4 步去掉"Fused"字样的样子：`FileScan → GpuFilter → GpuCoalesceBatches → GpuProject → GpuColumnarToRow`
（完整原文见同目录导出的 `plan_gpu.txt`）。`GpuFilter` 把条件里的每个运算各调一次 cuDF，得到一列布尔值再筛行；
`GpuProject` 同样每个运算一次 cuDF 调用。

### 3. 插件看到了什么、决定融合什么

```
== Fusion decisions (SAFE) ==
FUSED filter: fused 15 ops -> 1 mask kernel filt_10d674684724f049; 4 inputs, 0 cuDF-evaluated sub-expr(s) 
FUSED project: fused 14 ops -> 1 kernel fused_727b555473fc3846; 4 kernel outputs, 6 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s) 
```

- **过滤**：条件里的 15 个运算（3 个 `IS NOT NULL`、4 个比较、`AND`/`OR`）融成 **1 个"掩码 kernel"**，只读 4 列、直接写出一列布尔掩码；筛行仍由 cuDF 完成。
- **投影**：5 个特征里的 4 个算出列（`watch_ratio`、`log_play_sec`、`duration_bucket`、`interact_cnt`）共 14 个运算融成 **1 个 kernel**，读 6 列；`user_id`、`video_id`、`is_click` 原样带出。
- 没有需要交给 cuDF 的子表达式。

### 4. 融合后的执行计划和 kernel

```
GpuColumnarToRow false
+- FusedGpuProjectExec [user_id#0, video_id#1, gpuleast((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#146, LOG_BASE((1.0 + (cast(play_time_ms#12 as double) / 1000.0)), 2.718281828459045) AS log_play_sec#147, CASE WHEN (duration_ms#13 < 15000) THEN 0 WHEN (duration_ms#13 < 60000) THEN 1 ELSE 2 END AS duration_bucket#148, (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#149, is_click#5], true, fused 14 ops -> 1 kernel fused_727b555473fc3846; 4 kernel outputs, 6 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
   +- GpuCoalesceBatches targetsize(1073741824)
      +- FusedGpuFilterExec ((((((gpuisnotnull(duration_ms#13) AND gpuisnotnull(play_time_ms#12)) AND gpuisnotnull(is_rand#17)) AND (duration_ms#13 > 0)) AND (play_time_ms#12 >= 0)) AND (is_rand#17 = 0)) AND ((play_time_ms#12 > 3000) OR (is_click#5 = 1))), fused 15 ops -> 1 mask kernel filt_10d674684724f049; 4 inputs, 0 cuDF-evaluated sub-expr(s)
         +- GpuFileGpuScan parquet [user_id#0,video_id#1,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,play_time_ms#12,duration_ms#13,is_rand#17] Batched: true, DataFilters: [isnotnull(duration_ms#13), isnotnull(play_time_ms#12), isnotnull(is_rand#17), (duration_ms#13 > ..., Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(duration_ms), IsNotNull(play_time_ms), IsNotNull(is_rand), GreaterThan(duration_ms,0),..., ReadSchema: struct<user_id:int,video_id:int,is_click:int,is_like:int,is_follow:int,is_comment:int,is_forward:...
```

过滤掩码 kernel（`kernels/filt_10d674684724f049.cu`，共 281 行，前面是 device function 定义，下面是 kernel 主体，中间省略）：

```cuda
extern "C" __global__ void filt_10d674684724f049(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,
    const i32* __restrict__ c1, const u32* __restrict__ m1,
    const i32* __restrict__ c2, const u32* __restrict__ m2,
    const i32* __restrict__ c3, const u32* __restrict__ m3,
    bool* __restrict__ o0, u32* __restrict__ om0)
{
  const int stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
  const int i = base + threadIdx.x;
  const bool inb = i < n;
  i32 v1 = ld_i32(c0, i, inb);
  bool n1 = inb && vld(m0, i);
  bool v2 = n1;
  bool n2 = true;
  i32 v3 = ld_i32(c1, i, inb);
  bool n3 = inb && vld(m1, i);
  bool v4 = n3;
  bool n4 = true;
  bool v5 = (n2 && n4 && v2 && v4);
  bool n5 = ((n2 && !v2) || (n4 && !v4) || (n2 && n4));
  i32 v6 = ld_i32(c2, i, inb);
  bool n6 = inb && vld(m2, i);
  bool v7 = n6;
  bool n7 = true;
  bool v8 = (n5 && n7 && v5 && v7);
  bool n8 = ((n5 && !v5) || (n7 && !v7) || (n5 && n7));
  i32 v9 = (i32)0;
  bool n9 = true;
  ...   // 15 个运算：3 个 IS NOT NULL、4 个比较、AND / OR，全在寄存器里完成
  if (inb) o0[i] = st_bool(v23);
  { u32 b = __ballot_sync(0xffffffffu, inb && n23); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
}
```

对照第 2 步：基线每个比较、每个 AND 都是一次 cuDF 调用并写出一整列布尔；这里 15 个运算的中间值都在寄存器里，最后只写一列掩码。
投影 kernel 的结构同[第一类文档](group1-fusion-pays-off.md)里 Case A 的 kernel，只是更短（`kernels/fused_727b555473fc3846.cu`，共 306 行）。

### 5. 性能和结果一致性

**一致性**（过滤后 45,611,516 行，逐行比对）：spark-rapids vs 插件 **0 行差异**；CPU vs GPU 有 2,093,458 行不同，来源同第一类的 Case A（`log_play_sec` 最后一位）。

**性能**（读 Parquet + 过滤 + 特征 + 写 Parquet）：

| 配置 | 次数 | 最小 | 中位数 | 最大 | 相对 spark-rapids 基线 |
| --- | --- | --- | --- | --- | --- |
| CPU Spark（参考） | 5 | 5.241 | 5.472 | 6.125 | spark-rapids 比它快 7.7 倍 |
| spark-rapids（基线） | 20 | 0.699 | 0.711 | 0.775 | — |
| spark-rapids + 插件 | 20 | 0.692 | 0.722 | 0.828 | -1.4%（正数 = 插件更快）；20 轮里插件更快的有 10 轮 |

单位秒，整个查询的墙钟时间。spark-rapids 和插件各 20 次，**预热 10 轮不计**（此前预热 2 轮时，最前面几个计时轮会偏慢，所以加到 10 轮），**每一轮里两者先后顺序交替**（偶数轮基线先、奇数轮插件先），同一个 Spark 会话、同一份数据、同样的 spark-rapids 设置，只差 `spark.rapids.sql.fusion.mode`（OFF / SAFE）。CPU 只是参考，单独放在最前面跑 5 次（另有 1 次预热）。

### 6. 时间拆分（准备 / 读 / 算 / 写）

| 阶段（墙钟秒，中位数） | spark-rapids | + 插件 | 差（插件−基线） | 实测还是折算 |
| --- | --- | --- | --- | --- |
| 准备（规划、列文件、作业调度、提交） | 0.157 | 0.163 | +0.006 | **实测**：总墙钟 − 所有 Spark 作业耗时 |
| 读 | 0.340 | 0.333 | -0.007 | 折算：作业耗时 × 扫描算子耗时 ÷ 全部任务运行时间 |
| 算 | 0.024 | 0.009 | -0.015 | 折算：同上，用除扫描/写入/交换外各算子的耗时 |
| 写/输出 | 0.196 | 0.203 | +0.006 | 折算：同上，用写文件算子的耗时 |
| 其他(未被上述指标解释的任务时间) | 0.001 | 0.001 | +0.000 | 折算：剩余部分 |
| **总墙钟** | **0.733** | **0.716** | **-0.017** | **实测** |

| 任务累计秒（8 个线程相加，中位数） | spark-rapids | + 插件 | 差 |
| --- | --- | --- | --- |
| 读 | 2.037 | 1.937 | -0.100 |
| 算 | 0.145 | 0.053 | -0.092 |
| 写/输出 | 1.154 | 1.175 | +0.021 |

作业里 8 个任务线程并行，“任务累计秒”是各线程时间相加；“读 / 算 / 写”的墙钟是按各类算子耗时在任务总时间里的占比分摊作业耗时得到的**估计值，不是直接测量**；“准备”和总墙钟是实测。

- **融合的内容：** 过滤 15 个运算 → 1 个掩码 kernel，投影 14 个运算 → 1 个 kernel。实测算这一段：`GpuFilter` 0.050 + `GpuProject` 0.081 = 0.130（任务累计秒）→ `FusedGpuFilterExec` 0.016 + `FusedGpuProjectExec` 0.024 = 0.039（逐轮差中位数 −0.090，20/20 轮降）。nsys：算段 kernel 个数 1001 → 220，这段 kernel GPU 耗时 0.046 → 0.022 s，算段 stream sync 次数 2453 → 583（等待 0.082 → 0.033 s）。
- **能省的就这 0.09（任务累计秒），折算成墙钟只有 0.015 s**（算：0.024 → 0.009）。C 的"读"占大头（scan 2.0，其中 `filter time`——reader 里的行组过滤——就占 1.5），所以融合那一点点被盖住。
- 读段 scan time −0.069（逐轮中位数），去掉信号量等待后 −0.046，但四分位范围 [−0.33, −0.01] 很宽，只有 16/20 轮降；GPU decode 没变（+0.005）。**这个读段的差异在噪声内，不能算插件的收益**；与 A 不同，C 里算段的节省太小，不足以让别的线程少排队。
- 写段 +0.026（6/20 轮插件更低），噪声内。
- 对账：任务总运行时间逐轮差中位数 −0.197，算段 −0.09 + 读段 −0.07 + 写段 +0.03 ≈ −0.13，大体对得上（余下的是未单独列出的算子）。

### 7. 结论

- 结果逐位相同；插件中位数 0.722 s，基线 0.711 s，20 轮里插件更快的有 10 轮，**没有可测的差别**（此前另一次同口径运行得到插件快约 2%，两次方向不同，说明差距在噪声内）。
- **C 的内容能融合（过滤 15 个运算、投影 14 个运算，各融成 1 个 kernel），融合算子本身的耗时也确实降了**（过滤 + 投影从 0.13 降到 0.04，任务线程耗时相加的秒数），但这一段在整条查询里占比太小：时间的大头是读 Parquet（扫描）和写 Parquet，所以端到端看不出变化，也没有变慢。
- 因此 C 属于“能融合但占比太小”，**不是“证明插件有收益”的 case**。它能带来收益的前提是计算占比更大（见第一类）。

## Case D：拼表 + 字符串特征 + 分组统计

### 0. 这个 case 在算什么

把曝光日志分别和用户表、视频表拼起来，对视频标签取前两个字符，按（用户活跃度，视频类型，标签前缀）分组统计曝光数、点击率、长播率。
这是"特征工程里拼表 + 聚合"的典型形态，**几乎没有逐行数值运算可融合**，用来验证插件在这种场景下**不会拖慢**。

```sql
-- Expected: little or nothing for the plugin to fuse.
SELECT
  u.user_active_degree,
  v.video_type,
  substring(v.tag, 1, 2)                                      AS tag_prefix,
  count(*)                                                    AS exposures,
  avg(l.is_click)                                             AS ctr,
  avg(l.long_view)                                            AS long_view_rate
FROM log l
JOIN user u        ON l.user_id = u.user_id
JOIN video_basic v ON l.video_id = v.video_id
GROUP BY u.user_active_degree, v.video_type, substring(v.tag, 1, 2)
```

### 1. 原始 Spark（CPU）物理执行计划

```
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[user_active_degree#39, video_type#102, _groupingexpression#137], functions=[count(1), avg(is_click#5), avg(long_view#11)], output=[user_active_degree#39, video_type#102, tag_prefix#124, exposures#125L, ctr#126, long_view_rate#127])
   +- Exchange hashpartitioning(user_active_degree#39, video_type#102, _groupingexpression#137, 8), ENSURE_REQUIREMENTS, [plan_id=81]
      +- HashAggregate(keys=[user_active_degree#39, video_type#102, _groupingexpression#137], functions=[partial_count(1), partial_avg(is_click#5), partial_avg(long_view#11)], output=[user_active_degree#39, video_type#102, _groupingexpression#137, count#143L, sum#144, count#145L, sum#146, count#147L])
         +- Project [is_click#5, long_view#11, user_active_degree#39, video_type#102, substring(tag#111, 1, 2) AS _groupingexpression#137]
            +- SortMergeJoin [video_id#1], [video_id#100], Inner
               :- Sort [video_id#1 ASC NULLS FIRST], false, 0
               :  +- Exchange hashpartitioning(video_id#1, 8), ENSURE_REQUIREMENTS, [plan_id=73]
               :     +- Project [video_id#1, is_click#5, long_view#11, user_active_degree#39]
               :        +- BroadcastHashJoin [user_id#0], [user_id#38], Inner, BuildRight, false
               :           :- Filter (isnotnull(user_id#0) AND isnotnull(video_id#1))
               :           :  +- FileScan parquet [user_id#0,video_id#1,is_click#5,long_view#11] Batched: true, DataFilters: [isnotnull(user_id#0), isnotnull(video_id#1)], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(user_id), IsNotNull(video_id)], ReadSc ...
               :           +- BroadcastExchange HashedRelationBroadcastMode(List(cast(input[0, int, false] as bigint)),false), [plan_id=68]
               :              +- Filter isnotnull(user_id#38)
               :                 +- FileScan parquet [user_id#38,user_active_degree#39] Batched: true, DataFilters: [isnotnull(user_id#38)], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/user], PartitionFilters: [], PushedFilters: [IsNotNull(user_id)], ReadSchema: struct<user_id:int,user_active_degree:string>
               +- Sort [video_id#100 ASC NULLS FIRST], false, 0
                  +- Exchange hashpartitioning(video_id#100, 8), ENSURE_REQUIREMENTS, [plan_id=74]
                     +- Filter isnotnull(video_id#100)
                        +- FileScan parquet [video_id#100,video_type#102,tag#111] Batched: true, DataFilters: [isnotnull(video_id#100)], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/video_basic], PartitionFilters: [], PushedFilters: [IsNotNull(video_id)], ReadSchema: struct<video_id:int,video_type:string,tag ...

```

读法：两次 join（日志 × 用户表用广播 join，× 视频表用排序归并 join），Project 里有字符串 `substring`，最后按三个键分组聚合。
CPU 上能做的"融合"只是 `*(n)` 的 Java 代码生成。

### 2. spark-rapids 翻译后的 GPU 物理执行计划（基线）

所有主要算子都能上 GPU：`GpuBroadcastHashJoin`、`GpuShuffledSymmetricHashJoin`、`GpuHashAggregate`、`GpuProject`（含 `gpusubstring`）、`GpuFilter`。
完整原文见第 4 步（插件只替换了其中一个算子，其余与基线相同）。

### 3. 插件看到了什么、决定融合什么

```
== Fusion decisions (SAFE) ==
FUSED filter: fused 3 ops -> 1 mask kernel filt_804b8531df4b562e; 2 inputs, 0 cuDF-evaluated sub-expr(s) 
NOT FUSED GpuFilter (u.user_id IS NOT NULL): region too small (1 op(s) < minOps=2)
NOT FUSED GpuFilter (v.video_id IS NOT NULL): region too small (1 op(s) < minOps=2)
NOT FUSED GpuProject [is_click, long_view, user_active_degree, video_type, gpusubstring(v.tag, 1, 2) AS _groupingexpression]: no fusable expression; output 4 (gpusubstring(v.tag, 1, 2) AS _groupingexpression): GpuSubstring is not in the fusion whitelist
```

逐项解释：

| 位置 | 决定 | 原因 |
| --- | --- | --- |
| 日志表上的过滤 `user_id IS NOT NULL AND video_id IS NOT NULL` | **融合**（3 个运算 → 1 个掩码 kernel） | 运算数达到门槛 |
| 用户表上的过滤 `user_id IS NOT NULL` | 不融合 | 只有 1 个运算，低于门槛 `minOps=2`：融合省不下什么，还要付编译和启动的代价 |
| 视频表上的过滤 `video_id IS NOT NULL` | 不融合 | 同上，1 个运算 |
| `GpuProject [..., substring(tag,1,2)]` | 不融合 | 唯一的计算列是 `substring`，字符串函数不在插件支持名单内；其余列只是原样带出 |
| join、聚合、shuffle | 不碰 | 不是逐行表达式，不在插件范围 |

### 4. 融合后的执行计划和 kernel

```
GpuColumnarToRow false
+- GpuHashAggregate (keys=[user_active_degree#39, video_type#102, _groupingexpression#319], functions=[gpucount(1, false), avg(is_click#5, DoubleType, false), avg(long_view#11, DoubleType, false)], output=[user_active_degree#39, video_type#102, tag_prefix#232, exposures#233L, ctr#234, long_view_rate#235]) 
   +- GpuShuffleCoalesce 1073741824
      +- GpuColumnarExchange gpuhashpartitioning(user_active_degree#39, video_type#102, _groupingexpression#319, 8, Murmur3Mode), ENSURE_REQUIREMENTS, [plan_id=952]
         +- GpuHashAggregate (keys=[user_active_degree#39, video_type#102, _groupingexpression#319], functions=[partial_gpucount(1, false), partial_avg(is_click#5, DoubleType, false), partial_avg(long_view#11, DoubleType, false)], output=[user_active_degree#39, video_type#102, _groupingexpression#319, count#325L, sum#326, count# ...
            +- GpuProject [is_click#5, long_view#11, user_active_degree#39, video_type#102, gpusubstring(tag#111, 1, 2) AS _groupingexpression#319], true
               +- GpuShuffledSymmetricHashJoin Inner, [video_id#1], [video_id#100], false, 1073741824, 1.0, CoalesceReadOption(true,GPU,Never,None,false), false
                  :- GpuColumnarExchange gpuhashpartitioning(video_id#1, 8, Murmur3Mode), ENSURE_REQUIREMENTS, [plan_id=948]
                  :  +- GpuProject [video_id#1, is_click#5, long_view#11, user_active_degree#39], true
                  :     +- GpuBroadcastHashJoin [user_id#0], [user_id#38], Inner, GpuBuildRight, false
                  :        :- GpuCoalesceBatches targetsize(1073741824)
                  :        :  +- FusedGpuFilterExec (gpuisnotnull(user_id#0) AND gpuisnotnull(video_id#1)), fused 3 ops -> 1 mask kernel filt_804b8531df4b562e; 2 inputs, 0 cuDF-evaluated sub-expr(s)
                  :        :     +- GpuFileGpuScan parquet [user_id#0,video_id#1,is_click#5,long_view#11] Batched: true, DataFilters: [isnotnull(user_id#0), isnotnull(video_id#1)], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [IsNotNull(user_id), Is ...
                  :        +- GpuBroadcastExchange HashedRelationBroadcastMode(List(cast(input[0, int, false] as bigint)),false), [plan_id=910]
                  :           +- GpuCoalesceBatches targetsize(1073741824)
                  :              +- GpuFilter gpuisnotnull(user_id#38)
                  :                 +- GpuFileGpuScan parquet [user_id#38,user_active_degree#39] Batched: true, DataFilters: [isnotnull(user_id#38)], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/user], PartitionFilters: [], PushedFilters: [IsNotNull(user_id)], ReadSchema: struct<user_id:int, ...
                  +- GpuColumnarExchange gpuhashpartitioning(video_id#100, 8, Murmur3Mode), ENSURE_REQUIREMENTS, [plan_id=915]
                     +- GpuCoalesceBatches targetsize(1073741824)
                        +- GpuFilter gpuisnotnull(video_id#100)
                           +- GpuFileGpuScan parquet [video_id#100,video_type#102,tag#111] Batched: true, DataFilters: [isnotnull(video_id#100)], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/video_basic], PartitionFilters: [], PushedFilters: [IsNotNull(video_id)], ReadSchema: struct<video_id ...
```

整棵计划里只有一处变化：日志表上的 `GpuFilter` → `FusedGpuFilterExec`，其余算子原样。

唯一生成的 kernel 是这个过滤的掩码 kernel（`kernels/filt_804b8531df4b562e.cu`，共 1 行，前面是 device function 定义，下面是 kernel 主体）：

```cuda
extern "C" __global__ void filt_804b8531df4b562e(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,
    const i32* __restrict__ c1, const u32* __restrict__ m1,
    bool* __restrict__ o0, u32* __restrict__ om0)
{
  const int stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
  const int i = base + threadIdx.x;
  const bool inb = i < n;
  i32 v1 = ld_i32(c0, i, inb);
  bool n1 = inb && vld(m0, i);
  bool v2 = n1;
  bool n2 = true;
  i32 v3 = ld_i32(c1, i, inb);
  bool n3 = inb && vld(m1, i);
  bool v4 = n3;
  bool n4 = true;
  bool v5 = (n2 && n4 && v2 && v4);
  bool n5 = ((n2 && !v2) || (n4 && !v4) || (n2 && n4));
  bool v6 = (v5 && n5);
  bool n6 = true;
  if (inb) o0[i] = st_bool(v6);
  { u32 b = __ballot_sync(0xffffffffu, inb && n6); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
  }
}
```

3 个运算（两个 `IS NOT NULL` 和一个 `AND`）融成一个 kernel，读 2 列、写 1 列布尔掩码，筛行仍由 cuDF 完成。

### 5. 性能和结果一致性

**一致性**：分组结果 1,021 行，spark-rapids vs 插件、CPU vs spark-rapids、CPU vs 插件 **都是 0 行差异**。

**性能**（读三张表 + 拼表 + 聚合 + 写 Parquet）：

| 配置 | 次数 | 最小 | 中位数 | 最大 | 相对 spark-rapids 基线 |
| --- | --- | --- | --- | --- | --- |
| CPU Spark（参考） | 5 | 12.490 | 12.806 | 15.411 | spark-rapids 比它快 5.5 倍 |
| spark-rapids（基线） | 20 | 2.269 | 2.349 | 2.693 | — |
| spark-rapids + 插件 | 20 | 2.298 | 2.346 | 2.544 | +0.1%（正数 = 插件更快）；20 轮里插件更快的有 10 轮 |

单位秒，整个查询的墙钟时间。spark-rapids 和插件各 20 次，**预热 10 轮不计**（此前预热 2 轮时，最前面几个计时轮会偏慢，所以加到 10 轮），**每一轮里两者先后顺序交替**（偶数轮基线先、奇数轮插件先），同一个 Spark 会话、同一份数据、同样的 spark-rapids 设置，只差 `spark.rapids.sql.fusion.mode`（OFF / SAFE）。CPU 只是参考，单独放在最前面跑 5 次（另有 1 次预热）。

### 6. 时间拆分（准备 / 读 / 算 / 写）

| 阶段（墙钟秒，中位数） | spark-rapids | + 插件 | 差（插件−基线） | 实测还是折算 |
| --- | --- | --- | --- | --- |
| 准备（规划、列文件、作业调度、提交） | 0.111 | 0.114 | +0.003 | **实测**：总墙钟 − 所有 Spark 作业耗时 |
| 读 | 0.218 | 0.217 | -0.001 | 折算：作业耗时 × 扫描算子耗时 ÷ 全部任务运行时间 |
| 算 | 0.104 | 0.099 | -0.005 | 折算：同上，用除扫描/写入/交换外各算子的耗时 |
| 交换(shuffle) | 1.397 | 1.404 | +0.007 | 折算：同上，用 shuffle 读 + 分区 + 序列化三项 |
| 写/输出 | 0.004 | 0.004 | +0.000 | 折算：同上，用写文件算子的耗时 |
| 其他(未被上述指标解释的任务时间) | 0.581 | 0.582 | +0.000 | 折算：剩余部分 |
| **总墙钟** | **2.421** | **2.418** | **-0.002** | **实测** |

| 任务累计秒（8 个线程相加，中位数） | spark-rapids | + 插件 | 差 |
| --- | --- | --- | --- |
| 读 | 1.301 | 1.294 | -0.007 |
| 算 | 0.626 | 0.593 | -0.033 |
| 交换(shuffle) | 8.354 | 8.393 | +0.039 |
| 写/输出 | 0.022 | 0.022 | -0.000 |

作业里 8 个任务线程并行，“任务累计秒”是各线程时间相加；“读 / 算 / 写”的墙钟是按各类算子耗时在任务总时间里的占比分摊作业耗时得到的**估计值，不是直接测量**；“准备”和总墙钟是实测。

- **插件融合的只有一处：** 日志表上 3 个运算的过滤。实测 `GpuFilter` op time 0.026 → `FusedGpuFilterExec` 0.009（任务累计秒，省 0.019）；其余两个 1 运算的过滤、`substring` 投影都没融合（`GpuProject` 0.064 vs 0.064，没变），与融合决策一致。
- **这点节省相对于整体可以忽略：** 整个查询任务累计 13.8 秒（8 个线程并行，墙钟 2.4 s），0.019 是 0.1%。D 的大头是**交换（shuffle）**：序列化 4.65 + 反序列化读 3.12 + 分区 0.59（任务累计秒；这几项有重叠，只能说是大头），shuffle 写出 1.5 GB；其次是读 1.3、join/聚合等算子 0.6。折算墙钟：读 0.218、算 0.104、交换 1.397、其他 0.58。
- nsys 对比（8 轮）：kernel 总数 4289 vs 4256（少 33 个，正好是被融合的那个过滤），GPU 总耗时 0.374 vs 0.371 s，同步次数 6718 vs 6704，两边 H2D 3.8 GB / D2H 3.25 GB 完全一样。
- 墙钟 2.421 vs 2.418（40 轮）：没有可测的差别，也没有变慢（40 轮里插件更快的 21 轮）。

### 7. 结论

- D 说明**插件不会让不适合的查询变慢**（也不会变快）：中位数 2.346 s vs 2.349 s，差不到 0.2%，20 轮里插件更快的有 10 轮，没有可测的差别。
- 查询里的主要算子是拼表、shuffle、聚合，都不是插件管的范围（shuffle 是最大的一块）；插件只融合了一个 3 运算的过滤，其余两个 1 运算的过滤因 `minOps=2` 被主动放弃。
- 判断依据都能在 `fusion_decisions.txt` 里查到：不融合的每一处都写明了原因。
