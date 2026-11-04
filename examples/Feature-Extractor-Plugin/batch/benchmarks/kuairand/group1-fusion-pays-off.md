<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 第一类：融合有收益（Case A、Case B）

> 本文所有执行计划、融合判断、kernel 源码都是实跑时自动导出的原文，只做了换行、省略（标 `...`）和路径替换（数据目录写成 `<data>`）。
> 数据：KuaiRand-27K，转成 Parquet 后 **92,919,789 行**。
> 环境：1 张 NVIDIA L20，Spark 3.5.9 local[8]，spark-rapids 26.08.1，插件 `gpu-fusion-plugin-0.1.0`。
> 性能数字：预热 10 轮、各 20 个计时轮、两种配置每轮先后顺序交替；时间拆分来自另一次带 DEBUG 指标的 20 轮运行（预热 2 轮），所以总墙钟与性能表略有出入。

**这一类的特点**：每一行要算很多个特征（10 个特征、27 个运算，包括类型转换、除法、`least`、`log`、`CASE`、`pmod` 等）。spark-rapids 对每个运算各调一次 cuDF，每一步都把整列写进显存、下一步再读出来，而且每次调用后常要等一次同步；插件把这 27 个运算融成 **1 个 kernel**，中间值都留在寄存器里。

**一句话结论**：这一类里插件**稳定更快**——写 Parquet 的 Case A 快约 **6.6%**（0.759 s vs 0.809 s，20 轮里 17 轮更快），不写文件、结果留在 GPU 上的 Case B 快约 **21%**（0.299 s vs 0.361 s，20 轮全部更快）；B 去掉了文件读写的干扰，可以看作这套数据、这张卡上的上限。结果与 spark-rapids 逐位相同。

## Case A：常见样本特征（从 Parquet 读，写回 Parquet）

### 0. 这个 case 在算什么

对每一条曝光日志算 10 个特征和标签，是推荐样本生成里最常见的一类逐行计算（每个特征的出处见 [README](README.md#where-the-features-come-from)）：

```sql
SELECT
  user_id, video_id,
  CAST(hourmin / 100 AS INT)                              AS hour_of_day,     -- 小时
  dayofweek(CAST(time_ms DIV 1000 AS TIMESTAMP))          AS day_of_week,     -- 星期几
  least(play_time_ms / (duration_ms + 1D), 5D)            AS watch_ratio,     -- 观看比例，截断到 5
  log(1D + play_time_ms / 1000D)                          AS log_play_sec,    -- 播放秒数取 log
  CASE WHEN duration_ms < 7000 THEN 0 ... ELSE 5 END      AS duration_bucket, -- 视频时长分 6 档
  CASE WHEN play_time_ms >= duration_ms THEN 1 ELSE 0 END AS complete_play,   -- 是否看完
  is_like + is_follow + is_comment + is_forward           AS interact_cnt,    -- 互动次数
  is_click, long_view,                                                        -- 原样带出的标签
  pmod(user_id * 1000003 + tab, 1048576)                  AS user_tab_cross   -- 用户 × 频道 哈希交叉
FROM log
```

### 1. 原始 Spark（CPU）物理执行计划

```
*(1) Project [user_id#0, video_id#1,
              cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#124,
              dayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#125,
              least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#126,
              LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#127,
              CASE WHEN (duration_ms#13 < 7000) THEN 0 ... ELSE 5 END AS duration_bucket#128,
              CASE WHEN (play_time_ms#12 >= duration_ms#13) THEN 1 ELSE 0 END AS complete_play#129,
              (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#130,
              is_click#5, long_view#11,
              pmod(((user_id#0 * 1000003) + tab#18), 1048576) AS user_tab_cross#131]
+- *(1) ColumnarToRow
   +- FileScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,...,tab#18] Batched: true, Format: Parquet, ...
```

读法：
- 从下往上执行。`FileScan` 读 Parquet，`ColumnarToRow` 把列转成行，`Project` 逐行算出所有新列。
- `*(1)` 表示 Spark 把 `Project` 和行转换放进同一段 Java 代码生成（WholeStageCodegen），也就是 **CPU 上的"融合"**。
- Spark 自己补了很多类型转换：`play_time_ms` 是 int，除以 double 前先 `cast(... as double)`；`log(x)` 被改写成 `LOG(e, x)`。

### 2. spark-rapids 翻译后的 GPU 物理执行计划（基线）

```
GpuColumnarToRow false
+- GpuProject [user_id#0, video_id#1,
               cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#144,
               gpudayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#145,
               gpuleast((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#146,
               LOG_BASE((1.0 + (cast(play_time_ms#12 as double) / 1000.0)), 2.718281828459045) AS log_play_sec#147,
               CASE WHEN (duration_ms#13 < 7000) THEN 0 ... ELSE 5 END AS duration_bucket#148,
               CASE WHEN (play_time_ms#12 >= duration_ms#13) THEN 1 ELSE 0 END AS complete_play#149,
               (((is_like#6 + is_follow#7) + is_comment#8) + is_forward#9) AS interact_cnt#150,
               is_click#5, long_view#11,
               (((user_id#0 * 1000003) + tab#18) pmod 1048576) AS user_tab_cross#151], true
   +- GpuFileGpuScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,...,tab#18] Batched: true, Format: Parquet, ...
```

和第 1 步比，spark-rapids 做了三件事：
1. `FileScan` → `GpuFileGpuScan`：在 GPU 上解码 Parquet，数据直接以 cuDF 列格式放进显存。
2. `Project` → `GpuProject`：每个表达式换成 GPU 版（如 `dayofweek` → `gpudayofweek`，`least` → `gpuleast`，`LOG` → `LOG_BASE`）。
3. 中间的"列转行"没了，只在最后结果要交回 Spark 时才 `GpuColumnarToRow`。

**问题在 `GpuProject` 怎么执行**：它对表达式树里的每一个运算（转换、除法、加法、比较、CASE ……）各调一次 cuDF，
每一步都把结果写成一整列放进显存，下一步再读出来。

### 3. 插件看到了什么、决定融合什么

插件挂在 spark-rapids 之后（`postColumnarTransitions`），看到第 2 步的 `GpuProject`，逐个输出列翻译，导出的判断原文：

```
FUSED project: fused 27 ops -> 1 kernel fused_88a94965d48afb79; 8 kernel outputs, 10 inputs,
  1 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
  cut: CAST((log.time_ms div 1000L) AS TIMESTAMP) evaluated by cuDF (cast i64 -> ts not supported)
```

逐项解释：

| 输出列 | 处理方式 | 原因 |
| --- | --- | --- |
| `user_id`、`video_id`、`is_click`、`long_view` | 原样带出，不进 kernel | 只是取列，没有计算 |
| `hour_of_day`、`watch_ratio`、`log_play_sec`、`duration_bucket`、`complete_play`、`interact_cnt`、`user_tab_cross` | 进 kernel | 所有运算都有对应的 device function |
| `day_of_week` | 进 kernel，但其中 `CAST(time_ms DIV 1000 AS TIMESTAMP)` 这一小段先交给 cuDF 算 | 插件还不支持 bigint → timestamp 的转换。这一小段由 cuDF 算好后当作一列输入喂给 kernel，`dayofweek` 本身仍在 kernel 里算 |

结果：**8 个输出列、27 个运算 → 1 个 kernel**，kernel 读 10 个输入列（其中 1 列是 cuDF 预先算好的时间戳）。

### 4. 融合后的执行计划和 kernel

执行计划里 `GpuProject` 被换成 `FusedGpuProjectExec`，其余节点不变：

```
GpuColumnarToRow false
+- FusedGpuProjectExec [user_id#0, video_id#1, ... 同第 2 步的表达式 ...], true,
     fused 27 ops -> 1 kernel fused_88a94965d48afb79; 8 kernel outputs, 10 inputs, 1 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
   +- GpuFileGpuScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,...,tab#18] ...
```

运行时插件现场生成的 CUDA 源码（`kernels/fused_88a94965d48afb79.cu`，共 381 行，前 219 行是所有 device function 的定义，
下面是 kernel 主体，省略了部分重复行）：

```cuda
extern "C" __global__ void fused_88a94965d48afb79(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,     // 输入列 hourmin + NULL 位图
    const i64* __restrict__ c1, const u32* __restrict__ m1,     // cuDF 预先算好的时间戳
    ...                                                         // c2..c9：play_time_ms、duration_ms、is_like ... tab
    i32* __restrict__ o0, u32* __restrict__ om0,                // 输出 hour_of_day + NULL 位图
    ...                                                         // o1..o7：其余 7 个特征
{
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {   // 一个线程一行
  const int i = base + threadIdx.x;
  // hour_of_day = CAST(hourmin / 100.0 AS INT)
  i32 v1 = ld_i32(c0, i, inb);            bool n1 = inb && vld(m0, i);        // 读值 + 读是否 NULL
  f64 v2 = (n1 ? (f64)v1 : (f64)0);       bool n2 = n1;
  f64 v3 = 100.0;                         bool n3 = true;                     // 原文是 __longlong_as_double(...) 形式的常量
  f64 v4 = ((n2 && n3 && !is_zero(v3)) ? d_div(v2, v3) : (f64)0);             // 除以 0 得 NULL，照 Spark
  i32 v5 = (n4 ? f2i_i32(v4) : (i32)0);
  // day_of_week：时间戳 → 天数 → 星期几；日期越界就立"出界"标志，这批数据改走不融合路径
  i64 v6 = ld_ts(c1, i, inb);
  i32 v7 = (n6 ? (i32)fdiv64(v6, 86400000000LL) : (i32)0);
  if (inb && (n7 && (v7 > 2146000000 || v7 < -2146000000))) bail(flag);
  i32 v8 = ((n7) ? dt_dow(v7) : (i32)0);
  // watch_ratio = least(play_time_ms / (duration_ms + 1.0), 5.0)
  f64 v14 = ((n12 && n13) ? w_add(v12, v13) : (f64)0);
  f64 v15 = ((n10 && n14 && !is_zero(v14)) ? d_div(v10, v14) : (f64)0);
  f64 v17 = ((n15 && n16) ? sel_f64(c_lt(v15, v16), v15, v16) : (n16 ? v16 : v15));
  // log_play_sec：v10（play_time_ms 转成的 double）在上面已经读过，这里直接复用，不再读显存
  f64 v19 = ((n10 && n18 && !is_zero(v18)) ? d_div(v10, v18) : (f64)0);
  f64 v20 = ((n13 && n19) ? w_add(v13, v19) : (f64)0);
  f64 v22 = ((n20 && n21 && !(v20 <= 0.0) && !(v21 <= 0.0)) ? d_logbase(v20, v21) : 0.0);
  // duration_bucket：5 个比较 + CASE，全在寄存器里
  bool v24 = ((n11 && n23) ? c_lt(v11, v23) : (bool)0);
  ...
  i32 v43 = ((n24 && v24) ? v25 : v42);
  // complete_play、interact_cnt、user_tab_cross
  bool v44 = ((n9 && n11) ? c_ge(v9, v11) : (bool)0);
  i32 v52 = ((n50 && n51) ? w_add(v50, v51) : (i32)0);
  i32 v59 = ((n57 && n58 && !is_zero(v58)) ? d_pmod(v57, v58) : (i32)0);
  // 写出 8 列结果，每 32 行用一次 __ballot_sync 拼出 NULL 位图
  if (inb) o0[i] = st_i32(v5);
  { u32 b = __ballot_sync(0xffffffffu, inb && n5); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
  ...
  }
}
```

对照第 2 步：spark-rapids 每个运算一次 cuDF 调用、中间结果都落显存；这里**每行只把要用的 10 列各读一次**，
27 个运算的中间结果都在寄存器里（`v1`…`v59`），最后只写 8 个输出列。

### 5. 性能和结果一致性

**一致性**（92,919,789 行，逐行比对）：

| 比对 | 只在左边出现的行 | 只在右边出现的行 |
| --- | --- | --- |
| spark-rapids vs 插件 | 0 | 0 |
| CPU vs spark-rapids | 5,067,411 | 5,067,411 |
| CPU vs 插件 | 5,067,411 | 5,067,411 |

插件和 spark-rapids 的结果**逐位相同**。CPU 和 GPU 的差异来自同一处：`log_play_sec` 的最后一位（CPU 用 Java `StrictMath`，GPU 用 CUDA 数学库，spark-rapids 本身就有的已知差异），其它列都相同。

**性能**（写 Parquet，整个查询的墙钟时间）：

| 配置 | 次数 | 最小 | 中位数 | 最大 | 相对 spark-rapids 基线 |
| --- | --- | --- | --- | --- | --- |
| CPU Spark（参考） | 5 | 15.358 | 15.793 | 16.126 | spark-rapids 比它快 19.5 倍 |
| spark-rapids（基线） | 20 | 0.788 | 0.809 | 0.959 | — |
| spark-rapids + 插件 | 20 | 0.718 | 0.759 | 0.820 | +6.6%（正数 = 插件更快）；20 轮里插件更快的有 17 轮 |

单位秒，整个查询的墙钟时间。spark-rapids 和插件各 20 次，**预热 10 轮不计**（此前预热 2 轮时，最前面几个计时轮会偏慢，所以加到 10 轮），**每一轮里两者先后顺序交替**（偶数轮基线先、奇数轮插件先），同一个 Spark 会话、同一份数据、同样的 spark-rapids 设置，只差 `spark.rapids.sql.fusion.mode`（OFF / SAFE）。CPU 只是参考，单独放在最前面跑 5 次（另有 1 次预热）。

### 6. 时间拆分（准备 / 读 / 算 / 写）

| 阶段（墙钟秒，中位数） | spark-rapids | + 插件 | 差（插件−基线） | 实测还是折算 |
| --- | --- | --- | --- | --- |
| 准备（规划、列文件、作业调度、提交） | 0.193 | 0.196 | +0.003 | **实测**：总墙钟 − 所有 Spark 作业耗时 |
| 读 | 0.242 | 0.217 | -0.025 | 折算：作业耗时 × 扫描算子耗时 ÷ 全部任务运行时间 |
| 算 | 0.069 | 0.026 | -0.044 | 折算：同上，用除扫描/写入/交换外各算子的耗时 |
| 写/输出 | 0.298 | 0.312 | +0.014 | 折算：同上，用写文件算子的耗时 |
| 其他(未被上述指标解释的任务时间) | 0.001 | 0.001 | -0.000 | 折算：剩余部分 |
| **总墙钟** | **0.798** | **0.751** | **-0.047** | **实测** |

| 任务累计秒（8 个线程相加，中位数） | spark-rapids | + 插件 | 差 |
| --- | --- | --- | --- |
| 读 | 1.511 | 1.349 | -0.162 |
| 算 | 0.433 | 0.161 | -0.272 |
| 写/输出 | 1.847 | 1.960 | +0.113 |

作业里 8 个任务线程并行，“任务累计秒”是各线程时间相加；“读 / 算 / 写”的墙钟是按各类算子耗时在任务总时间里的占比分摊作业耗时得到的**估计值，不是直接测量**；“准备”和总墙钟是实测。

- **收益主要在"算"这一段，和插件融合的内容对得上。** 27 个运算融成 1 个 kernel：`GpuProject` 的 op time 中位数 0.433 → `FusedGpuProjectExec` 0.161（任务累计秒，20 轮里 20 轮都降，逐轮差中位数 −0.274）。nsys（8 轮）里算这一段的 kernel 个数 1001 → 209（每轮），这段 kernel 的 GPU 耗时 0.108 → 0.045 s，这段里的 `cudaStreamSynchronize` 次数 3234 → 1232、等待合计 0.312 → 0.133 s。也就是省的主要是"很多次小 cuDF 调用 + 每次调用后的同步等待"，不是 GPU 算得更快。
- **"读数据"那段少了 0.16~0.19（任务累计秒）：读本身没有变快，主要是等 GPU 的时间变少了。**
  - 源码：`scan time` 是扫描算子 `hasNext()` 的整段耗时，里面包含 reader 获取 GPU 信号量（`GpuSemaphore.acquireIfNecessary`）的等待；DEBUG 级别还会额外给出"去掉信号量等待"的版本。
  - 实测（DEBUG 指标，中位数）：scan time 1.511 → 1.349（−0.162）；去掉信号量等待后 0.846 → 0.800（−0.046）。**减少的 0.162 里约 0.116 是信号量等待，只有约 0.046 是别的**；GPU decode time 0.413 → 0.388（−0.025）。
  - nsys 旁证：读这一段的 kernel（解码）个数、耗时两边完全一样（1356 个，0.135 vs 0.128 s），`read_chunk` 区间的 CPU 墙钟 0.433 vs 0.436 s，读段里的同步等待 0.310 vs 0.303 s。
  - 解释（与数据一致，但没有直接量 GPU 信号量的持有时间）：每个任务线程在"算"这段占 GPU 的时间变短了，别的任务线程排队等 GPU 的时间随之变短，等待记在了它们的 scan time 里。**所以这 0.16 不是额外收益，是算这段收益的另一种体现，不能和算这段的 0.27 再相加。**
- **写这一段反而多了约 0.08（任务累计秒）**：写文件节点的 op time 1.847 → 1.960，逐轮 20 轮里只有 2 轮是插件更低；其中 GPU encode and buffer 0.747 → 0.812。nsys 里写段 kernel 个数、耗时一样（580 个，0.423 vs 0.416 s），写段 CPU 墙钟 0.703 vs 0.740 s。**原因没查清。**
- 对账：算 −0.27 + 读 −0.16 + 写 +0.11（中位数相减）≈ −0.33，任务总运行时间逐轮差中位数 −0.350，对得上。折算成墙钟：作业时间 −0.05 s，与总墙钟差（0.798 → 0.751）一致。

### 7. 结论

- 结果逐位相同；插件比 spark-rapids 快 **约 6.6%**（中位数 0.759 s vs 0.809 s，20 轮里 17 轮插件更快）。幅度不大：整个查询还要读、写 9300 万行 Parquet。
- 收益来自融合的那一段：逐算子的计时里，投影算子的耗时从 0.43 降到 0.16（8 个任务线程耗时相加的秒数，中位数，20 轮里 20 轮都降）；其余（读、写）没有被融合，没有变快。
- 想看到融合本身的收益，要把 I/O 去掉：见本文 Case B 一节（同样的 SQL，结果在 GPU 上汇总，不写文件）。

## Case B：同 A 的 SQL，结果在 GPU 上汇总（不回 CPU、不写文件）

### 0. 这个 case 在算什么

SQL 和 本文 Case A **完全一样**（10 个样本特征，见 Case A 的 0 小节），区别只在"算完之后做什么"：
不写 Parquet，而是把每个特征列在 GPU 上求和，只把一行汇总结果取回来。这模拟"特征算好后直接喂给 GPU 训练"，
也就是把文件读写之外的时间尽量压低，看融合本身能带来多少。（求和只是为了让每一列都真的被算出来，不被优化掉。）

### 1. 原始 Spark（CPU）物理执行计划

```
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[], functions=[count(1), sum(cast(user_id#0 as double)), sum(cast(video_id#1 as double)), sum(cast(hour_of_day#124 as double)), sum(cast(day_of_week#125 as double)), sum(watch_ratio#126), sum(log_play_sec#127), sum(cast(duration_bucket#128 as double)), sum(cast(complete_play#129 as double)), sum(cast(interact_cnt#130 as double)), sum(cast(is_click#5 as double)), sum(cast(long_view#11 as double)) ...
   +- Exchange SinglePartition, ENSURE_REQUIREMENTS, [plan_id=28]
      +- HashAggregate(keys=[], functions=[partial_count(1), partial_sum(cast(user_id#0 as double)), partial_sum(cast(video_id#1 as double)), partial_sum(cast(hour_of_day#124 as double)), partial_sum(cast(day_of_week#125 as double)), partial_sum(watch_ratio#126), partial_sum(log_play_sec#127), partial_sum(cast(duration_bucket#128 as double)), partial_sum(cast(complete_play#129 as double)), partial_sum(cast(interact_c ...
         +- Project [user_id#0, video_id#1, cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#124, dayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#125, least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#126, LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#127, CASE WHEN (duratio ...
            +- FileScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,long_view#11,play_time_ms#12,duration_ms#13,tab#18] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,hourmin:int,time_ms:bigint,is_click:int,is_li ...

```

读法：最上面是汇总（`HashAggregate`，先各分区内部求部分和，再 `Exchange` 汇到一个分区求总和）；
中间的 `Project` 就是 Case A 里那一大串特征表达式；底下是读文件。

### 2. spark-rapids 翻译后的 GPU 物理执行计划（基线）

```
AdaptiveSparkPlan isFinalPlan=false
+- HashAggregate(keys=[], functions=[count(1), sum(cast(user_id#0 as double)), sum(cast(video_id#1 as double)), sum(cast(hour_of_day#209 as double)), sum(cast(day_of_week#210 as double)), sum(watch_ratio#211), sum(log_play_sec#212), sum(cast(duration_bucket#213 as double)), sum(cast(complete_play#214 as double)), sum(cast(interact_cnt#215 as double)), sum(cast(is_click#5 as double)), sum(cast(long_view#11 as double)) ...
   +- Exchange SinglePartition, ENSURE_REQUIREMENTS, [plan_id=79]
      +- HashAggregate(keys=[], functions=[partial_count(1), partial_sum(cast(user_id#0 as double)), partial_sum(cast(video_id#1 as double)), partial_sum(cast(hour_of_day#209 as double)), partial_sum(cast(day_of_week#210 as double)), partial_sum(watch_ratio#211), partial_sum(log_play_sec#212), partial_sum(cast(duration_bucket#213 as double)), partial_sum(cast(complete_play#214 as double)), partial_sum(cast(interact_c ...
         +- Project [user_id#0, video_id#1, cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#209, dayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#210, least((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#211, LOG(2.718281828459045, (1.0 + (cast(play_time_ms#12 as double) / 1000.0))) AS log_play_sec#212, CASE WHEN (duratio ...
            +- FileScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,long_view#11,play_time_ms#12,duration_ms#13,tab#18] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,hourmin:int,time_ms:bigint,is_click:int,is_li ...

```

（这里导出的是自适应执行（AQE）开始前的计划外壳，算子名还是 Spark 原名，GPU 算子在执行时才替换进去。替换后的结构见第 4 步：把 `FusedGpuProjectExec` 看作 `GpuProject` 即是基线的样子，其余算子相同。）

### 3. 插件看到了什么、决定融合什么

```
== Fusion decisions (SAFE) ==
FUSED project: fused 27 ops -> 1 kernel fused_88a94965d48afb79; 8 kernel outputs, 10 inputs, 1 cuDF-evaluated sub-expr(s), 0 cuDF output(s) cut: CAST((log.time_ms div 1000L) AS TIMESTAMP) evaluated by cuDF (cast i64 -> ts not supported)
```

和 Case A 完全相同：8 个输出列、27 个运算融成 1 个 kernel（kernel 编号 `fused_88a94965d48afb79` 也一样，源码见 Case A 的 4 小节），
`CAST(time_ms DIV 1000 AS TIMESTAMP)` 暂不支持，由 cuDF 先算好再喂给 kernel。

### 4. 融合后的执行计划和 kernel

```
GpuColumnarToRow false
+- GpuHashAggregate (keys=[], functions=[gpucount(1, false), gpubasicsum(cast(user_id#0 as double), DoubleType, false), gpubasicsum(cast(video_id#1 as double), DoubleType, false), gpubasicsum(cast(hour_of_day#450 as double), DoubleType, false), gpubasicsum(cast(day_of_week#451 as double), DoubleType, false), gpubasicsum(watch_ratio#452, DoubleType, false), gpubasicsum(log_play_sec#453, DoubleType, false), gpubasicsum(cast(duration_bucket#454 as double), DoubleType, false), gpubasicsum(cast(complete_play#455 as double), DoubleType, false), gpubasicsum(cast(interact_cnt#456 as double), DoubleType, false), gpubasicsum(cast(is_click#5 as double), DoubleType, false), gpubasicsum(cast(long_view#11 ...
   +- GpuShuffleCoalesce 1073741824
      +- GpuColumnarExchange gpusinglepartitioning$(), ENSURE_REQUIREMENTS, [plan_id=244]
         +- GpuHashAggregate (keys=[], functions=[partial_gpucount(1, false), partial_gpubasicsum(cast(user_id#0 as double), DoubleType, false), partial_gpubasicsum(cast(video_id#1 as double), DoubleType, false), partial_gpubasicsum(cast(hour_of_day#450 as double), DoubleType, false), partial_gpubasicsum(cast(day_of_week#451 as double), DoubleType, false), partial_gpubasicsum(watch_ratio#452, DoubleType, false), partial_gpubasicsum(log_play_sec#453, DoubleType, false), partial_gpubasicsum(cast(duration_bucket#454 as double), DoubleType, false), partial_gpubasicsum(cast(complete_play#455 as double), DoubleType, false), partial_gpubasicsum(cast(interact_cnt#456 as double), DoubleType, false),  ...
            +- FusedGpuProjectExec [user_id#0, video_id#1, cast((cast(hourmin#3 as double) / 100.0) as int) AS hour_of_day#450, gpudayofweek(cast(cast((time_ms#4L div 1000) as timestamp) as date)) AS day_of_week#451, gpuleast((cast(play_time_ms#12 as double) / (cast(duration_ms#13 as double) + 1.0)), 5.0) AS watch_ratio#452, LOG_BASE((1.0 + (cast(play_time_ms#12 as double) / 1000.0)), 2.718281828459045) AS log_play_sec#453, CASE WHEN (duration_ms#13 < 7000) THEN 0 WHEN (duration_ms#13 < 15000) THEN 1 WHEN (duration_ms#13 < 30000) THEN 2 WHEN (duration_ms#13 < 60000) THEN 3 WHEN (duration_ms#13 < 120000) THEN 4 ELSE 5 END AS duration_bucket#454, CASE WHEN (play_time_ms#12 >= duration_ms#13) T ...
               +- GpuFileGpuScan parquet [user_id#0,video_id#1,hourmin#3,time_ms#4L,is_click#5,is_like#6,is_follow#7,is_comment#8,is_forward#9,long_view#11,play_time_ms#12,duration_ms#13,tab#18] Batched: true, DataFilters: [], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,hourmin:int,time_ms:bigint,is_click:int,is_like:int,is_follow:int...
```

`GpuProject` 换成了 `FusedGpuProjectExec`；上面的汇总算子（`GpuHashAggregate`）、下面的 `GpuFileGpuScan` 都不动。

生成的 kernel 与 Case A **是同一个**（`kernels/fused_88a94965d48afb79.cu`，381 行，同一份源码），完整源码和逐段注释见 Case A 的 4 小节；这里 kernel 的输入（10 列）、输出（8 列）和运算（27 个）与 Case A 完全一致，不重复贴。

### 5. 性能和结果一致性

**一致性**：92,919,789 行逐行比对，spark-rapids vs 插件 **0 行差异**（CPU vs GPU 的差异同 Case A：`log_play_sec` 最后一位）。

**性能**（读 Parquet + 特征计算 + GPU 汇总，只取回一行）：

| 配置 | 次数 | 最小 | 中位数 | 最大 | 相对 spark-rapids 基线 |
| --- | --- | --- | --- | --- | --- |
| CPU Spark（参考） | 5 | 3.650 | 3.773 | 3.950 | spark-rapids 比它快 10.5 倍 |
| spark-rapids（基线） | 20 | 0.336 | 0.361 | 0.408 | — |
| spark-rapids + 插件 | 20 | 0.267 | 0.299 | 0.354 | +20.7%（正数 = 插件更快）；20 轮里插件更快的有 20 轮 |

单位秒，整个查询的墙钟时间。spark-rapids 和插件各 20 次，**预热 10 轮不计**（此前预热 2 轮时，最前面几个计时轮会偏慢，所以加到 10 轮），**每一轮里两者先后顺序交替**（偶数轮基线先、奇数轮插件先），同一个 Spark 会话、同一份数据、同样的 spark-rapids 设置，只差 `spark.rapids.sql.fusion.mode`（OFF / SAFE）。CPU 只是参考，单独放在最前面跑 5 次（另有 1 次预热）。

### 6. 时间拆分（准备 / 读 / 算 / 写）

| 阶段（墙钟秒，中位数） | spark-rapids | + 插件 | 差（插件−基线） | 实测还是折算 |
| --- | --- | --- | --- | --- |
| 准备（规划、列文件、作业调度、提交） | 0.073 | 0.080 | +0.007 | **实测**：总墙钟 − 所有 Spark 作业耗时 |
| 读 | 0.187 | 0.180 | -0.007 | 折算：作业耗时 × 扫描算子耗时 ÷ 全部任务运行时间 |
| 算 | 0.118 | 0.061 | -0.058 | 折算：同上，用除扫描/写入/交换外各算子的耗时 |
| 交换(shuffle) | 0.001 | 0.001 | -0.000 | 折算：同上，用 shuffle 读 + 分区 + 序列化三项 |
| 其他(未被上述指标解释的任务时间) | 0.002 | 0.002 | +0.000 | 折算：剩余部分 |
| **总墙钟** | **0.388** | **0.321** | **-0.067** | **实测** |

| 任务累计秒（8 个线程相加，中位数） | spark-rapids | + 插件 | 差 |
| --- | --- | --- | --- |
| 读 | 1.228 | 1.100 | -0.128 |
| 算 | 0.792 | 0.394 | -0.398 |
| 交换(shuffle) | 0.006 | 0.006 | -0.000 |

作业里 8 个任务线程并行，“任务累计秒”是各线程时间相加；“读 / 算 / 写”的墙钟是按各类算子耗时在任务总时间里的占比分摊作业耗时得到的**估计值，不是直接测量**；“准备”和总墙钟是实测。

- **收益在"算"这一段，对得上。** 同 A 的 27 运算 1 kernel；`GpuProject` op time 0.547 → `FusedGpuProjectExec` 0.143（20 轮里 20 轮降，差中位数 −0.404）。聚合（`GpuHashAggregate`）0.250 vs 0.247，没变。
- 读这一段 scan time 1.228 → 1.100（−0.100），其中信号量等待 0.322 → 0.207（−0.115），去掉信号量后只差 −0.013；GPU decode −0.018。**和 A 一样：读本身没变，是等 GPU 的时间变少，属于算这段收益的间接体现。**
- 折算墙钟：算 0.118 → 0.061（−0.057），读 0.187 → 0.180（−0.007），与总墙钟差 0.388 → 0.321（−0.067）基本对上。
- B 没有写文件这一段；driver 侧固定开销 0.073 / 0.080（比 A 的 0.2 小，因为不写文件）。

### 7. 结论

- 结果与 spark-rapids 逐位相同。
- 去掉 Parquet 写入、结果留在 GPU 上之后，插件比 spark-rapids **快约 21%**（中位数 0.299 s vs 0.361 s，20 轮全部插件更快）。这是本组 case 里收益最大的一个。
- 即便如此收益也只有两成左右：插件能省的只是 `Project` 那一段（融合后 0.55 降到 0.14，任务线程耗时相加的秒数），读文件等其它部分没变。
- 这是本组 case 里对插件最有利的设置（算得多、不写文件），可以看作这套数据、这张卡上的大致上限。
