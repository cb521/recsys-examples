<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# 第三类：几乎没东西可融（Case E）

> 本文所有执行计划、融合判断、kernel 源码都是实跑时自动导出的原文，只做了换行、省略（标 `...`）和路径替换（数据目录写成 `<data>`）。
> 数据：KuaiRand-27K，转成 Parquet 后 **92,919,789 行**。
> 环境：1 张 NVIDIA L20，Spark 3.5.9 local[8]，spark-rapids 26.08.1，插件 `gpu-fusion-plugin-0.1.0`。
> 性能数字：预热 10 轮、各 20 个计时轮、两种配置每轮先后顺序交替；时间拆分来自另一次带 DEBUG 指标的 20 轮运行（预热 2 轮），所以总墙钟与性能表略有出入。

**这一类的特点**：表达式里只有 2 个运算（一次类型转换、一次除法），整个查询就是读文件、几乎不算东西、再写出。插件恰好达到融合门槛（`minOps=2`）所以融合了，但融合能省的显存读写远小于读写 Parquet 的时间。

**一句话结论**：这类查询**不值得用插件，也不会因此变慢或变快**：中位数 0.682 s vs 0.713 s（插件快 4.6%，13/20 轮更快，但同口径其它几次运行是 −0.6%~+1.2%，按 ±5% 算噪声）；结果与 spark-rapids 逐位相同。

## Case E：只读文件，只做一次除法

### 0. 这个 case 在算什么

读日志，只带出 3 列，再算 1 个新列 `play_time_ms / 1000`。运算量极小，用来说明**运算太少时融合没有意义**。

```sql
-- Expected: the plugin may fuse it, but the Parquet scan dominates, so no gain.
SELECT
  user_id,
  video_id,
  is_click,
  play_time_ms / 1000D                                        AS play_sec
FROM log
```

### 1. 原始 Spark（CPU）物理执行计划

```
*(1) Project [user_id#0, video_id#1, is_click#5, (cast(play_time_ms#12 as double) / 1000.0) AS play_sec#124]
+- *(1) ColumnarToRow
   +- FileScan parquet [user_id#0,video_id#1,is_click#5,play_time_ms#12] Batched: true, DataFilters: [], Format: Parquet, Location: InMemoryFileIndex(1 paths)[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,is_click:int,play_time_ms:int>

```

### 2. spark-rapids 翻译后的 GPU 物理执行计划（基线）

`GpuColumnarToRow → GpuProject [user_id, video_id, is_click, (cast(play_time_ms as double) / 1000.0) AS play_sec] → GpuFileGpuScan`。
`GpuProject` 里只有 2 个运算（一次类型转换、一次除法），spark-rapids 本来就是各调一次 cuDF，一共 2 次调用。

### 3. 插件看到了什么、决定融合什么

```
== Fusion decisions (SAFE) ==
FUSED project: fused 2 ops -> 1 kernel fused_4e611515e875cbba; 1 kernel outputs, 1 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s) 
```

2 个运算刚好达到 `minOps=2` 的门槛，所以**融合了**：转换 + 除法 → 1 个 kernel，读 1 列、写 1 列。

### 4. 融合后的执行计划和 kernel

```
GpuColumnarToRow false
+- FusedGpuProjectExec [user_id#0, video_id#1, is_click#5, (cast(play_time_ms#12 as double) / 1000.0) AS play_sec#134], true, fused 2 ops -> 1 kernel fused_4e611515e875cbba; 1 kernel outputs, 1 inputs, 0 cuDF-evaluated sub-expr(s), 0 cuDF output(s)
   +- GpuFileGpuScan parquet [user_id#0,video_id#1,is_click#5,play_time_ms#12] Batched: true, DataFilters: [], Eager_IO_Prefetch: false, Format: Parquet, Location: InMemoryFileIndex[file:<data>/log], PartitionFilters: [], PushedFilters: [], ReadSchema: struct<user_id:int,video_id:int,is_click:int,play_time_ms:int>
```

kernel 主体（`kernels/fused_4e611515e875cbba.cu`，共 240 行，前面是 device function 定义）：

```cuda
extern "C" __global__ void fused_4e611515e875cbba(int n, int* __restrict__ flag,
    const i32* __restrict__ c0, const u32* __restrict__ m0,
    f64* __restrict__ o0, u32* __restrict__ om0)
{
  const int stride = gridDim.x * blockDim.x;
  const int lane = threadIdx.x & 31;
  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
  const int i = base + threadIdx.x;
  const bool inb = i < n;
  i32 v1 = ld_i32(c0, i, inb);
  bool n1 = inb && vld(m0, i);
  f64 v2 = (n1 ? (f64)v1 : (f64)0);
  bool n2 = n1;
  f64 v3 = __longlong_as_double(4652007308841189376LL);
  bool n3 = true;
  f64 v4 = ((n2 && n3 && !is_zero(v3)) ? d_div(v2, v3) : (f64)0);
  bool n4 = n2 && n3 && !is_zero(v3);
  if (inb) o0[i] = st_f64(v4);
  { u32 b = __ballot_sync(0xffffffffu, inb && n4); if (om0 != nullptr && lane == 0 && inb) om0[i >> 5] = b; }
  }
}
```

融合只是把"转换"和"除法"两次 cuDF 调用并成一次，中间那一列 double 省了一次显存读写。但这一列只有 92,919,789 × 8 字节 ≈ 0.7 GB，
在显存带宽下是毫秒级的事，而整个查询要 0.7 秒。

### 5. 性能和结果一致性

**一致性**：92,919,789 行，spark-rapids vs 插件 **0 行差异**；CPU vs GPU、CPU vs 插件也都是 0 行差异。

**性能**（读 Parquet + 1 次除法 + 写 Parquet）：

| 配置 | 次数 | 最小 | 中位数 | 最大 | 相对 spark-rapids 基线 |
| --- | --- | --- | --- | --- | --- |
| CPU Spark（参考） | 5 | 4.793 | 5.139 | 6.511 | spark-rapids 比它快 7.2 倍 |
| spark-rapids（基线） | 20 | 0.667 | 0.713 | 0.815 | — |
| spark-rapids + 插件 | 20 | 0.645 | 0.682 | 0.905 | +4.6%（正数 = 插件更快）；20 轮里插件更快的有 13 轮 |

单位秒，整个查询的墙钟时间。spark-rapids 和插件各 20 次，**预热 10 轮不计**（此前预热 2 轮时，最前面几个计时轮会偏慢，所以加到 10 轮），**每一轮里两者先后顺序交替**（偶数轮基线先、奇数轮插件先），同一个 Spark 会话、同一份数据、同样的 spark-rapids 设置，只差 `spark.rapids.sql.fusion.mode`（OFF / SAFE）。CPU 只是参考，单独放在最前面跑 5 次（另有 1 次预热）。

### 6. 时间拆分（准备 / 读 / 算 / 写）

| 阶段（墙钟秒，中位数） | spark-rapids | + 插件 | 差（插件−基线） | 实测还是折算 |
| --- | --- | --- | --- | --- |
| 准备（规划、列文件、作业调度、提交） | 0.129 | 0.129 | +0.001 | **实测**：总墙钟 − 所有 Spark 作业耗时 |
| 读 | 0.213 | 0.201 | -0.012 | 折算：作业耗时 × 扫描算子耗时 ÷ 全部任务运行时间 |
| 算 | 0.016 | 0.002 | -0.014 | 折算：同上，用除扫描/写入/交换外各算子的耗时 |
| 写/输出 | 0.359 | 0.378 | +0.019 | 折算：同上，用写文件算子的耗时 |
| 其他(未被上述指标解释的任务时间) | 0.002 | 0.002 | +0.000 | 折算：剩余部分 |
| **总墙钟** | **0.716** | **0.713** | **-0.003** | **实测** |

| 任务累计秒（8 个线程相加，中位数） | spark-rapids | + 插件 | 差 |
| --- | --- | --- | --- |
| 读 | 1.231 | 1.226 | -0.005 |
| 算 | 0.096 | 0.014 | -0.082 |
| 写/输出 | 2.114 | 2.248 | +0.133 |

作业里 8 个任务线程并行，“任务累计秒”是各线程时间相加；“读 / 算 / 写”的墙钟是按各类算子耗时在任务总时间里的占比分摊作业耗时得到的**估计值，不是直接测量**；“准备”和总墙钟是实测。

- **融合的内容：** 2 个运算（转换 + 除法）→ 1 个 kernel。实测算这一段：`GpuProject` op time 0.096 → `FusedGpuProjectExec` 0.014（任务累计秒，逐轮差中位数 −0.083，19/20 轮降）；折算墙钟 0.016 → 0.002。nsys：算段 kernel 33 → 22 个/轮，GPU 耗时 0.005 → 0.004 s。
- **但这 0.08 被写段抵消了：** 写文件节点 op time 2.114 → 2.248（+0.13；逐轮差中位数 +0.067，8/20 轮插件更低），主要是 GPU encode and buffer 1.430 → 1.560。结果：任务总运行时间逐轮差中位数 +0.008，≈0。读段 scan time 1.231 vs 1.226，没变。
- 为什么写段变多：**没查清**（nsys 里写段 kernel 个数和耗时一样：462 个，1.005 vs 1.002 s；写段 CPU 墙钟 1.319 vs 1.389 s，写段同步等待 1.281 vs 1.225 s，方向不一致）。A 里也有同样的"写段多一点"（+0.08），C 里只有 +0.03。
- 总之 E 里插件能省的（0.08 任务累计秒 ≈ 0.016 s 墙钟）不到总墙钟 0.7 s 的 3%，在噪声里看不出来。

### 7. 结论

- 插件中位数 0.682 s，基线 0.713 s，快约 4.6%，20 轮里插件更快的有 13 轮；但同口径的另几次运行里是 −0.6%~+1.2%，范围也宽（两者最小到最大都在 0.65~0.9 s），**按"±5% 以内算噪声"的原则，不当作收益**。
- 这个 case 说明：运算只有两三个时，融合省下的那点显存读写远小于读写 Parquet 的时间，**不值得用**，也不会带来稳定可见的变化。
- 是否值得融合的粗略判断：表达式里的运算数、以及整个查询里 I/O 之外的时间占比。运算数越多、I/O 越少，越可能有收益。
