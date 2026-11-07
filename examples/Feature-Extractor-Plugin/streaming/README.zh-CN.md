<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Feature-Extractor-Plugin：streaming（Flink）

[English](README.md) | 中文

Feature-Extractor-Plugin 的流式部分，计划做成一个小的 Flink GPU 算子库。用户在作业里显式调用，只用在 GPU 值得上的少数几步。它不覆盖 Flink 的规划器类，结果会与 Flink 自己的 CPU 输出对拍。算子将从研究性插件 [Mega-Flink](https://github.com/cb521/Mega-Flink) 中抽取。**库的代码还没放进这个目录。**这里目前只有设计依据的分析和基准测试结果。

这一页回答：

1. [Flink 作业的时间花在哪](#1-flink-作业的时间花在哪)，GPU 能去掉其中多少；
2. [什么时候用 CPU，什么时候用 GPU](#2-什么时候用-cpu什么时候用-gpu)；
3. [库里会有什么](#3-组件组成)；
4. [我们跑的六个测试用例](#4-测试用例与跳转链接)和结果；
5. [这个目录里有什么](#5-这个目录里有什么)；
6. [其他项目怎么加速 Flink](#6-相关工作其他项目是怎么加速-flink-的)。

**我们数字的测量条件**：KuaiRand-27K 日志（9,292 万行）按事件时间回放，没有 Kafka，堆内状态；1 张 NVIDIA L20；Flink 2.2.1，用 Mega-Flink 的构建；除注明外并行度为 1；不限速打满，预热 30 万条后测 300 万条，3 轮取中位数；CPU 和 GPU 两边都开 `pipeline.object-reuse=true`；CPU 与 GPU 的最终结果表按 SHA-256 对拍，全部一致。没有在其他 GPU、多卡、其他数据集或集群上测过。基准报告在评审后改写过，改写后尚未独立复核。详见 [benchmarks/README.zh-CN.md](benchmarks/README.zh-CN.md)。

## 1. Flink 作业的时间花在哪

每类瓶颈的公开证据、常见解法和我们的完整实测见 [docs/bottlenecks.zh-CN.md](docs/bottlenecks.zh-CN.md)。每个公开数字只对应一家公司的一个场景。我们的实测摘要：

| 类别 | 时间花在哪 | 我们的实测 |
|---|---|---|
| 序列化与复制 | Kryo/POJO、算子间深拷贝 | c1 过滤，object-reuse 关到开：CPU 0.84M 到 2.47M 条/秒，GPU 1.03M 到 2.06M |
| 状态读写 | RocksDB 的序列化、seek、compaction；堆内哈希查找 | Interval Join 状态占采样时间 52%（CPU）/ 43%（GPU）；窗口聚合 14%（CPU）；去重 14% / 17% |
| Source、Sink、解析 | JSON/Avro/PB 解码、连接器 | source 线程占采样时间 10% 到 18%；Kafka 和真实格式没测 |
| 行列转换 | RowData 与列格式互转；GPU 上还有 H2D/D2H | c1 在 GPU 上：转换占计时的 GPU 步骤的 88%，kernel 占 5% |
| JVM 流水线上限 | 一个 source 线程、交换、一个 sink 线程 | 无计算探针：CPU 0.856M，GPU 0.851M 条/秒；TopN 一例里 GPU 空闲超过 95% |
| 倾斜、checkpoint、网络、GC | 稳定性、尾延迟、恢复 | 没测 |

时间构成是估算，来自更早一次用 Flink 默认配置（object-reuse 关）跑的 JFR 采样。

![时间花在哪](docs/images/time-breakdown.zh-CN.svg)

图和表一起看。两边的时间大头都是序列化与交换、状态读写、行列转换，GPU kernel 只占很小一块。所以光靠 kernel 更快，很难让 Flink 作业变快。

## 2. 什么时候用 CPU，什么时候用 GPU

### 2.1 一眼看结论

**适合**：窗口聚合（HOP 和 TUMBLE）和对维表的 INNER 查询，条件是流量高、每批足够大。我们调优后的实测里，窗口聚合 GPU 快 25% 到 26%，维表查询 GPU 快 76%。

**几乎没收益**：过滤/投影、TopN、keep-last 去重、Interval Join。这些步骤 CPU 快 11% 到 22%，因为成本在序列化、JVM 状态读写和行列转换，不在计算。

**预期**：这是单卡、一个数据集、主要在并行度 1 下的结果。只有这一步够重才会有收益，且要和已调优的 CPU 作业比。

### 2.2 为什么不是什么都上 GPU

GPU 只替换"计算"这一步。Flink 作业的时间大头在序列化和交换、source 回放、状态读写，以及 JVM 侧的行列转换。我们分析的算子里，GPU kernel 只占 3% 到 5%。TopN 一例里 GPU 超过 95% 的时间空闲（kernel 约 108 ms，整个作业约 3.2 s）。对每一步要问的是：它够不够重，值不值得付出打包、拷贝、解包的代价。

### 2.3 第一条：先把 CPU 侧调好

仅 `pipeline.object-reuse` 一项就翻转了过滤的结论。默认配置下 GPU 快 23%（103 万对 83 万条/秒）。两边都开后 CPU 快 20%（247 万对 206 万）。CPU 提升 2.95 倍，GPU 提升 2.0 倍。TopN 另把 `table.exec.rank.topn-cache-size` 调到 100000。其他 CPU 选项没有试过。

### 2.4 第二条：分步骤的实测结果（调优后、打满、并行度 1）

| 步骤 | CPU | GPU（batch） | 谁赢 | 其他并行度 |
|---|---|---|---|---|
| 过滤 / 投影 | 247 万/秒 | 206 万/秒 | CPU +20% | 未测 |
| HOP 窗口聚合（两阶段） | 148 万/秒 | 186 万/秒 | GPU +26% | 并行度 2 +26%，4 +16% |
| TUMBLE 窗口聚合 | 161 万/秒 | 202 万/秒 | GPU +25% | 未测 |
| 每用户 TopN（append-only） | 136 万/秒 | 123 万/秒 | CPU +11% | CPU +38%（2），+21%（4） |
| keep-last 去重 | 103 万/秒 | 85 万/秒 | CPU +22% | 未测 |
| Interval Join（INNER） | 29.2 万/秒 | 25.3 万/秒 | CPU +15% | 未测 |
| Interval Join 后接过滤 / 聚合 | 30.8 万 / 29.3 万 | 26.2 万 / 23.6 万 | CPU +17% / +24% | 未测 |
| 维表查询（INNER，单 BIGINT key） | 130 万/秒 | 228 万/秒 | GPU +76% | 并行度 2 +88%，4 +20%；400 万行维表 +83% |

- **选 CPU**：过滤/投影、TopN、去重、Interval Join。这些步骤的成本在序列化、JVM 状态读写和行列转换，GPU 换不掉。每秒 2 万、10 万条时两边都跟得上，GPU 多约 3 ms 延迟，CPU 过滤延迟不到 0.01 ms。
- **选 GPU**：窗口聚合和 INNER 维表查询，条件是流量高、每批足够大。每条记录要进多个窗口或做一次查表。HOP 两阶段 p50 延迟打满时 36 ms 对 48 ms，每秒 10 万条时 37 ms 对 63 ms。维表查询在低、中档多半走的是 JVM 查表（每批不足 1024 行会回落），这两档的延迟不能算作 GPU 的结果。
- **边界**：维表查询只测了 INNER，且只有单个 BIGINT key 和少量 BIGINT 列；本构建下 LEFT 无法上 GPU。维表 104.0 万行，GPU 上限是 4,194,304 行，3,204 万行全表没有测。**没测过**：其他 key 类型、更重的表达式、多卡、并行度大于 4、去重和 Interval Join 在并行度 2/4 下的表现、KuaiRand 以外的负载。不要把这张表当成通用规律。

### 2.5 第三条：GPU 怎么用

- **攒批（batch）**：我们的数据只支持这一种。窗口聚合和维表查询在打满时赢；流量要足够大才能攒满批。
- **常驻（persistent）**：我们的数据不支持。TopN 打满时只有 43 万条/秒，CPU 是 136 万；Interval Join 各档都更差（p99 约 200 ms）。这些数字除 TopN 打满一项外都来自默认配置。
- **自适应（adaptive）**：低、中档 0 行上 GPU。打满时 TopN 不如 batch，去重与 batch 相当。没有发现它胜过 CPU 和 batch 中较好者的情形。

### 2.6 第四条：连续几步留在 GPU（融合）

融合让数据在两个 GPU 步骤之间留在 GPU，不搬回主机。Interval Join 后接过滤或聚合，两边调优后融合没有吞吐收益（join +0.8%，join+过滤 +1.1%，均在噪声内；join+聚合 −7%），只有 join+聚合的中位延迟更低（打满低 6.5%，每秒 10 万条时低 12%）。两边最大的成本都是 JVM 里的 Join 状态。其他算子组合**未验证**。

### 2.7 选 GPU 前的检查清单

1. 先把 CPU 作业调好（至少 `pipeline.object-reuse`）并测一次。
2. 跑 qualification 和 profiling 工具（见第 3 节），确认重的部分正是你打算挪走的那一步。
3. 只挪窗口聚合或上述形状的 INNER 维表查询，或你自己测过的形状。
4. 用你的数据做 CPU/GPU 对比，并核对最终输出。

## 3. 组件组成

状态：**已有** = Mega-Flink 里现在就有；**计划** = 打算抽取或新写到这里（方向已定：显式调用的库，不覆盖规划器）；**尚未开始**。抽取方案仍是草稿，部分问题未定（见 3.2）。

**第一版我们的建议：窗口聚合 + 有界 INNER 维表查询；其余为可选，只有重新测量后确实赢才放进去。这是建议，不是决定。**

### 3.1 组件

| 组件 | 说明 | 现状 |
|---|---|---|
| GPU 算子库 | 由用户在 Flink 作业里显式调用的算子（拟用 DataStream API 加 builder），不覆盖规划器。可选：去重、TopN、Interval Join、带过滤/聚合融合的 INNER Join | 算子和 kernel 在 Mega-Flink 里**已有**，绑定在规划器上。**计划**：抽取；本目录暂无代码 |
| builder API 与形状校验 | 构建时检查类型、append-only、key 非空 | **尚未开始** |
| Table/SQL 桥接 | Table 转流、调用 GPU 算子、再转回 | **尚未开始**；是否放进第一版未定 |
| 原生层（CUDA、JNI） | 行列打包、显存管理、需要的 kernel | kernel **已有**。**计划**：拷贝并裁剪 |
| 选档逻辑 | 按流量选 CPU、常驻或攒批；回退 CPU | **已有**（自适应控制器）。**计划**：带过来；默认值要重新核对，因为 2.5 没看到自适应的收益 |
| qualification / profiling 工具 | 纯 Python，不需要 GPU。判断作业是否值得做 CPU/GPU 对照；看热点、反压、倾斜。只读访问 Flink REST 或历史归档。按 SQL 节点名识别算子，纯 DataStream 作业识别不了 | **已有**（源码 0.4.0，发布包 0.2.0，对应 Flink 2.2.1），含可选的 XGBoost 预测（是否带入未定）。**计划**：调整命名 |
| 基准与对拍脚本 | 按多档速率回放 KuaiRand，跑 CPU/GPU 各变体，按 SHA-256 对拍 | **已有**，但不在本目录。**计划**：路径改成参数后公开 |
| 示例作业 | 与基准用例对应的 KuaiRand 示例 | **尚未开始** |
| GPU 指标与告警模板 | Prometheus 规则、Grafana 面板、运维手册 | **已有**，是静态候选，未部署；是否带入未定 |

不打算放进库的：规划器覆盖层、安装器，以及在 Mega-Flink 自己的测试里没有赢过 CPU 的算子（LEFT/SEMI/ANTI Join、单独的 Calc、COUNT DISTINCT、SESSION 窗口、Cross Join）。

### 3.2 待定设计决策（都没定）

1. Java 包名：保留原名，还是改名（约 100 个 JNI 符号要替换并重测）。
2. 版权行和 NOTICE 的写法。
3. 第一版之外再加哪些算子（Window Join、GroupAggregate、Over，以及可选候选）。
4. Flink 版本：只承诺 2.2.x，还是做 2.0 及以上的编译矩阵（2.0、2.1 能否编译未验证）。
5. Table 桥接是否进第一版。
6. 工具里的 XGBoost 子命令是否保留，native 侧的 Calc 解释器是否裁掉。

这份算子清单原本依据 Mega-Flink 自己的基准结论（例如 TopN、去重比 CPU 快）。在我们调优后的测试里这两项是 CPU 赢，不在清单里的维表查询反而 GPU 赢。负载和 CPU 设置不同，尚未对齐，所以第一版以我们自己的实测为准。已知风险：从 SQL 作业改用本库，算子状态名会变，原 SQL 作业的 savepoint 无法恢复；用到的部分 Flink 类标注为 `@Internal`，小版本升级时可能变化。

## 4. 测试用例与跳转链接

我们在 NVIDIA L20 单卡上，用 KuaiRand 公开数据集，把六类常见的推荐特征 / 样本计算写成 Flink SQL，分别在 Flink CPU 和 GPU（Mega-Flink）上运行对比。测量方法、数据、结果表、已知局限和复现要求见 [benchmarks/README.zh-CN.md](benchmarks/README.zh-CN.md)。下列结论取自调优后（两边都开 object-reuse）打满档的吞吐。该文档也有默认配置下的数字，部分结论不同（例如 c1）。

| 用例 | 做什么（场景） | 结论（调优后，打满档） | 链接 |
| --- | --- | --- | --- |
| c1 过滤清洗 | 按时长、播放、随机曝光等条件过滤，并计算观看比例、时长分桶、互动次数（样本清洗） | CPU 更快：2.47M 对 2.06M 条/秒（CPU +20%）；低/中档延迟 CPU <0.01 ms，GPU 约 3 ms | [c1](benchmarks/README.zh-CN.md#c1-过滤清洗) |
| c2 窗口聚合 | 每用户最近 30 分钟（5 分钟步长 HOP）及 5 分钟 TUMBLE 的曝光数、点击数、播放时长（实时统计特征） | GPU 更快：HOP 两阶段 1.86M 对 1.48M（GPU +26%），TUMBLE 2.02M 对 1.61M（GPU +25%）；延迟也更低 | [c2](benchmarks/README.zh-CN.md#c2-每用户最近-30-分钟点击数的窗口聚合hop-与-tumble) |
| c3 每用户 TopN | 每个用户观看时长最长的 10 条，ROW_NUMBER（Top-K 行为特征） | CPU 更快：1.36M 对 1.23M（CPU +11%）；并行度 2/4 下 CPU +38% / +21%。GPU 大部分时间空闲，瓶颈在 JVM 侧 | [c3](benchmarks/README.zh-CN.md#c3-每用户观看最久的前-10-个视频topn) |
| c4 keep-last 去重 | 按 (user_id, video_id) 保留最新一条（样本去重） | CPU 更快：1.03M 对 0.85M（CPU +22%） | [c4](benchmarks/README.zh-CN.md#c4-keep-last-去重) |
| c5 Interval Join 及融合 | 曝光流与点击流在 30 分钟区间内 join；另测后接 Filter、后接聚合，GPU 上多步融合（曝光-点击归因） | 吞吐 CPU 更快：join +15%、join-filter +17%、join-agg +24%；接聚合时 GPU 延迟 p50 低约 23%。融合在调优后没有吞吐收益，仅 join-agg 延迟略低 | [c5](benchmarks/README.zh-CN.md#c5-曝光--点击-interval-join及后接-filter--聚合的融合) |
| c6 Lookup Join | 事件流查视频维表，取 author_id，INNER（物品侧特征拼接） | GPU 更快：2.28M 对 1.30M（GPU +76%）；维表只装了 104 万行，未测全量；LEFT 在本构建不能上 GPU | [c6](benchmarks/README.zh-CN.md#c6-查特征维表-lookup-join) |

一句话归纳：本次数据下，GPU 在窗口聚合（c2）和维表查询（c6）上有收益。过滤、TopN、去重、Interval Join 上 CPU 不输，因为成本在序列化、JVM 状态读写和行列转换。这是单卡、并行度主要为 1 的结果。

## 5. 这个目录里有什么

| 路径 | 内容 |
| --- | --- |
| [README.md](README.md)、[README.zh-CN.md](README.zh-CN.md) | 本页 |
| [docs/bottlenecks.zh-CN.md](docs/bottlenecks.zh-CN.md) | 完整的瓶颈清单，含公开证据和我们的实测 |
| [docs/related-work.zh-CN.md](docs/related-work.zh-CN.md) | 其他项目怎么加速 Flink，每个一张流程图 |
| [docs/images/](docs/images) | 这些页面用到的图 |
| [benchmarks/README.zh-CN.md](benchmarks/README.zh-CN.md) | 六个测试用例的方法、结果和局限 |
| 还没放进来 | 算子库、基准测试脚本、示例作业和工具（见第 3 节） |

## 6. 相关工作：其他项目是怎么加速 Flink 的

我们比较五个把 Flink 的一部分执行换成更快引擎的项目，资料来自它们公开的代码、文档和博客（读于 2026 年 11 月）。各家性能数字所用的工作负载和基线不同，这里不做比较。逐个项目的流程图和说明见 [docs/related-work.zh-CN.md](docs/related-work.zh-CN.md)。

### 从三个角度比较"怎么接入 Flink"

这五个项目在三个互相独立的角度上不同：**深度**（沿着 Flink 的流程一路换到哪一段）、**单位**（一次判断"能不能用原生"看多大一块）、**时机**（什么时候做这个判断）。

| 项目 | 深度 | 判断单位 | 决定时机 | 不支持时 |
|---|---|---|---|---|
| 蚂蚁 Flex | 只在规划阶段：表达式（以及 Join / Correlate 条件里的表达式） | 表达式 | 规划时 | 退回 Flink，细到单个表达式 |
| Mega-Flink（我们） | 物理计划节点；GPU 算子在 Flink 自己的任务里运行 | 节点 | 规划时（运行时只在已选好的算子内部切换路径） | 按节点退回 CPU 节点 |
| Gluten-Flink | 节点、作业图翻译、任务入口、状态 | 节点 | 规划时 | 计算类节点让作业失败；数据源/输出退回 |
| StreamFusion | 节点、自己的算子、状态在 Rust | 一连串列式节点组成的一块 | 规划时（Flink 优化之后） | 整块退回 Flink（早期设计笔记写的是逐算子，未核实） |
| 华为 OmniStream | 节点（只写说明）、作业图、任务运行时、网络、状态 | 算子链（`JobVertex`） | 生成作业图时（任务启动时才创建原生算子） | 按算子链；SQL 作业只要有一条边不是直连，整个作业退回 |
| 阿里 Flash | 执行计划和状态（ForStDB），其余细节未核实 | 整条 SQL（博客另一处提到算子级回退，没有统一） | 规划时 | 整条查询走 Java 计划 |

Gluten-Flink 和 OmniStream 的判断单位都是节点，但 OmniStream 换得深得多；Flex 单位最小，深度也最浅。作者只对少数选择写过理由：StreamFusion（避免块内转换；单独换一个原生算子可能比 Flink 慢，0.58 倍和 0.81 倍）、阿里（状态落盘太慢）、Flex（Velox 暂时不支持有状态算子）。我们没有找到任何把单位、时机和状态位置联系起来的说法，所以只比较各家的位置，不解释原因。

### 各项目替换了哪几段

![总览](docs/images/flow-overview.zh-CN.svg)
列是一个 Flink SQL 作业依次经过的阶段。越往右，项目接管的 Flink 自身运行时越多。
**没有一家用官方扩展点。** Flink 源码里 `PlannerFactory`、`ExecNode`、`ExecNodeMetadata` 都标着 `@Internal`，[FLIP-32](https://cwiki.apache.org/confluence/spaces/FLINK/pages/103088143/FLIP-32+Restructure+flink-table+for+future+contributions) 只是让整个规划器可以替换，不是替换单个算子。我们在 dev 邮件列表和 JIRA 里没找到算子级替换的提案。没查的：能否通过 `CalciteConfig` / `Programs` 注入自定义规则，以及 `CompiledPlan` 能否带自定义 `ExecNode`。作为对照，Spark 有 [`SparkSessionExtensions`](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/SparkSessionExtensions.html)（标着实验性），spark-rapids 就是靠它接入的。

### 我们从中得到的（我们的判断，未验证）

- 状态是最主要的差别。阿里、华为、Gluten-Flink 把状态搬到原生侧，Flex 不碰有状态算子，Mega-Flink 把状态留在 JVM。我们的 Interval Join 实测里，状态读写占采样时间的 43% 到 52%，GPU kernel 接管不了这部分。
- 我们没找到把加速算子做成用户显式调用的库的项目，也没找到别的用 GPU 跑 Flink SQL 流式算子的项目。显式调用的库是否更好，目前没有证据支持任何一方。
