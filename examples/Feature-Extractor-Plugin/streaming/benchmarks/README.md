# Streaming Feature Computation Benchmarks: Flink CPU vs GPU (Mega-Flink)

This document describes the six performance cases (c1-c6) of the `streaming` part: how each is defined, how it is measured, what the results are, what the limitations are, and what is needed to reproduce them. The entry point for readers is "Part 5" of the parent README.

> Status: the numbers come from the latest rewrite of an internal measurement report. That rewrite, made after supplementary runs, has **not been independently re-reviewed**. An earlier version was independently reviewed and every comment was addressed. "Not verified" means no direct evidence was found.

## Contents

- [1. Overview](#overview)
- [2. Environment and dataset](#env)
- [3. Methodology](#method)
- [4. Cases and results](#cases): [c1](#c1-filter), [c2](#c2-window), [c3](#c3-topn), [c4](#c4-dedup), [c5](#c5-join), [c6](#c6-lookup)
- [5. Higher parallelism and a larger dimension table](#parallelism)
- [6. Where the time goes (breakdown and nsys)](#breakdown)
- [7. Known limitations](#limits)
- [8. Reproduction](#repro)

<a id="overview"></a>
## 1. Overview

Saturated-input throughput (records/s), single GPU, parallelism 1, exclusive node, with `pipeline.object-reuse` enabled on both sides ("tuned"):

| Case | CPU | GPU | Faster |
| --- | --- | --- | --- |
| c1 filter / cleaning | 2,472,421 | 2,059,561 | CPU +20% |
| c2 HOP two-phase | 1,475,619 | 1,861,833 | GPU +26% |
| c2 TUMBLE | 1,613,516 | 2,020,259 | GPU +25% |
| c3 TopN | 1,363,629 | 1,231,075 (batch) | CPU +11% |
| c4 dedup | 1,033,538 | 850,292 (batch) | CPU +22% |
| c5 join | 291,797 | 253,233 (fused) | CPU +15% |
| c5 join-filter | 307,921 | 262,297 (fused) | CPU +17% |
| c5 join-agg | 293,385 | 236,056 (fused) | CPU +24% (GPU p50 latency about 23% lower) |
| c6 Lookup (INNER) | 1,299,187 | 2,284,322 | GPU +76% |

Percentage = faster throughput / slower throughput - 1. Every number is a median of 3 rounds.

**The configuration matters.** Under default settings (`pipeline.object-reuse=false`) the GPU is faster on c1 (1.03M vs 0.83M, GPU +23%); with object-reuse enabled on both sides the result flips to CPU faster. Object-reuse helps the CPU more than the GPU (c1: roughly x2.95 vs x2.0). Tune the CPU side before choosing. Each case below gives both default and tuned results.

<a id="env"></a>
## 2. Environment and dataset

**Hardware and software**

- GPU: a single NVIDIA L20 (driver for CUDA 12.4, native code built for sm_89).
- Flink 2.2.1; Mega-Flink is the matching build from the same source drop. JVM heap 16 GB, G1, default heap state backend, watermark delay 20 ms, everything else default (including the default buffer-timeout).
- Mostly parallelism 1; input read from local disk. Host CPU model and core count: not verified, to be added before publishing.
- GPU operator settings: batch of 4,096 rows, 5 ms maximum wait; the c6 device hash table is capped at 4,194,304 rows; for c6, batches smaller than 1,024 rows fall back to a JVM lookup.

**Dataset**

- [KuaiRand](https://kuairand.com/) (a public recommendation dataset with randomly exposed videos). We use the KuaiRand-27K user behavior log: 92,919,789 rows, sorted by `time_ms`, converted to int64 column files and replayed in-process in time order. Please follow the dataset's public page for citation and license terms (re-check the link before publishing).
- The video table has 32,038,725 rows. For c6, only the videos that appear in the replayed slice are loaded as the dimension table (1,039,412 rows for mid/saturated tiers, 413,436 rows for the low tier; two BIGINT columns, video_id and author_id). The CPU HashMap and the GPU device hash table use the same data.

<a id="method"></a>
## 3. Methodology

**Three load tiers**

| Tier | Input rate | Records |
| --- | --- | --- |
| Low | 20,000 rec/s | 600 thousand |
| Mid | 100,000 rec/s | 3 million |
| Saturated | unthrottled | 3 million |

Each tier is warmed up with 300 thousand records at the same rate before timing. Low and mid tiers are rate-limited, so both sides reach the target throughput and the tiers compare latency; the saturated tier compares throughput.

**Rounds and statistics**: each case x tier x mode runs 3 rounds with the mode order rotated; throughput is the median of 3; p50 / p95 / p99 are the per-round percentiles, then the median across rounds (not the values from the median-throughput round).

**Latency definition**: from the record's scheduled send time to its arrival at the sink. Rate-limited tiers use the scheduled time to avoid coordinated omission. In the saturated tier the source is back-pressured and latency reflects a backlog, so it is only useful for relative comparison. "<0.01" or "0.00" means below 0.01 ms (for c1 and c6, source to sink sits in one operator chain). A few c2 window outputs do not trigger a reference record (24,354 rows at mid/saturated, 19,205 at low) and are excluded from latency.

**Correctness check**: for each case, the SHA-256 of the final table of the CPU run and each GPU mode is compared per case x tier. All published results are MATCH (zero mismatches). The c5 join pair count was also checked against an independent Java program (1,284,869 pairs in the formal run). Note that the comparison is on the **final table**; the intermediate changelog row counts differ between CPU and GPU (GPU in-batch merging emits fewer or more intermediate updates), for example c3 saturated: CPU 965,196 vs GPU batch 876,508 rows; c4: 3,361,901 vs 3,302,645; c5-join-agg (default config): CPU 765,548 vs fused 825,672. Latency sample counts differ slightly as a result.

**Exclusive runs**: each job owns the whole node (one job at a time) with 1 GPU. A separate "concurrent screening" pass (4 jobs on one node at once) was used for screening only and is not used in the tables below. Same-model nodes differ little (mostly within +/-5%); concurrent-screening throughput was mostly 3-6% higher than exclusive, with an outlier for c3 GPU batch, so only exclusive results are shown.

**CPU baseline**: Flink's own best configuration, with identical parallelism and mini-batch settings on both sides. For c2 we measured both the default two-phase (Local+Global) and one-phase (`1p`) plans; for c4 we also measured CPU with mini-batch (`cpu-mb`); each side's best is the reference. "Tuned" means `pipeline.object-reuse=true` on both sides (plus `table.exec.rank.topn-cache-size=100000` for c3, because the saturated tier has about 200 thousand distinct users). No further per-case tuning was done for either side.

**Breakdown data has two kinds, with different standing**:

- JFR sampling (CPU-side share per stage) is a **sampled estimate**: grouped by thread stack, with sampling overhead; use it for the rough distribution only.
- The CUDA-event timers built into Mega-Flink operators (row-to-column, H2D, kernel, D2H, column-to-row) are **measured**. H2D/D2H include queuing and synchronization and read 1.5-2.5x higher than the copy time nsys reports.
- nsys was used as an independent check on a few cases: total kernel time agrees with the Mega-Flink kernel timer to within about 10%.
- Breakdown jobs ran 1 round per case with profiling on, so their throughput is slightly lower and is not used to compare speed.

<a id="cases"></a>
## 4. Cases and results

In the tables below, "default" is Flink's default configuration and "tuned" has object-reuse on both sides (c3 adds topn-cache). Throughput is in records/s, latency in ms. The "check" column refers to the whole case x tier.

<a id="c1-filter"></a>
### c1 Filter / cleaning

**What / scenario**: sample cleaning and basic features. Drop invalid or randomly exposed rows, keep rows with play time over 3 s or a click, and compute watch ratio, duration bucket and interaction count. The plan is `Source -> Calc -> Sink`; CPU and GPU logical and physical plans are identical, and the GPU mode swaps Calc for a GPU Calc.

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

**Default**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Low | 20,000 | <0.01 / <0.01 | 19,999 | 2.74 / 6.16 | MATCH |
| Mid | 100,000 | <0.01 / 0.21 | 99,989 | 2.96 / 7.60 | MATCH |
| Saturated | 831,997 | <0.01 / <0.01 | 1,025,360 | 1.86 / 3.38 | MATCH |

**Tuned (object-reuse on both sides)**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Low | not measured | not measured | not measured | not measured | not measured |
| Mid | 100,000 | <0.01 / 0.01 | 99,978 | 2.80 / 6.23 | MATCH |
| Saturated | 2,472,421 | <0.01 / <0.01 | 2,059,561 | 0.92 / 1.77 | MATCH |

**Takeaway**: GPU +23% under default settings, CPU +20% after tuning. GPU-side measured timers (default, saturated, 927 batches): row-to-column 254 ms, H2D 20 ms, kernel 25 ms, D2H 14 ms, column-to-row 186 ms. These five sum to 499 ms; the kernel is about 5% and row/column conversion about 88%. nsys cross-check: kernel 13.2 ms, H2D 15.9 ms, D2H 8.1 ms.

<a id="c2-window"></a>
### c2 Per-user clicks over the last N minutes: window aggregation (HOP and TUMBLE)

**What / scenario**: real-time statistical features such as "a user's impressions / clicks / play time over the last N minutes". Event-time windows, aggregated per user_id. HOP uses a 30-minute window with a 5-minute slide (each record lands in several windows); TUMBLE uses a 5-minute tumbling window. On CPU, Flink splits the window aggregation into Local + Global phases by default; we also measured one-phase (`1p`, a direct WindowAggregate). The GPU two-phase plan uses the experimental `LocalWindowAggregate` path; the one-phase plan corresponds to Mega-Flink's more mature path.

```sql
-- HOP
SELECT user_id, window_start, window_end, COUNT(*) AS expo_cnt,
       SUM(is_click) AS click_cnt, SUM(play_time_ms) AS play_ms
FROM TABLE(HOP(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE, INTERVAL '30' MINUTE))
GROUP BY user_id, window_start, window_end;

-- TUMBLE: replace the HOP(...) call with
TABLE(TUMBLE(TABLE events, DESCRIPTOR(ts), INTERVAL '5' MINUTE))
```

**HOP, default** ("1p" = one-phase)

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU two-phase | 19,951 | 64.08 / 123.64 | MATCH |
| Low | CPU 1p | 19,951 | 30.69 / 67.84 | MATCH |
| Low | GPU two-phase | 19,971 | 54.76 / 105.96 | MATCH |
| Low | GPU 1p | 19,973 | 26.36 / 52.98 | MATCH |
| Mid | CPU two-phase | 99,752 | 68.30 / 132.81 | MATCH |
| Mid | CPU 1p | 99,747 | 25.19 / 64.30 | MATCH |
| Mid | GPU two-phase | 99,821 | 38.24 / 69.66 | MATCH |
| Mid | GPU 1p | 99,870 | 17.28 / 40.35 | MATCH |
| Saturated | CPU two-phase | 744,721 | 53.80 / 117.34 | MATCH |
| Saturated | CPU 1p | 628,892 | 26.11 / 78.29 | MATCH |
| Saturated | GPU two-phase | 898,493 | 37.33 / 83.32 | MATCH |
| Saturated | GPU 1p | 892,390 | 16.62 / 57.67 | MATCH |

**HOP, tuned**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU two-phase | 99,824 | 62.89 / 124.43 | MATCH |
| Mid | CPU 1p | 99,828 | 22.46 / 65.07 | MATCH |
| Mid | GPU two-phase | 99,924 | 37.16 / 64.68 | MATCH |
| Mid | GPU 1p | 99,920 | 16.06 / 34.12 | MATCH |
| Saturated | CPU two-phase | 1,475,619 | 48.34 / 97.40 | MATCH |
| Saturated | CPU 1p | 1,079,916 | 25.01 / 86.26 | MATCH |
| Saturated | GPU two-phase | 1,861,833 | 36.44 / 99.76 | MATCH |
| Saturated | GPU 1p | 1,184,784 | 16.02 / 51.10 | MATCH |

Tuned low tier: not measured.

**TUMBLE, default**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Low | 19,992 | 73.37 / 119.64 | 19,993 | 52.67 / 105.76 | MATCH |
| Mid | 99,961 | 59.86 / 122.12 | 99,967 | 34.96 / 77.36 | MATCH |
| Saturated | 724,616 | 43.42 / 105.17 | 965,671 | 32.37 / 94.38 | MATCH |

**TUMBLE, tuned**

| Tier | CPU throughput | CPU p50 / p99 | GPU throughput | GPU p50 / p99 | Check |
| --- | --- | --- | --- | --- | --- |
| Mid | 99,972 | 61.94 / 121.82 | 99,976 | 34.66 / 63.30 | MATCH |
| Saturated | 1,613,516 | 37.31 / 107.71 | 2,020,259 | 33.91 / 86.85 | MATCH |

**Takeaway**: the GPU is better on both throughput and latency. After tuning, HOP two-phase GPU +26% and TUMBLE +25%; one-phase vs one-phase is GPU +10% (the CPU one-phase plan is itself slower than its two-phase plan, so best-vs-best is +26%). The "GPU +42% (vs CPU one-phase)" figure under default settings compares against a non-optimal CPU plan and should not be quoted. CPU one-phase also varies between rounds under default settings (0.59-0.69M). On the GPU side, state read and merge segments are still present in the breakdown; "GPU batching removes state access" has no measured support.

<a id="c3-topn"></a>
### c3 Per-user top 10 by watch time (TopN)

**What / scenario**: Top-K features over user behavior, such as "the 10 longest-watched videos". An append-only `ROW_NUMBER` TopN. The plan is `Calc -> Exchange(hash user_id) -> Rank(AppendFast)`; the GPU replaces Rank with a GPU TopN in three forms: batch, persistent and adaptive.

```sql
SELECT seq, user_id, video_id, play_time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY play_time_ms DESC, seq ASC) AS rn
  FROM events) WHERE rn <= 10
```

**Default**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU | 19,995 | 14.70 / 33.03 | MATCH |
| Low | GPU batch | 19,995 | 16.81 / 33.23 | MATCH |
| Low | GPU persistent | 19,995 | 14.73 / 33.14 | MATCH |
| Low | GPU adaptive | 19,995 | 14.99 / 33.04 | MATCH |
| Mid | CPU | 99,977 | 3.58 / 6.69 | MATCH |
| Mid | GPU batch | 99,976 | 8.59 / 16.11 | MATCH |
| Mid | GPU persistent | 99,976 | 4.14 / 8.92 | MATCH |
| Mid | GPU adaptive | 99,979 | 4.15 / 7.40 | MATCH |
| Saturated | CPU | 1,024,400 | 4.69 / 20.25 | MATCH |
| Saturated | GPU batch | 930,760 | 5.61 / 25.01 | MATCH |
| Saturated | GPU persistent | 489,815 | 11.20 / 33.67 | MATCH |
| Saturated | GPU adaptive | 690,440 | 10.28 / 32.89 | MATCH |

GPU batch varies a lot between rounds at saturation: an 8-round rerun gave 0.90-1.16M with two clusters (5 rounds near 0.93M, 3 rounds at 1.13-1.16M); CPU gave 1.02-1.12M over 8 rounds (median 1.06M). It is therefore not correct to say the GPU is consistently ahead or behind.

**Tuned (object-reuse on both sides, plus topn-cache-size=100000)**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU | 99,978 | 3.53 / 6.70 | MATCH |
| Mid | GPU batch | 99,972 | 8.64 / 15.12 | MATCH |
| Mid | GPU persistent | 99,977 | 4.16 / 13.48 | MATCH |
| Mid | GPU adaptive | 99,978 | 4.05 / 10.77 | MATCH |
| Saturated | CPU | 1,363,629 | 3.59 / 22.79 | MATCH |
| Saturated | GPU batch | 1,231,075 | 5.22 / 29.97 | MATCH |
| Saturated | GPU persistent | 430,154 | 12.87 / 38.05 | MATCH |
| Saturated | GPU adaptive | 718,364 | 9.48 / 31.29 | MATCH |

Tuned low tier: not measured.

**Takeaway**: the CPU is not behind: CPU +11% over GPU batch after tuning, and +38% / +21% at parallelism 2/4 (section 5). Persistent has lower latency than batch only at low/mid rates and is not better than CPU, with low saturated throughput. Adaptive sends 0 rows to the GPU at low/mid rates (its latency equals CPU and is not a GPU result); at saturation about 36% of rows take the CPU path.

**Why the GPU does not win** (default config, exclusive, nsys and parameter sweeps): 910 `select_top_n_by_group` kernels total 108 ms, plus 5 ms H2D and 2 ms D2H, about 3.6% of the roughly 3.2 s job, so the GPU is more than 95% idle. Sweeping batch size (1,024 to 65,536) and max wait (1/5/20 ms) leaves saturated throughput within 0.87-1.11M with no clear trend. A ceiling probe with only source + exchange + sink and no computation reaches 0.856M on CPU and 0.851M on GPU, which indicates the JVM pipeline is the bottleneck.

<a id="c4-dedup"></a>
### c4 Keep-last dedup

**What / scenario**: sample dedup and latest-state extraction. Keep the newest row per (user_id, video_id). The plan is `Calc -> Exchange -> Rank(AppendFast, keep-last) -> Calc`; the GPU replaces Rank with a GPU Deduplicate (batch / adaptive). CPU with mini-batch (`cpu-mb`) was also measured.

```sql
SELECT seq, user_id, video_id, play_time_ms, time_ms FROM (
  SELECT seq, user_id, video_id, play_time_ms, time_ms,
         ROW_NUMBER() OVER (PARTITION BY user_id, video_id ORDER BY ts DESC) AS rn
  FROM events) WHERE rn = 1
```

**Default**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU | 19,995 | 11.49 / 24.98 | MATCH |
| Low | CPU mini-batch | 19,995 | 22.50 / 43.65 | MATCH |
| Low | GPU batch | 19,995 | 14.54 / 25.23 | MATCH |
| Low | GPU adaptive | 19,995 | 11.52 / 24.95 | MATCH |
| Mid | CPU | 99,962 | 2.88 / 5.06 | MATCH |
| Mid | CPU mini-batch | 99,950 | 14.93 / 30.94 | MATCH |
| Mid | GPU batch | 99,967 | 5.99 / 16.08 | MATCH |
| Mid | GPU adaptive | 99,967 | 2.86 / 5.06 | MATCH |
| Saturated | CPU | 685,882 | 5.23 / 19.63 | MATCH |
| Saturated | CPU mini-batch | 580,305 | 7.74 / 26.90 | MATCH |
| Saturated | GPU batch | 660,054 | 7.22 / 27.85 | MATCH |
| Saturated | GPU adaptive | 675,786 | 6.98 / 28.34 | MATCH |

**Tuned**

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU | 99,970 | 2.74 / 5.06 | MATCH |
| Mid | GPU batch | 99,964 | 5.93 / 13.95 | MATCH |
| Mid | GPU adaptive | 99,970 | 2.86 / 7.19 | MATCH |
| Saturated | CPU | 1,033,538 | 3.30 / 17.82 | MATCH |
| Saturated | GPU batch | 850,292 | 5.96 / 21.96 | MATCH |
| Saturated | GPU adaptive | 852,773 | 6.03 / 22.28 | MATCH |

Tuned low tier: not measured.

**Takeaway**: CPU is slightly ahead under default settings (0.686M vs 0.660M) and +22% after tuning. The main costs of dedup are serialization and JVM state access, which the GPU does not replace (JFR estimate: state about 14% on CPU and 17% in the GPU mode; GPU measured kernel 13 ms, H2D 8 ms, D2H 8 ms over 983 batches). Adaptive sends 0 rows to the GPU at low/mid rates; at saturation about 4.5% of rows take the CPU path.

<a id="c5-join"></a>
### c5 Impression x click Interval Join, and fusion with a following Filter / aggregation

**What / scenario**: impression-click attribution and training-sample assembly. The impression stream `imp` and click stream `clk` (`is_click=1`) come from the same log; the join key is `user_id*10^8+video_id` (a single BIGINT) and the click must fall within 30 minutes after the impression. Three variants:

- `c5-join`: the join only.
- `c5-join-filter`: adds `c.play_time_ms >= i.duration_ms` after the join (Flink pushes it down as a residual join condition; in GPU fused mode the residual is evaluated on the device, so "fused vs not fused" = fusion + residual push-down).
- `c5-join-agg`: adds a per-user_id aggregation (mini-batch 5 ms / 4096 on both sides).

In GPU fused mode, the Calc / LocalAggregate after the join runs in the same GPU stage as the join, saving one column-to-row and one row-to-column conversion. For `c5-join`, GPU batch is fused (Join -> projection) by default.

```sql
CREATE TEMPORARY VIEW imp AS
  SELECT seq, user_id * 100000000 + video_id AS uv, user_id, duration_ms, ts FROM events;
CREATE TEMPORARY VIEW clk AS
  SELECT seq, user_id * 100000000 + video_id AS uv, play_time_ms, ts FROM events WHERE is_click = 1;

-- c5-join
SELECT i.seq, c.seq, i.uv, c.play_time_ms, i.duration_ms
FROM imp i JOIN clk c ON i.uv = c.uv AND c.ts BETWEEN i.ts AND i.ts + INTERVAL '30' MINUTE;

-- c5-join-filter: add to the above
WHERE c.play_time_ms >= i.duration_ms      -- selecting i.seq, c.seq, i.uv, c.play_time_ms

-- c5-join-agg: after the join condition
SELECT i.user_id, COUNT(*) AS clicks, SUM(c.play_time_ms) AS play_ms
FROM imp i JOIN clk c ON ... GROUP BY i.user_id;
```

**Default** (explicit fuse / nofuse in the same job, median of 3; throughput / p50 / p99)

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join | Low | 19,994 / 29.91 / 79.26 | 19,993 / 29.88 / 80.09 | 19,993 / 29.98 / 79.91 | MATCH |
| join | Mid | 99,928 / 10.57 / 24.84 | 99,921 / 14.08 / 27.56 | 99,933 / 14.23 / 26.58 | MATCH |
| join | Saturated | 259,430 / 20.62 / 55.11 | 232,091 / 24.97 / 56.24 | 221,821 / 26.10 / 56.35 | MATCH |
| join-filter | Low | 19,993 / 29.64 / 79.08 | 19,990 / 30.00 / 80.25 | 19,989 / 29.97 / 80.03 | MATCH |
| join-filter | Mid | 99,928 / 10.36 / 26.64 | 99,953 / 13.95 / 26.62 | 99,904 / 13.95 / 25.87 | MATCH |
| join-filter | Saturated | 272,760 / 20.17 / 56.22 | 254,153 / 22.89 / 54.92 | 250,751 / 23.30 / 54.38 | MATCH |
| join-agg | Low | 19,861 / 107.58 / 195.47 | 19,958 / 92.68 / 180.70 | 19,859 / 96.89 / 183.59 | MATCH |
| join-agg | Mid | 99,252 / 55.91 / 102.08 | 99,774 / 37.97 / 76.68 | 99,397 / 43.59 / 86.78 | MATCH |
| join-agg | Saturated | 268,556 / 60.56 / 121.23 | 233,707 / 44.49 / 83.84 | 232,152 / 50.97 / 98.09 | MATCH |

For join-filter at saturation, "not fused + device-side residual" gave 251,131, essentially the same as "not fused" (about 251k), so residual push-down alone gives almost no benefit. GPU persistent (fused by default): join 166,494 saturated (p99 199.94 ms), join-filter 171,651 saturated (p99 197.54 ms); tail latency is about 200 ms across tiers, clearly worse than both CPU and batch.

**Tuned** (object-reuse on both sides; low tier: not measured)

| Case | Tier | CPU | GPU fused | GPU not fused | Check |
| --- | --- | --- | --- | --- | --- |
| join | Mid | 99,928 / 10.38 / 37.56 | 99,931 / 13.89 / 26.79 | 99,938 / 13.84 / 25.97 | MATCH |
| join | Saturated | 291,797 / 19.63 / 60.01 | 253,233 / 24.07 / 54.08 | 251,270 / 24.24 / 52.71 | MATCH |
| join-filter | Mid | 99,927 / 10.09 / 30.54 | 99,903 / 13.82 / 25.54 | 99,921 / 13.71 / 25.68 | MATCH |
| join-filter | Saturated | 307,921 / 18.80 / 63.22 | 262,297 / 23.49 / 51.55 | 259,502 / 23.67 / 52.45 | MATCH |
| join-agg | Mid | 99,141 / 55.20 / 102.33 | 99,798 / 37.57 / 76.84 | 99,377 / 42.76 / 86.52 | MATCH |
| join-agg | Saturated | 293,385 / 58.96 / 127.08 | 236,056 / 45.31 / 84.37 | 253,569 / 48.46 / 98.14 | MATCH |

For join-agg at saturation, the three CPU rounds were 293k, 257k and 300k, a wide spread.

**Takeaway**: CPU is faster on throughput: after tuning join +15%, join-filter +17%, join-agg +24% (against GPU fused). With a following aggregation, GPU latency is lower (p50 about 23% lower). Fusion: after tuning there is no throughput gain (join +0.8%, join-filter +1.1%, both within noise; join-agg fused is about 7% lower), only a lower join-agg p50 (about 6.5% saturated, 12% mid); join and join-filter latency is unchanged. The largest cost on both sides is JVM state access (the two join buffers; JFR estimate: about 52% on CPU join and 43% on GPU join), which the GPU does not replace. The join fan-out on this data is about 1, so there is little data movement for fusion to save.

<a id="c6-lookup"></a>
### c6 Feature dimension-table Lookup Join

**What / scenario**: real-time item-side feature joins: each event looks up a video dimension table by video_id to fetch author_id. Only INNER was measured. The CPU uses a fully in-memory HashMap (a favorable case, equivalent to Flink's full cache), and the GPU loads the table into a device hash table. Note that when a batch has fewer than 1,024 rows the GPU path falls back to a JVM lookup, so the `gpu` mode at low/mid rates mostly does not use the GPU; `gpu-min1` (force the GPU on every batch) is included as a control.

```sql
SELECT e.seq, e.video_id, d.author_id
FROM events AS e
JOIN video_dim FOR SYSTEM_TIME AS OF e.proc_time AS d
  ON e.video_id = d.video_id
```

The dimension table has 1,039,412 rows (413,436 at the low tier), not the full 32 million.

**Did the GPU actually run (CUDA kernel instances counted with nsys, default config, one run each)**

| Tier | `gpu` kernel instances | `gpu-min1` kernel instances |
| --- | --- | --- |
| Low | 3 (almost no GPU use) | 8,484 |
| Mid | 22 (almost no GPU use) | 6,151 |
| Saturated | 878 | 848 |

So the roughly 3 ms latency of `gpu` at low/mid rates is not caused by the GPU; only the saturated-tier GPU result has runtime evidence. The formal timed runs of c6 have no runtime GPU metrics (Lookup Join exposes none); the evidence is the planning-time `shim_status` (GPU-replaced on all three tiers) plus nsys.

**Default** (three modes rotated within one job)

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Low | CPU | 20,000 | <0.01 / <0.01 | MATCH |
| Low | GPU (actually JVM lookup) | 19,998 | 2.70 / 6.10 | MATCH |
| Low | GPU min1 | 19,998 | 2.71 / 6.14 | MATCH |
| Mid | CPU | 100,000 | <0.01 / 14.76 | MATCH |
| Mid | GPU (actually JVM lookup) | 99,990 | 3.18 / 9.48 | MATCH |
| Mid | GPU min1 | 99,991 | 2.96 / 8.97 | MATCH |
| Saturated | CPU | 601,930 | <0.01 / <0.01 | MATCH |
| Saturated | GPU | 838,326 | 2.33 / 4.70 | MATCH |
| Saturated | GPU min1 | 834,406 | 2.33 / 4.76 | MATCH |

**Tuned** (low tier: not measured)

| Tier | Mode | Throughput | p50 / p99 | Check |
| --- | --- | --- | --- | --- |
| Mid | CPU | 100,000 | <0.01 / 19.20 | MATCH |
| Mid | GPU | 99,990 | 2.93 / 6.28 | MATCH |
| Mid | GPU min1 | 99,994 | 2.78 / 6.20 | MATCH |
| Saturated | CPU | 1,299,187 | <0.01 / <0.01 | MATCH |
| Saturated | GPU | 2,284,322 | 0.88 / 1.75 | MATCH |
| Saturated | GPU min1 | 2,237,176 | 0.89 / 1.76 | MATCH |

**Takeaway**: at saturation the GPU is faster: +39% by default (838k vs 602k) and +76% after tuning. At low/mid rates throughput is capped by the input rate and is equal; the GPU adds about 3 ms of latency from batching. Only the simplest shape was verified: a single BIGINT key with a few BIGINT columns.

**LEFT lookup**: cannot run on the GPU in this build and falls back to CPU, so there is no performance comparison. Planning-time status: with a primary key on the dimension table, the output dimension columns do not match the dimension schema (LEFT makes dimension columns nullable while the primary-key column is non-null, so the types differ); without the primary key, the lookup key is reported as not proven to contain the dimension primary key.

**The c6 JFR share is not used**: the JFR classifier counts every class under the Mega-Flink package as "GPU operator JVM side", and the CPU lookup's dimension-table class happens to live in that package, so the CPU job also shows about 24% "GPU-related". This is confirmed to be a misclassification, so no conclusion is drawn from c6 JFR shares; the "GPU-related" share for other cases that use the same package (including the driver class) should also be re-checked.

<a id="parallelism"></a>
## 5. Higher parallelism and a larger dimension table

Method: the same replay, with one lightweight Long sequence source distributing by `k mod p` to p replay maps (each subtask replays in time order); downstream operators run at parallelism p on both CPU and GPU, and the p subtasks share one L20. Only c2, c3 and c6 were run. Under default settings, 18 groups (case x tier x parallelism) were compared and all MATCH; the check status for the tuned parallelism 2/4 runs: not verified.

**Tuned, saturated throughput (records/s)** (parallelism 2/4 figures are taken from the internal report; this draft did not re-check them against the raw results one by one)

| Case | Parallelism 2 (CPU / GPU) | Parallelism 4 (CPU / GPU) | Faster |
| --- | --- | --- | --- |
| c2 HOP two-phase | 2,015k / 2,532k | 2,907k / 3,365k | GPU +26% / +16% |
| c2 HOP one-phase | 1,846k / 2,017k | 3,028k / 3,340k | GPU +9% / +10% |
| c3 TopN (GPU batch) | 2,517k / 1,824k | 3,827k / 3,154k | CPU +38% / +21% |
| c6 Lookup | 1,592k / 2,991k | 3,168k / 3,809k | GPU +88% / +20% |

c3 GPU adaptive: 980k at parallelism 2 and 1,545k at parallelism 4.

**c6 larger dimension table** (tuned, parallelism 1, table padded to 4 million rows; GPU lookup cap 4,194,304 rows): saturated CPU 1,292k, GPU 2,360k (+83%), gpu-min1 2,370k; at mid rate both sides keep up, GPU p50 3.1 ms and CPU <0.01 ms.

**Latency caveat**: with parallelism above 1 there is an extra network exchange between replay and operators (Flink's default buffer-timeout is 100 ms, so at low rates a buffer waits for the timeout before being sent). Low/mid p50 values are therefore much higher than at parallelism 1 (for example c3 low tier 83-126 ms vs 14.7 ms at parallelism 1). This is introduced by the test driver and is common to CPU and GPU; use low/mid latency at parallelism above 1 only for CPU-vs-GPU comparison.

<a id="breakdown"></a>
## 6. Where the time goes (breakdown and nsys)

- At parallelism 1, most of the time goes to Flink's own serialization / network exchange, source replay, state access and DataStream-Table row conversion; GPU kernels are a small share. For example, c3 GPU batch at saturation: the kernel takes about 0.12 s while the whole job (3.3M rows) takes about 3.2 s (formal median).
- The new cost on the GPU side is row-to-column conversion, H2D / D2H, column-to-row conversion and JVM-side batch packing/unpacking. State access, timers, changelog merging and Global aggregation stay on the JVM and are not replaced by the GPU.
- See section 3 for how sampled estimates differ from measurements. JFR samples only task threads whose name contains "(1/1)"; if the persistent form has a separate resident thread it is not counted, so its GPU share is indicative only.
- CPU and GPU logical and physical plans (Calc / Rank / WindowAggregate / IntervalJoin / LookupJoin structure) are identical and EXPLAIN shows no difference; the replacement happens when ExecNodes are translated into Transformations.

<a id="limits"></a>
## 7. Known limitations

1. **Single GPU, mostly parallelism 1.** Parallelism 2 and 4 were run only for c2, c3 and c6; c4 and c5 at parallelism 2/4 were not measured; multi-GPU and higher parallelism were not measured.
2. **c6 with the full dimension table was not measured.** The full 32.04 million rows exceed the GPU lookup cap (4,194,304 rows); only the 1,039,412-row slice and a table padded to 4 million rows were tested.
3. **The rewritten report has not been independently re-reviewed.** The version rewritten after the supplementary runs (object-reuse tuning, parallelism, TUMBLE, nsys) has not been through independent review.
4. **LEFT lookup cannot run on the GPU in this build**, so there is no LEFT performance comparison.
5. **The c6 JFR share is not used because of a misclassification.** The "GPU-related" share needs re-checking for any case that uses the same package.
6. **JFR is a sampled estimate,** not an exact timing; the Mega-Flink timers and nsys are the measurements, and H2D / D2H include queuing.
7. **Tuned results do not cover every tier.** The tuned low tier (20k/s) was not measured; c4, c5 and c6 tuning covers only mid and saturated. Tuning is limited to `pipeline.object-reuse` (and the c3 topn-cache); no further per-case tuning was done, and other CPU-side options were not explored (apart from c4 `cpu-mb` and the c5-join-agg mini-batch).
8. **Default configuration is not the best configuration.** Conclusions under defaults (for example the GPU winning c1) change after tuning; state the basis when quoting.
9. **The check compares final tables,** and intermediate changelogs differ between CPU and GPU (see section 3).
10. **GPU persistent and adaptive:** no support was found for "choose persistent"; adaptive sends 0 rows to the GPU at low/mid rates. For c5 persistent, command counts were recorded only in the breakdown round and were 0 in the formal rounds; the reason is not verified.
11. **Not done:** other GPU models (for example H20), multi-GPU, larger dimension tables, other datasets. Only one dataset and one GPU model were used.
12. **c3 GPU batch shows two clusters of results at saturation** (see the c3 section); there is no finer evidence for the mechanism and none is claimed.

<a id="repro"></a>
## 8. Reproduction

**The scripts and the test driver have not yet been added to this repository, and their paths have not yet been parameterized** (the existing scripts contain hard-coded local data and output paths that must be turned into parameters before publishing). The list below therefore says what is needed; it is not yet a one-command reproduction.

What you need:

- One NVIDIA L20 (or another sm_89 GPU) with a driver supporting CUDA 12.4, on an exclusive node, with enough host memory for a 16 GB JVM heap plus the data files.
- Flink 2.2.1 and the matching Mega-Flink build (GPU operators and shim). How to obtain and build it: not verified, to be added.
- The KuaiRand-27K dataset (from its public page), and a converter that turns the log into int64 column files sorted by `time_ms` (exists internally, not yet in the repository).
- A test driver that does the in-process replay, defines the SQL, collects the sink changelog, computes latency and does the SHA-256 comparison (exists internally, not yet in the repository).
- Run scripts that rotate "case x mode x tier x round", summarize medians and run the comparison (exist internally, not yet in the repository).
- Optional: JFR and nsys for breakdown and kernel cross-checks.

Mode names (internal naming, given only to map to the result tables): `cpu`, `cpu-mb`, `cpu-1p`, `gpu`, `gpu-1p`, `gpu-batch`, `gpu-persistent`, `gpu-adaptive`, `*-fuse` / `*-nofuse`, `gpu-min1`; tuned modes add a `-tuned` suffix.

Notes:

- Do not edit a shell script while it is being read by a running job; internally this once caused a few rounds to be mislabeled as failed (the data were valid and were counted).
- Latency at rate-limited tiers is computed from the scheduled send time; saturated-tier latency is for relative comparison only.
- Concurrent runs change throughput (in screening, concurrent numbers were mostly 3-6% higher, with outliers for some cases); compare on an exclusive node.
