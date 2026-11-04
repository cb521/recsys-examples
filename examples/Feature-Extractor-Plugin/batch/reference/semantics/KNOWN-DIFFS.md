# 已知差异清单：spark-rapids（不融合）GPU vs CPU Spark

这些差异在**融合关闭**时就存在，不是融合插件引入的。融合插件的硬性要求是“与不融合的 GPU 逐位一致”，所以融合内核要**照抄 GPU 的行为**（不是 CPU 的）。
环境：L20，spark-rapids 26.08.1 (cuDF 26.08.0)，Spark 3.5.9，本地模式，ANSI 关，session 时区 UTC。
证据来源：`tests/fz/directed.py`（518 个模板 x 全部特殊值笛卡尔积，结果 `tests/out/base/directed/directed.json`）、
`tests/fz/fuzz.py`（随机表达式树 200 例 x 100 万行，自动最小化，`tests/out/base/fuzz/`）。比较方式：整数/布尔/日期完全一致；浮点按位一致（任意 NaN 归一）。

| # | 表达式 | CPU | GPU(不融合) | 触发条件 | 融合应当 |
|---|---|---|---|---|---|
| 1 | float/double 一元负号 `-a` | 翻转符号位：`-(0.0) = -0.0` | `0 - a`：`-(0.0) = +0.0`，`-(-0.0) = +0.0` | a 为 ±0.0 | 照 GPU：`0 - a`。（-0.0 流向 1/x 在 Spark 里是 NULL，影响很小，但按位比较会看见） |
| 2 | float/double `greatest/least` | 平局保留先出现的参数 | 平局取**后**一个参数（cuDF NULL_MAX/NULL_MIN: `x>y ? x : y`） | 0.0 与 -0.0 比较相等 | 照 GPU：`acc=(acc>c)?acc:c`（least 用 `<`）；NaN 另按 greatest=NaN 赢 / least=NaN 输 |
| 3 | `date_add(date, int)`（及其下游如 `dayofmonth(...)`） | 日期天数按 Int 回绕 | 溢出后结果不同（i=Int.MinValue 时 `dayofmonth` 13 vs 25） | 天数参数为极值 | 排除或照 GPU（v2）；融合 v1 不建议收 date_add 的极值语义 |
| 4 | `CAST(date_add(dt,i) AS TIMESTAMP)` | CPU 抛溢出异常（TASK_WRITE_FAILED） | 不抛，返回值 | 日期 + 极值天数 | 融合不得改变“GPU 不抛”这一点；该组合不入白名单 |
| 5 | `dayofyear(date)`（可靠性问题，不是数值差异） | 正常 | 同一个查询**第二次**执行时卡死（Java 线程忙转，GPU 空闲），本环境可复现 | 与其它列同批 | 不入 v1 白名单；随机测试默认排除（`FZ_HANG_PRONE=1` 打开） |

未发现差异（directed 全部通过）：整数 + - * 回绕、/ 和 % 和 pmod 除零得 NULL（含 float 的 0.0/-0.0 除数）、Long.MinValue div -1、整数 % -1、NaN 比较（NaN=NaN 真、NaN 最大）、
`<=>`、AND/OR 三值逻辑、IF/CASE WHEN/COALESCE、IN、isnan/nanvl、位运算与移位（含计数掩码）、bit_count、各数值类型间 Cast（窄化回绕、float→int 饱和、NaN→0）、
sqrt/floor/ceil/signum/rint/degrees/radians、year/month/day/quarter/dayofweek/weekday/last_day/datediff、hour/minute/second、timestamp↔date/long/double 转换。

## 融合版 vs 不融合 GPU 发现的不一致（插件 bug，均已修复）
| 表达式 | 现象 | 状态 |
|---|---|---|
| byte/short `pmod(a,b)` | a=-127,b=-128：不融合 -127，融合 1（`(r+n)%n` 应在 int 上算，不能在 byte 上回绕） | **已修复**（最终 jar：directed 518 模板 + 随机 300 例 0 不一致） |
| float/double `greatest/least` 的 ±0.0 平局 | 融合取先到者（CPU 行为），不融合 GPU 取后者 | **已修复** |
