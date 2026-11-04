# Cast 融合要点（数值/布尔/日期/时间戳）

来源：spark-rapids `GpuCast.scala`（doCast 在 :295，默认分支 cudf castTo 在 :641，hasSideEffects 在 :1666）；Spark 3.5 语义；详见 cast.json。

1. **整数收窄 = 补码回绕**（CPU 与 GPU 默认分支一致）。ANSI 下越界抛异常，GpuCast.hasSideEffects=true，第一版排除 ANSI。
2. **float/double -> int/long**：向零截断、饱和、NaN->0。GPU 对 (Float,Long)(Double,Int|Long) 先把 NaN 置 0（:544），其余依赖 CUDA 转换饱和。设备函数要显式写出饱和（注意 2147483647 的 float 值是 2^31，必须用 `>=` 判断）。
3. **float/double -> byte/short**：CPU 是先转 int 再回绕（300.0f -> 44），GPU 走 cudf static_cast，很可能是饱和，两者可能不同。放 v2，先用差分测试确认，登记到 KNOWN-DIFFS。融合基准以"不融合 GPU"为准。
4. **int/long -> float/double**、**double -> float** 必须是 round-to-nearest-even；用 `__ll2float_rn`、`__double2float_rn`，不要 fast-math，`--fmad=false`。
5. **任意数值 -> bool** 是 `x != 0`（NaN 为 true，-0.0 为 false）。
6. **timestamp -> long** 是 floorDiv(micros, 1e6)，负数向下取整（不是 C 的 `/`）。**timestamp -> int/short/byte** 先 floor 再回绕；Spark 4.0 / db143 改为溢出给 NULL（CastTimeToIntShim），随 Spark 版本分支。**timestamp -> double** 是 micros/1e6 的 double 除法。
7. **long -> timestamp**：乘 1e6，溢出时饱和到 Long.Min/Max（不是回绕），下界用 -(Long.Max/1e6) 判断（:1611）。byte/short/int -> timestamp 不会溢出。
8. **float/double -> timestamp**：NaN/Inf -> NULL，并且 Spark 3.3+ 的 shim 标 hasSideEffects=true，输出可空性变化。第一版不融合。
9. **timestamp <-> date**：只有 UTC 会话时区可用简单除法（floorDiv 86400e6）；非 UTC 需要时区表（GpuTimeZoneDB），v1 拒绝非 UTC。
10. **date -> 数值/bool**：GpuCast 直接返回全 NULL（:314），不融合。
11. 空值：Cast 是 NullIntolerant，输入 NULL 输出 NULL；但上述 float->timestamp、Spark 4 的 timestamp->窄整数等会"非空输入产生 NULL"，融合的输出有效位必须按 `Cast.forceNullable` 同步。
12. 字符串、decimal、嵌套、interval：不在 v1（变长/128 位/溢出转 NULL/配置开关 castFloatToDecimal 等差异）。
