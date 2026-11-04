"""Benchmark queries.  E* = expression-heavy (Project/Filter dominated), N* = ordinary ETL / join / shuffle controls (TPC-H shaped).
Every query is a SQL string over parquet views registered by run_bench.py (wide, lineitem, orders, ...)."""

def _e1():
    ex = []
    ex.append("i1 * i2 + i3 AS a1")
    ex.append("(i1 - i2) * 3 + i4 AS a2")
    ex.append("i3 % 97 + i4 % 13 AS a3")
    ex.append("l1 % 1000 + l2 % 77 AS a4")
    ex.append("CASE WHEN d1 > 0 THEN d1 * d2 WHEN d1 < -500 THEN d3 - d4 ELSE d5 / 2.0 END AS b1")
    ex.append("CASE WHEN d2 > d3 AND d4 > 0 THEN d5 + d6 WHEN d2 <= d3 AND d4 <= 0 THEN d5 - d6 ELSE d7 * d8 END AS b2")
    ex.append("CASE WHEN i1 > 100000 THEN 3 WHEN i1 > 0 THEN 2 WHEN i1 > -100000 THEN 1 ELSE 0 END AS b3")
    ex.append("CASE WHEN f1 > f2 THEN f1 - f2 ELSE f2 - f1 END AS b4")
    ex.append("(d1 * d2 + d3 * d4) / (abs(d5) + 1.0) AS c1")
    ex.append("(d1 + d2 + d3 + d4 + d5 + d6 + d7 + d8) / 8.0 AS c2")
    ex.append("d1 * 0.5 + d2 * 0.25 + d3 * 0.125 + d4 * 0.0625 AS c3")
    ex.append("(d5 - d6) * (d5 - d6) + (d7 - d8) * (d7 - d8) AS c4")
    ex.append("CAST(f1 AS DOUBLE) * d1 + CAST(f2 AS DOUBLE) * d2 AS c5")
    ex.append("d1 > d2 AS e1")
    ex.append("(d3 < 0 OR d4 < 0) AND NOT flag AS e2")
    ex.append("i1 > i2 AND i3 > i4 AS e3")
    ex.append("if(flag, d1, d2) + if(NOT flag, d3, d4) AS g1")
    ex.append("greatest(d1, d2, d3) - least(d4, d5, d6) AS g2")
    ex.append("coalesce(d1 / d2, d3 / d4, 0.0) AS g3")
    ex.append("CAST(d1 AS INT) + CAST(d2 AS INT) * 2 AS h1")
    ex.append("CAST(i1 AS DOUBLE) * d1 + CAST(l1 AS DOUBLE) AS h2")
    ex.append("(i1 + i2) * (i3 - i4) % 10007 AS h3")
    ex.append("abs(d1) + abs(d2) + abs(d3) AS h4")
    return "SELECT id, " + ", ".join(ex) + " FROM wide"

def _e2():
    return """SELECT id, d1 * d2 + d3 AS p1, i1 + i2 * 2 AS p2, CASE WHEN d4 > 0 THEN d4 ELSE -d4 END AS p3, (d5 + d6) / 2.0 AS p4,
       f1 * f2 AS p5, i3 % 11 AS p6, l1 % 100 AS p7, if(flag, d7, d8) AS p8, d1 - d2 AS p9, abs(d3) * 2.0 AS p10
FROM wide
WHERE d1 > -900 AND d1 < 900 AND i1 % 7 <> 0 AND d2 * d3 > -100000 AND (i2 > -900000 OR flag) AND f1 + f2 < 40 AND l1 % 5 < 4"""

def _e3():
    # feature engineering with exactly-reproducible functions only: normalise, clip, polynomial, bucketise, sqrt/floor
    ex = []
    for k in range(1, 9):
        d = f"d{k}"
        z = f"(({d} - 3.5) / 120.0)"
        clip = f"greatest(least({z}, 4.0), -4.0)"
        poly = f"({clip} * {clip} * {clip} * 0.1 + {clip} * {clip} * 0.3 - {clip} * 0.7 + 1.0)"
        ex.append(f"CASE WHEN {poly} > 2.0 THEN sqrt({poly}) WHEN {poly} > 0.5 THEN {poly} * 0.5 ELSE floor({poly} * 10.0) / 10.0 END AS feat{k}")
    ex.append("signum(d1 * d2) * sqrt(abs(d1 * d2)) AS sgnroot")
    ex.append("(d1 - d2) / (abs(d1) + abs(d2) + 1.0) AS ratio")
    return "SELECT id, " + ", ".join(ex) + " FROM wide"

def _e4():
    return """SELECT id, exp(d1 / 500.0) + log(abs(d2) + 1.0) AS t1, sin(d3 / 100.0) * cos(d4 / 100.0) AS t2, pow(abs(d5) / 100.0 + 1.0, 1.5) AS t3,
       tanh(d6 / 300.0) AS t4, atan(d7 / 50.0) + log10(abs(d8) + 1.0) AS t5, exp(-abs(d1) / 200.0) / (1.0 + exp(-d2 / 200.0)) AS t6 FROM wide"""

def _e5():
    return """SELECT id, year(dt) AS y, month(dt) AS m, dayofmonth(dt) AS dm, dayofweek(dt) AS dw, quarter(dt) AS q, datediff(dt, DATE'2020-01-01') AS dd,
       CASE WHEN dayofweek(dt) IN (1, 7) THEN 1 ELSE 0 END AS weekend, hour(ts) AS hh, minute(ts) AS mi, second(ts) AS ss,
       CAST(ts AS DATE) AS tsd, date_add(dt, i1 % 30) AS dplus, datediff(CAST(ts AS DATE), dt) AS dd2,
       CASE WHEN month(dt) IN (12, 1, 2) THEN 'W' WHEN month(dt) BETWEEN 3 AND 5 THEN 'S' ELSE 'O' END AS season_str FROM wide"""

def _e6():
    # very deep dependent chains (stress for kernel codegen and for intermediate-column traffic)
    outs = []
    for k in range(4):
        e = f"d{k+1}"
        for j in range(30):
            c = [("* 1.0001", ), ("+ 0.5", ), ("- 0.25", ), ("* 0.999", )][j % 4][0]
            e = f"({e} {c})"
        e = f"CASE WHEN {e} > 0 THEN {e} * 2.0 ELSE {e} - 1.0 END"
        outs.append(f"{e} AS deep{k}")
    return "SELECT id, " + ", ".join(outs) + " FROM wide"

H = {
"N1_q1_agg": """SELECT l_returnflag, l_linestatus, sum(l_quantity) sum_qty, sum(l_extendedprice) sum_base, sum(l_extendedprice*(1-l_discount)) sum_disc,
  sum(l_extendedprice*(1-l_discount)*(1+l_tax)) sum_charge, avg(l_quantity) avg_qty, avg(l_extendedprice) avg_price, avg(l_discount) avg_disc, count(*) cnt
FROM lineitem WHERE l_shipdate <= DATE'1998-09-02' GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus""",
"N2_q6_filter_sum": """SELECT sum(l_extendedprice * l_discount) AS revenue FROM lineitem
WHERE l_shipdate >= DATE'1994-01-01' AND l_shipdate < DATE'1995-01-01' AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24""",
"N3_q3_join": """SELECT l_orderkey, sum(l_extendedprice*(1-l_discount)) revenue, o_orderdate, o_shippriority
FROM customer, orders, lineitem WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey AND l_orderkey = o_orderkey
 AND o_orderdate < DATE'1995-03-15' AND l_shipdate > DATE'1995-03-15' GROUP BY l_orderkey, o_orderdate, o_shippriority ORDER BY revenue DESC, o_orderdate LIMIT 10""",
"N4_q5_multijoin": """SELECT n_name, sum(l_extendedprice * (1 - l_discount)) AS revenue FROM customer, orders, lineitem, supplier, nation, region
WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey AND s_nationkey = n_nationkey
 AND n_regionkey = r_regionkey AND r_name = 'ASIA' AND o_orderdate >= DATE'1994-01-01' AND o_orderdate < DATE'1995-01-01' GROUP BY n_name ORDER BY revenue DESC""",
"N5_q12_case_agg": """SELECT l_shipmode, sum(CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' THEN 1 ELSE 0 END) AS high_line_count,
  sum(CASE WHEN o_orderpriority <> '1-URGENT' AND o_orderpriority <> '2-HIGH' THEN 1 ELSE 0 END) AS low_line_count
FROM orders, lineitem WHERE o_orderkey = l_orderkey AND l_shipmode IN ('MAIL', 'SHIP') AND l_commitdate < l_receiptdate AND l_shipdate < l_commitdate
 AND l_receiptdate >= DATE'1994-01-01' AND l_receiptdate < DATE'1995-01-01' GROUP BY l_shipmode ORDER BY l_shipmode""",
"N6_q14_join_case": """SELECT 100.00 * sum(CASE WHEN p_type LIKE 'PROMO%' THEN l_extendedprice*(1-l_discount) ELSE 0 END) / sum(l_extendedprice*(1-l_discount)) AS promo_revenue
FROM lineitem, part WHERE l_partkey = p_partkey AND l_shipdate >= DATE'1995-09-01' AND l_shipdate < DATE'1995-10-01'""",
"N7_shuffle_sort": """SELECT l_suppkey, l_partkey, sum(l_quantity) q, count(*) c FROM lineitem GROUP BY l_suppkey, l_partkey ORDER BY q DESC, l_suppkey, l_partkey LIMIT 100""",
"N8_scan_count": """SELECT count(*), sum(l_orderkey) FROM lineitem""",
}

E = {"E1_wide_project": _e1(), "E2_filter_project": _e2(), "E3_feature_chain": _e3(), "E4_transcendental": _e4(), "E5_date_parts": _e5(), "E6_deep_chain": _e6()}
S = {"S0_scan_E1_columns": "SELECT id, i1, i2, i3, i4, l1, l2, d1, d2, d3, d4, d5, d6, d7, d8, f1, f2, flag FROM wide",
     "S1_scan_E6_columns": "SELECT id, d1, d2, d3, d4 FROM wide"}
ALL = {**S, **E, **H}
KIND = {k: ("E" if k in E else "S" if k in S else "N") for k in ALL}
# 'write' sink variants (parquet output) for 2 representative expression queries
WRITE_VARIANTS = ["E1_wide_project", "E2_filter_project"]
