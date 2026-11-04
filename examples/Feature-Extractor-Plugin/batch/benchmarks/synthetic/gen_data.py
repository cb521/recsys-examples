#!/usr/bin/env python3
"""Synthetic benchmark data (parquet).  Deterministic functions of the row id (no data files to ship).

  python3 bench/gen_data.py --out DIR --wide-rows 50000000 --h-rows 30000000      (run on a GPU node: gpu-run.sh -- python3 ...)
Tables
  wide      : feature-style wide table for expression-heavy queries (ints, longs, doubles, floats, date, timestamp, bool)
  lineitem/orders/customer/part/supplier/nation/region : TPC-H *shaped* tables (synthetic values, doubles instead of decimals,
              uniform distributions) for the 'ordinary ETL / join / shuffle' control queries.  NOT official TPC-H data.
"""
import sys, os, argparse, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tests", "fz"))
from common import *

def h(k, mod): return f"pmod(xxhash64(id, {k}), {mod})"
def dbl(k, lo, hi): return f"CAST({lo} + (CAST(pmod(xxhash64(id, {k}), 1000000007) AS DOUBLE) / 1000000007.0) * ({hi} - {lo}) AS DOUBLE)"

def wide_cols():
    c = ["id"]
    for i in range(1, 5): c.append(f"CAST({h(i, 2000000)} - 1000000 AS INT) AS i{i}")
    for i in range(1, 3): c.append(f"CAST({h(10 + i, 1000000007)} AS BIGINT) * 1000 + id AS l{i}")
    for i in range(1, 9): c.append(f"{dbl(20 + i, -1000.0, 1000.0)} AS d{i}")
    for i in range(1, 5): c.append(f"CAST({dbl(40 + i, -50.0, 50.0)} AS FLOAT) AS f{i}")
    c.append(f"date_add(DATE'2015-01-01', CAST({h(50, 3650)} AS INT)) AS dt")
    c.append(f"timestamp_micros(1420070400000000 + CAST({h(51, 315360000)} AS BIGINT) * 1000000) AS ts")
    c.append(f"{h(52, 100)} < 37 AS flag")
    return c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--wide-rows", type=int, default=50_000_000)
    ap.add_argument("--h-rows", type=int, default=30_000_000, help="lineitem rows; orders = /4, customer = /40, part=/30")
    ap.add_argument("--skip-wide", action="store_true"); ap.add_argument("--skip-h", action="store_true")
    a = ap.parse_args()
    sp = build_session(use_gpu=True, local_n=4, extra={"spark.sql.files.maxRecordsPerFile": "2000000"})
    sp.conf.set("spark.sql.shuffle.partitions", "16")
    sp.conf.set("spark.sql.parquet.compression.codec", "snappy")
    def w(df, name):
        t0 = time.time(); df.write.mode("overwrite").parquet(os.path.join(a.out, name))
        print(f"[{now()}] wrote {name} in {time.time()-t0:.1f}s", flush=True)
    R = lambda n, p=40: sp.range(n, numPartitions=p)
    if not a.skip_wide:
        w(R(a.wide_rows).selectExpr(*wide_cols()), "wide")
    if not a.skip_h:
        n = a.h_rows; no = max(n // 4, 1); nc = max(n // 40, 10); npt = max(n // 30, 10); ns = max(n // 600, 10)
        modes = "'AIR','AIR REG','FOB','MAIL','RAIL','SHIP','TRUCK'"
        # lineitem
        w(R(n).selectExpr(
            "id AS l_idx", f"CAST(id DIV 4 + 1 AS BIGINT) AS l_orderkey", f"CAST({h(1, npt)} + 1 AS BIGINT) AS l_partkey", f"CAST({h(2, ns)} + 1 AS BIGINT) AS l_suppkey",
            "CAST(id % 4 + 1 AS INT) AS l_linenumber", f"CAST(1 + {h(3, 50)} AS DOUBLE) AS l_quantity",
            f"CAST(1 + {h(3, 50)} AS DOUBLE) * (900.0 + {h(4, 100000)} / 100.0) AS l_extendedprice", f"{h(5, 11)} / 100.0 AS l_discount", f"{h(6, 9)} / 100.0 AS l_tax",
            f"elt(1 + {h(7, 3)}, 'A', 'N', 'R') AS l_returnflag", f"elt(1 + {h(8, 2)}, 'O', 'F') AS l_linestatus",
            f"date_add(DATE'1992-01-02', CAST({h(9, 2526)} AS INT)) AS l_shipdate", f"date_add(DATE'1992-01-02', CAST({h(9, 2526)} + {h(10, 60)} - 30 AS INT)) AS l_commitdate",
            f"date_add(DATE'1992-01-02', CAST({h(9, 2526)} + 1 + {h(11, 30)} AS INT)) AS l_receiptdate",
            f"elt(1 + {h(12, 4)}, 'DELIVER IN PERSON', 'COLLECT COD', 'NONE', 'TAKE BACK RETURN') AS l_shipinstruct", f"elt(1 + {h(13, 7)}, {modes}) AS l_shipmode").drop("l_idx"), "lineitem")
        w(R(no).selectExpr("CAST(id + 1 AS BIGINT) AS o_orderkey", f"CAST({h(1, nc)} + 1 AS BIGINT) AS o_custkey", f"elt(1 + {h(2, 3)}, 'O', 'F', 'P') AS o_orderstatus",
                           f"{dbl(3, 1000.0, 450000.0)} AS o_totalprice", f"date_add(DATE'1992-01-01', CAST({h(4, 2406)} AS INT)) AS o_orderdate",
                           f"elt(1 + {h(5, 5)}, '1-URGENT', '2-HIGH', '3-MEDIUM', '4-NOT SPECIFIED', '5-LOW') AS o_orderpriority", "0 AS o_shippriority"), "orders")
        w(R(nc, 8).selectExpr("CAST(id + 1 AS BIGINT) AS c_custkey", f"CAST({h(1, 25)} AS BIGINT) AS c_nationkey", f"elt(1 + {h(2, 5)}, 'AUTOMOBILE', 'BUILDING', 'FURNITURE', 'MACHINERY', 'HOUSEHOLD') AS c_mktsegment",
                              f"{dbl(3, -999.0, 9999.0)} AS c_acctbal", "concat('Customer#', lpad(CAST(id + 1 AS STRING), 9, '0')) AS c_name"), "customer")
        w(R(npt, 8).selectExpr("CAST(id + 1 AS BIGINT) AS p_partkey", f"elt(1 + {h(1, 6)}, 'PROMO BRUSHED', 'STANDARD POLISHED', 'ECONOMY ANODIZED', 'PROMO PLATED', 'LARGE BURNISHED', 'SMALL PLATED') AS p_type",
                               f"concat('Brand#', CAST(1 + {h(2, 5)} AS STRING), CAST(1 + {h(3, 5)} AS STRING)) AS p_brand", f"CAST(1 + {h(4, 50)} AS INT) AS p_size"), "part")
        w(R(ns, 4).selectExpr("CAST(id + 1 AS BIGINT) AS s_suppkey", f"CAST({h(1, 25)} AS BIGINT) AS s_nationkey"), "supplier")
        w(sp.range(25, numPartitions=1).selectExpr("CAST(id AS BIGINT) AS n_nationkey", "concat('NATION', CAST(id AS STRING)) AS n_name", "CAST(id % 5 AS BIGINT) AS n_regionkey"), "nation")
        w(sp.range(5, numPartitions=1).selectExpr("CAST(id AS BIGINT) AS r_regionkey", "elt(CAST(id AS INT) + 1, 'AFRICA', 'AMERICA', 'ASIA', 'EUROPE', 'MIDDLE EAST') AS r_name"), "region")

if __name__ == "__main__":
    main()
