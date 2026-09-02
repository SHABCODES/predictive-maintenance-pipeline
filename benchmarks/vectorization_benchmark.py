From 71a049be74c28a80a95d4823e47337e24bbcc608 Mon Sep 17 00:00:00 2001
From: Shabda Mantripragada <shabdamantripragada@gmail.com>
Date: Wed, 2 Sep 2026 14:45:58 +0000
Subject: [PATCH] perf: vectorize failure label generation (44x speedup)

Replace row-wise df['RUL'].apply(lambda x: 1 if x <= t else 0) with a
vectorized boolean comparison (df['RUL'] <= threshold).astype(int).

Benchmarked on the full FD001 dataset (20,631 rows, 50-run average):
- apply(lambda):        ~5.0 ms
- vectorized comparison: ~0.11 ms
- Speedup: ~44x, identical output verified

See benchmarks/vectorization_benchmark.py for the reproducible benchmark.
All existing tests (20/20) pass unchanged.
---
 benchmarks/vectorization_benchmark.py | 58 +++++++++++++++++++++++++++
 src/transform.py                      |  7 ++--
 2 files changed, 62 insertions(+), 3 deletions(-)
 create mode 100644 benchmarks/vectorization_benchmark.py

diff --git a/benchmarks/vectorization_benchmark.py b/benchmarks/vectorization_benchmark.py
new file mode 100644
index 0000000..9dea6ae
--- /dev/null
+++ b/benchmarks/vectorization_benchmark.py
@@ -0,0 +1,58 @@
+"""
+Benchmark: row-wise .apply(lambda) vs. vectorized boolean comparison
+for generating the binary 'failure' label from RUL.
+
+Run from repo root:
+    python benchmarks/vectorization_benchmark.py
+
+Background:
+The original implementation used df['RUL'].apply(lambda x: 1 if x <= t else 0),
+which evaluates the lambda once per row in a Python-level loop under the hood.
+Pandas/NumPy support vectorized comparison directly on the underlying array,
+which avoids the per-row Python function call overhead entirely.
+"""
+import sys
+import time
+
+sys.path.insert(0, "src")
+sys.path.insert(0, ".")
+
+from extract import extract_data
+from config import CONFIG
+
+N_RUNS = 50
+
+
+def main():
+    df = extract_data("data/train_FD001.txt")
+
+    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
+    max_cycle.columns = ["engine_id", "max_cycle"]
+    df = df.merge(max_cycle, on="engine_id")
+    df["RUL"] = df["max_cycle"] - df["cycle"]
+
+    threshold = CONFIG["failure_threshold"]
+
+    # Old approach: row-wise apply
+    t0 = time.perf_counter()
+    for _ in range(N_RUNS):
+        result_apply = df["RUL"].apply(lambda x: 1 if x <= threshold else 0)
+    apply_time = (time.perf_counter() - t0) / N_RUNS
+
+    # New approach: vectorized boolean comparison
+    t0 = time.perf_counter()
+    for _ in range(N_RUNS):
+        result_vec = (df["RUL"] <= threshold).astype(int)
+    vec_time = (time.perf_counter() - t0) / N_RUNS
+
+    assert result_apply.equals(result_vec), "Vectorized result diverges from apply() result"
+
+    print(f"Dataset size: {len(df)} rows")
+    print(f".apply(lambda) time:   {apply_time * 1000:.4f} ms (avg over {N_RUNS} runs)")
+    print(f"vectorized comparison: {vec_time * 1000:.4f} ms (avg over {N_RUNS} runs)")
+    print(f"Speedup: {apply_time / vec_time:.1f}x")
+    print(f"Results identical: {result_apply.equals(result_vec)}")
+
+
+if __name__ == "__main__":
+    main()
diff --git a/src/transform.py b/src/transform.py
index 93937a7..eaa9d5e 100644
--- a/src/transform.py
+++ b/src/transform.py
@@ -25,9 +25,10 @@ def transform_data(df):
     df = df.dropna()
     
     # Failure label
-    df['failure'] = df['RUL'].apply(
-        lambda x: 1 if x <= CONFIG["failure_threshold"] else 0
-    )
+    # Vectorized boolean comparison instead of row-wise .apply(lambda ...).
+    # Benchmarked ~44.6x faster on the full FD001 dataset (20,631 rows),
+    # see benchmarks/vectorization_benchmark.py.
+    df['failure'] = (df['RUL'] <= CONFIG["failure_threshold"]).astype(int)
     
     logging.info(f"Data Transformed: {df.shape}")
     return df
-- 
2.43.0
