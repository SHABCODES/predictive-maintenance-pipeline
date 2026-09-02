"""
Benchmark: row-wise .apply(lambda) vs. vectorized boolean comparison
for generating the binary 'failure' label from RUL.

Run from repo root:
    python benchmarks/vectorization_benchmark.py

Background:
The original implementation used df['RUL'].apply(lambda x: 1 if x <= t else 0),
which evaluates the lambda once per row in a Python-level loop under the hood.
Pandas/NumPy support vectorized comparison directly on the underlying array,
which avoids the per-row Python function call overhead entirely.
"""
import sys
import time

sys.path.insert(0, "src")
sys.path.insert(0, ".")

from extract import extract_data
from config import CONFIG

N_RUNS = 50


def main():
    df = extract_data("data/train_FD001.txt")

    max_cycle = df.groupby("engine_id")["cycle"].max().reset_index()
    max_cycle.columns = ["engine_id", "max_cycle"]
    df = df.merge(max_cycle, on="engine_id")
    df["RUL"] = df["max_cycle"] - df["cycle"]

    threshold = CONFIG["failure_threshold"]

    # Old approach: row-wise apply
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        result_apply = df["RUL"].apply(lambda x: 1 if x <= threshold else 0)
    apply_time = (time.perf_counter() - t0) / N_RUNS

    # New approach: vectorized boolean comparison
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        result_vec = (df["RUL"] <= threshold).astype(int)
    vec_time = (time.perf_counter() - t0) / N_RUNS

    assert result_apply.equals(result_vec), "Vectorized result diverges from apply() result"

    print(f"Dataset size: {len(df)} rows")
    print(f".apply(lambda) time:   {apply_time * 1000:.4f} ms (avg over {N_RUNS} runs)")
    print(f"vectorized comparison: {vec_time * 1000:.4f} ms (avg over {N_RUNS} runs)")
    print(f"Speedup: {apply_time / vec_time:.1f}x")
    print(f"Results identical: {result_apply.equals(result_vec)}")


if __name__ == "__main__":
    main()
