import os

CONFIG = {
    "window_size": int(os.getenv("WINDOW_SIZE", "5")),
    "failure_threshold": int(os.getenv("FAILURE_THRESHOLD", "30")),
    "test_split": float(os.getenv("TEST_SPLIT", "0.2")),
}

assert CONFIG["window_size"] > 0, "window_size must be positive"
assert CONFIG["failure_threshold"] > 0, "failure_threshold must be positive"
assert 0 < CONFIG["test_split"] < 1, "test_split must be between 0 and 1"
