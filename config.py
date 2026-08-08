CONFIG = {
    "window_size": 5,
    "failure_threshold": 30,
    "test_split": 0.2,
}

assert CONFIG["window_size"] > 0, "window_size must be positive"
assert CONFIG["failure_threshold"] > 0, "failure_threshold must be positive"
assert 0 < CONFIG["test_split"] < 1, "test_split must be between 0 and 1"
