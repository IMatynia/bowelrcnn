import logging
import time


def log_exec_time(func):
    def inner(*args, **kwargs):
        start_time = time.perf_counter_ns()
        ret_val = func(*args, **kwargs)
        total = time.perf_counter_ns() - start_time
        func_name = func.__name__
        logging.info(f"{func_name:50} @ {total/1e6:5.5}ms")
        return ret_val
    return inner
