from bise.threshold_fn import tanh_threshold


def tanh_run_times(start_value, end_value, fade_start: float = None):
    def fn(t):
        return (-tanh_threshold(t - fade_start - 1) + 1) * (start_value - end_value) + end_value
    return fn
