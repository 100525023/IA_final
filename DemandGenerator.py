import numpy as np


def random_recursive_signal(n_samples: np.int32, start: np.float64,
                             scale: np.float64 = 1.0) -> np.ndarray:
    """
    Builds a signal step by step, adding a small Gaussian jump at each point.
    The result looks like a realistic power demand curve. Use scale to control
    how aggressively the signal can change between steps.
    """
    signal = np.zeros(shape=n_samples, dtype=np.float64)
    noise = np.random.normal(loc=0.0, scale=scale, size=n_samples - 1)

    signal[0] = start
    for i in range(1, n_samples):
        signal[i] = signal[i - 1] + noise[i - 1]

    return signal


def scale_signal(signal: np.ndarray, method: str = 'MinMax') -> np.ndarray:
    """
    Normalizes a signal using either MinMax (rescales to [0, 1]) or STD (zero mean,
    unit variance). Raises a ValueError if the method is not recognized.
    """
    method_lower = method.lower()

    match method_lower:
        case 'minmax':
            _min, _max = np.min(a=signal), np.max(a=signal)
            return (signal - _min) / (_max - _min)

        case 'std':
            mu, sigma = np.mean(a=signal), np.std(a=signal)
            return (signal - mu) / sigma

        case _:
            raise ValueError(f"Normalization method '{method}' is not recognized.")


def moving_average_filter(signal: np.ndarray, window_size: np.int32 = 7) -> np.ndarray:
    """
    Smooths the signal by averaging each point with its neighbors. Larger windows
    produce cleaner curves but react more slowly to sudden changes. The end of the
    signal is padded with the last known value to preserve the original length.
    """
    if window_size <= 0:
        raise ValueError("The window size must be a positive integer.")

    p_size = window_size - 1

    signal_padded = np.zeros(shape=signal.shape[0] + p_size, dtype=np.float64)
    signal_padded[:signal.shape[0]] = signal
    signal_padded[signal.shape[0]:] = signal[-1]

    output_signal = np.zeros(shape=signal.shape[0], dtype=np.float64)
    for i in range(signal.shape[0]):
        output_signal[i] = np.mean(a=signal[i : i + window_size])

    return output_signal


def generate_demand(n_samples: np.int32, start: np.float64 = None,
                    scale: np.float64 = None, apply_filtering: bool = True) -> np.ndarray:
    """
    Generates a normalized power demand curve in [0, 1]. It builds a random walk,
    scales it with MinMax, and optionally smooths it with a moving average filter
    to make the result look more like a real demand signal.
    """
    demand_signal = random_recursive_signal(
        n_samples=n_samples,
        start=start if start is not None else np.random.uniform(low=0.0, high=100.0),
        scale=scale if scale is not None else 1.0
    )

    demand_signal_norm = scale_signal(signal=demand_signal, method='MinMax')

    if apply_filtering:
        return moving_average_filter(signal=demand_signal_norm)

    return demand_signal_norm
