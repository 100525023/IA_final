# Import required dependencies
import numpy as np

def random_recursive_signal(n_samples: np.int32, start: np.float64, scale: np.float64=1.0) -> np.ndarray:
    """ Builds a realistic recursive signal based on random gaussian noise """
    # Initialize a Numpy array of dimension = n_samples
    signal = np.zeros(shape=n_samples, dtype=np.float64)

    # Compute random gaussian noise that will be recursively added to the signal
    noise = np.random.normal(loc=0.0, scale=scale, size=n_samples - 1)

    # Set the starting point without noise
    signal[0] = start

    # Recursively add the random noise
    for i in range(1, n_samples):
        signal[i] = signal[i - 1] + noise[i - 1]

    # Return the already computed signal
    return signal

def scale_signal(signal: np.ndarray, method: str='MinMax') -> np.ndarray:
    """ Scales a signal using different normalization techniques """
    method_lower_case = method.lower()  # Performed to allow for slight changes in the method string

    # Select the normalization method
    match method_lower_case:
        # Apply the MinMax normalization
        case 'minmax':
            _min, _max = np.min(a=signal), np.max(a=signal)
            return (signal - _min) / (_max - _min)

        # Apply the STD normal distribution standarization
        case 'std':
            mu, sigma = np.mean(a=signal), np.std(a=signal)
            return (signal - mu) / sigma

        # Default case for error management
        case _:
            raise ValueError(f"Error: '{method}' method is not recognised")
        
def moving_average_filter(signal: np.ndarray, window_size: np.int32=7) -> np.ndarray:
    """ Implementation of the MA-filter for signal smoothing (removing high frequency noise) """
    # Manage exceptions when the window-size is inconsistent
    if window_size <= 0:
        raise ValueError("Error: window size must be greater than zero!")
    
    # Compute the required amount of padding to keep the signal size after the convolutions
    p_size         = window_size - 1

    # Add the padding to the signal (repeating the last value rather than adding zeros)
    signal_padding                   = np.zeros(shape=signal.shape[0] + p_size, dtype=np.float64)
    signal_padding[:signal.shape[0]] = signal
    signal_padding[signal.shape[0]:] = signal[-1]

    # Initialize the output signal array and fill it with the corresponding convolutions (of the MA-filter)
    output_signal = np.zeros(shape=signal.shape[0], dtype=np.float64)
    for i in range(signal.shape[0]):
        output_signal[i] = np.mean(a=signal[ i : i + window_size ])

    # Return the filtered signal
    return output_signal

def generate_demand(n_samples: np.int32, start: np.float64=None, scale: np.float64=None, apply_filtering: bool=True) -> np.ndarray:
    """ Generates a realistic random power-demand signal (time-series) """
    # Generate a random recursive signal
    demand_signal = random_recursive_signal(n_samples=n_samples,
                                            start=start if start is not None else np.random.uniform(low=0.0, high=100.0),
                                            scale=scale if scale is not None else 1.0)
    
    # Normalize the signal by using the MinMax method
    demand_signal_norm = scale_signal(signal=demand_signal, method='MinMax')

    # Check if filtering is required by the programmer
    if apply_filtering:
        # Filter the signal to remove high frequency noise from it and return the final result
        return moving_average_filter(signal=demand_signal_norm)
    
    # If not, simply return the normalized signal
    return demand_signal_norm
