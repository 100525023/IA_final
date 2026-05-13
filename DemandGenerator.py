import numpy as np


def random_recursive_signal(n_samples: np.int32, start: np.float64,
                             scale: np.float64 = 1.0) -> np.ndarray:
    """
    Genera una señal construida paso a paso, sumando pequeños saltos aleatorios.

    Funciona como un paseo aleatorio: partimos de un valor inicial y en cada paso
    añadimos un poco de ruido gaussiano. El resultado se parece bastante a cómo
    evoluciona una demanda eléctrica real a lo largo del tiempo.

    'scale' controla cuánto puede variar la señal en cada paso; a mayor escala,
    más agresivos son los cambios.
    """
    signal = np.zeros(shape=n_samples, dtype=np.float64)
    noise = np.random.normal(loc=0.0, scale=scale, size=n_samples - 1)

    signal[0] = start
    for i in range(1, n_samples):
        signal[i] = signal[i - 1] + noise[i - 1]

    return signal


def scale_signal(signal: np.ndarray, method: str = 'MinMax') -> np.ndarray:
    """
    Normaliza una señal para que sus valores queden en un rango manejable.

    Soporta dos modos:
    - 'MinMax': lleva todos los valores al rango [0, 1]. Ideal cuando queremos
      representar algo como un porcentaje de potencia.
    - 'STD': centra la señal en cero con varianza unitaria. Útil para análisis
      estadístico.

    Lanza un ValueError si se pide un método que no existe.
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
    Suaviza una señal promediando cada punto con sus vecinos cercanos.

    Esto elimina los picos bruscos y el ruido de alta frecuencia, dejando una
    curva más limpia y realista. El tamaño de la ventana decide cuánto se suaviza:
    ventanas más grandes producen señales más lisas pero también más lentas en
    reaccionar a los cambios.

    Para no perder puntos al final, rellenamos con el último valor conocido antes
    de aplicar el filtro.
    """
    if window_size <= 0:
        raise ValueError("The window size must be a positive integer.")

    p_size = window_size - 1

    # Rellenamos el final con el último valor para conservar la longitud original
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
    Genera una curva de demanda de potencia sintética y normalizada.

    El proceso es sencillo: primero creamos un paseo aleatorio, lo normalizamos
    a [0, 1] y, si se quiere, le pasamos un filtro de media móvil para que la
    curva resulte más suave y natural.

    Si no se especifica un punto de arranque, se elige uno al azar entre 0 y 100.
    Si no se especifica escala de ruido, se usa 1.0 por defecto.
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
