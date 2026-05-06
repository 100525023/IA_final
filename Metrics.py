# Import required dependencies
import numpy as np

def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Absolute Error (MAE) """
    return np.mean(np.abs(y_true - y_pred))

def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Squared Error (MSE) """
    return np.mean((y_true - y_pred) ** 2)

def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Implementation of the R2 (coefficient of determination) metric.
    Compares the model against a mean-based baseline.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # Avoid division by zero when all true values are identical
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - (ss_res / ss_tot)

def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Implementation of the Pearson Correlation Coefficient.
    Corr = Cov(y, y_hat) / (std(y) * std(y_hat))
    """
    cov    = np.cov(y_true, y_pred)[0, 1]
    std_y  = np.std(y_true)
    std_yh = np.std(y_pred)
    # Avoid division by zero
    if std_y == 0.0 or std_yh == 0.0:
        return 0.0
    return cov / (std_y * std_yh)
