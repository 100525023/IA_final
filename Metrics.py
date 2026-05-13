import numpy as np


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the mean absolute error between demand and reactor response.
    Every error contributes proportionally to its size, so large spikes don't
    dominate the result the way they would with squared errors.
    """
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the mean squared error between demand and reactor response.
    Squaring the residuals makes large errors weigh much more than small ones,
    which is useful for spotting isolated moments of poor control.
    """
    return np.mean((y_true - y_pred) ** 2)


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the R² coefficient of determination. Values close to 1 mean the
    reactor tracks the demand well; values near or below 0 mean it does no better
    than simply predicting the mean at every step. Returns 0.0 if all true values
    are identical to avoid division by zero.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the Pearson correlation coefficient between demand and response.
    It measures whether the two signals move together, regardless of their absolute
    values. Returns 0.0 if either signal has zero variance.
    """
    cov    = np.cov(y_true, y_pred)[0, 1]
    std_y  = np.std(y_true, ddof=1)
    std_yh = np.std(y_pred, ddof=1)

    if std_y == 0.0 or std_yh == 0.0:
        return 0.0

    return cov / (std_y * std_yh)
