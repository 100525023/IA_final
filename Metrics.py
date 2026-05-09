import numpy as np


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the Mean Absolute Error (MAE) between the true and predicted values.

    MAE measures the average magnitude of the errors without considering their direction.
    It is expressed in the same units as the target variable and is less sensitive to
    outliers than MSE.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (demand curve).
    y_pred : np.ndarray
        Predicted values (reactor response curve).

    Returns
    -------
    float
        Mean absolute error. A value of 0 indicates a perfect match.
    """
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the Mean Squared Error (MSE) between the true and predicted values.

    MSE penalizes larger errors more heavily than MAE due to the squaring of residuals.
    It is particularly useful for detecting and discouraging large deviations between
    the demand and the reactor response.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (demand curve).
    y_pred : np.ndarray
        Predicted values (reactor response curve).

    Returns
    -------
    float
        Mean squared error. A value of 0 indicates a perfect match.
    """
    return np.mean((y_true - y_pred) ** 2)


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the R² coefficient of determination.

    R² measures how much of the variance in the true signal is explained by the
    predicted signal, benchmarked against a trivial mean-based predictor. Values
    close to 1 indicate a high-quality fit; values at or below 0 indicate that the
    model performs no better than simply predicting the mean.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (demand curve).
    y_pred : np.ndarray
        Predicted values (reactor response curve).

    Returns
    -------
    float
        R² coefficient, typically in the range (-inf, 1].
        Returns 0.0 if all true values are identical (degenerate case).
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Computes the Pearson Correlation Coefficient between the true and predicted values.

    The Pearson coefficient measures the strength and direction of the linear
    relationship between the two signals. It is bounded in [-1, 1], where 1 means
    perfect positive correlation, -1 means perfect negative correlation, and 0
    indicates no linear relationship.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values (demand curve).
    y_pred : np.ndarray
        Predicted values (reactor response curve).

    Returns
    -------
    float
        Pearson correlation coefficient in the range [-1, 1].
        Returns 0.0 if either signal has zero variance (degenerate case).
    """
    cov    = np.cov(y_true, y_pred)[0, 1]
    std_y  = np.std(y_true, ddof=1)
    std_yh = np.std(y_pred, ddof=1)

    if std_y == 0.0 or std_yh == 0.0:
        return 0.0

    return cov / (std_y * std_yh)
