import numpy as np


def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calcula el Error Absoluto Medio entre la demanda real y la respuesta del reactor.

    Básicamente mide, en promedio, cuánto se aleja la respuesta de lo que se pedía.
    Un MAE de 0 sería perfecto. Al usar valor absoluto en lugar de cuadrados,
    los errores grandes no pesan desproporcionadamente: un error de 0.1 cuenta
    exactamente el doble que uno de 0.05.
    """
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calcula el Error Cuadrático Medio entre la demanda real y la respuesta del reactor.

    A diferencia del MAE, elevar al cuadrado penaliza mucho más los errores grandes.
    Eso lo hace útil para detectar si el reactor tiene picos puntuales de descontrol,
    aunque sea muy preciso el resto del tiempo. Un MSE de 0 indica ajuste perfecto.
    """
    return np.mean((y_true - y_pred) ** 2)


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calcula el coeficiente de determinación R², que mide qué tan bien sigue el
    reactor la forma de la curva de demanda.

    Un R² cercano a 1 significa que el reactor captura bien la tendencia de la demanda.
    Un R² en torno a 0 o negativo indica que el reactor no lo hace mejor que
    simplemente predecir siempre la media, lo cual sería bastante malo.

    En el caso degenerado en que todos los valores de demanda sean iguales,
    devuelve 0.0 para evitar división por cero.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0.0:
        return 0.0

    return 1.0 - (ss_res / ss_tot)


def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """
    Calcula el coeficiente de correlación de Pearson entre demanda y respuesta.

    Mide si las dos curvas "suben y bajan juntas". Un valor de 1 indica que
    se mueven perfectamente al unísono, -1 que van en sentido contrario, y 0
    que no hay relación lineal entre ellas.

    A diferencia del R², esto no mide exactitud sino sincronía: un reactor
    puede correlacionar bien con la demanda pero estar sistemáticamente desfasado
    en amplitud.

    Si alguna de las dos señales no varía nada (varianza cero), devuelve 0.0.
    """
    cov    = np.cov(y_true, y_pred)[0, 1]
    std_y  = np.std(y_true, ddof=1)
    std_yh = np.std(y_pred, ddof=1)

    if std_y == 0.0 or std_yh == 0.0:
        return 0.0

    return cov / (std_y * std_yh)
