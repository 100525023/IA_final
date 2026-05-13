import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Backend no interactivo: renderiza a fichero sin abrir ventana
import matplotlib.pyplot as plt
from Reactor import Reactor

# Carpeta donde se guardarán todas las gráficas generadas
_PLOTS_DIR = "plots"
os.makedirs(_PLOTS_DIR, exist_ok=True)

# Contador global para prefijar los nombres de archivo con un índice ordenado
_plot_counter = [0]


def _save(filename: str) -> None:
    """
    Guarda la figura actual de matplotlib en la carpeta de plots y la cierra.

    Añade automáticamente un prefijo numérico al nombre del archivo para que,
    al listarlos, aparezcan en el orden en que se generaron.
    """
    _plot_counter[0] += 1
    path = os.path.join(_PLOTS_DIR, f"{_plot_counter[0]:02d}_{filename}")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot saved] {path}")


def plot_demand(demand: np.ndarray) -> None:
    """
    Dibuja la curva de demanda normalizada a lo largo del tiempo.

    Es la primera gráfica que conviene ver: muestra qué se le estará pidiendo
    al reactor a lo largo de toda la simulación.
    """
    plt.figure(figsize=(8, 8))
    plt.title("Evolution in the power demand")
    plt.plot(range(demand.shape[0]), demand, label='Demand')
    plt.xlabel("Time")
    plt.ylabel("Power demand (0 - 1)")
    plt.legend()
    plt.grid(True)
    _save("demand.png")


def plot_demand_response(demand: np.ndarray, response: np.ndarray) -> None:
    """
    Superpone la demanda y la respuesta del reactor en la misma gráfica.

    Así se puede ver a simple vista si el reactor sigue bien la demanda o se
    queda rezagado / se dispara en los momentos de cambio brusco.
    """
    plt.figure(figsize=(8, 8))
    plt.title("Power demand vs. Power response")
    plt.plot(range(demand.shape[0]), demand, label='Demand')
    plt.plot(range(response.shape[0]), response, label='Response')
    plt.xlabel("Time")
    plt.ylabel("Power value (0 - 1)")
    plt.legend()
    plt.grid(True)
    _save("demand_response.png")


def plot_correlation(demand: np.ndarray, response: np.ndarray) -> None:
    """
    Muestra la correlación entre demanda y respuesta como un scatter plot,
    con una recta de regresión lineal encima.

    Si los puntos se agrupan cerca de la diagonal, el reactor está siguiendo
    bien la demanda. Una nube dispersa o una pendiente muy diferente de 1
    indica problemas de seguimiento.
    """
    # Regresión lineal por ecuaciones normales
    X = np.ones(shape=(demand.shape[0], 2), dtype=np.float64)
    X[:, 1] = demand
    thetas = np.linalg.inv(X.T @ X) @ X.T @ response
    x_reg  = np.array([np.min(a=demand), np.max(a=demand)], dtype=np.float64)
    y_reg  = thetas[0] + thetas[1] * x_reg

    plt.figure(figsize=(8, 8))
    plt.title("Demand - Response correlation")
    plt.scatter(demand, response, label='Data', edgecolor='white', zorder=2)
    plt.plot(x_reg, y_reg, label='Linear Regression', color='black')
    plt.xlabel("Power value (0 - 1)")
    plt.ylabel("Power value (0 - 1)")
    plt.legend()
    plt.grid(True)
    _save("correlation.png")


def plot_reactor_as_radar(probs: np.ndarray) -> None:
    """
    Muestra el perfil estocástico del reactor como un gráfico de radar.

    Cada eje representa una acción (Disminuir, Mantener, Aumentar), y el valor
    trazado es la probabilidad de que esa acción produzca exactamente el resultado
    deseado. Un polígono grande y cercano al de referencia gris significa un reactor
    muy predecible; uno pequeño o irregular significa que el reactor tiene tendencia
    a hacer "lo que le da la gana".
    """
    labels = ['D', 'M', 'I']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = probs[:, 1].tolist()
    ideal_values = [1.0, 1.0, 1.0]

    # Cerramos el polígono repitiendo el primer elemento
    values       += values[:1]
    angles       += angles[:1]
    labels       += labels[:1]
    ideal_values += ideal_values[:1]

    _, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_title("Nuclear reactor stochastic dynamics", fontsize=20)
    ax.plot(angles, values, linewidth=2, c='g', zorder=2)
    ax.fill(angles, values, alpha=0.25, c='g', zorder=2, label='Current reactor')
    ax.plot(angles, ideal_values, linewidth=2, alpha=0.25, c='gray')
    ax.fill(angles, ideal_values, alpha=0.2, c='gray', label='Ideal reactor')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=15)
    ax.set_ylim(0.0, 1.0)
    plt.legend()
    _save("radar.png")


def plot_control_bars_usage(reactor: Reactor, response: np.ndarray) -> None:
    """
    Dibuja juntos la respuesta de potencia y el porcentaje de inserción de barras
    de control necesario para conseguirla.

    Es útil para entender la relación física: cuando el reactor está a poca potencia,
    las barras tienen que estar muy metidas, y viceversa. Si ambas curvas se cruzan
    de forma extraña, puede indicar un comportamiento anómalo.
    """
    control_bar_usage = np.zeros_like(a=response, dtype=np.float64)
    for i in range(response.shape[0]):
        control_bar_usage[i] = reactor.compute_control_bars_insertion(power=response[i])

    plt.figure(figsize=(8, 8))
    plt.title("Response and control bar insertion plot")
    plt.plot(range(response.shape[0]), response, label='Response', color='gray')
    plt.plot(range(control_bar_usage.shape[0]), control_bar_usage,
             label='Control bar insertion', color='black')
    plt.xlabel("Time")
    plt.ylabel("Power (%) | Insertion (%)")
    plt.legend()
    plt.grid(True)
    _save("control_bars.png")


def plot_mae_and_mse(MAE: np.float64, MSE: np.float64) -> None:
    """
    Representa MAE y MSE como un gráfico de barras para comparar visualmente
    los dos errores de un vistazo.

    Si el MSE es mucho mayor que el MAE, significa que hay algunos instantes con
    errores grandes que están inflando el promedio cuadrático.
    """
    plt.figure(figsize=(6, 6))
    categories = ['MAE', 'MSE']
    values     = [MAE, MSE]
    plt.bar(categories, values, color=['blue', 'orange'], edgecolor='black', zorder=2)
    plt.title('MAE and MSE bar-plot')
    plt.xlabel('Regression error metric')
    plt.ylabel('Error')
    plt.grid(True)
    _save("mae_mse.png")


def plot_r2_and_pearson(R2: np.float64, Pearson: np.float64) -> None:
    """
    Representa R² y la correlación de Pearson como un gráfico de barras.

    Ambos miden cosas ligeramente distintas: R² valora la calidad del ajuste
    absoluto y Pearson mide la sincronía de las dos curvas. Verlos juntos ayuda
    a entender si el reactor falla en magnitud, en timing, o en ambas cosas.
    """
    plt.figure(figsize=(6, 6))
    categories = ['R2', 'Pearson']
    values     = [R2, Pearson]
    plt.bar(categories, values, color=['blue', 'orange'], edgecolor='black', zorder=2)
    plt.title("R² and Pearson's Correlation bar-plot")
    plt.xlabel('Regression quality metric')
    plt.ylabel('Quality')
    plt.grid(True)
    _save("r2_pearson.png")
