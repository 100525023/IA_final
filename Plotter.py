import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend: renders to file without opening a window
import matplotlib.pyplot as plt
from Reactor import Reactor

_PLOTS_DIR = "plots"
os.makedirs(_PLOTS_DIR, exist_ok=True)

# Global counter so files are saved with an ordered numeric prefix
_plot_counter = [0]


def _save(filename: str) -> None:
    """Saves the current figure to the plots directory with a sequential prefix and closes it."""
    _plot_counter[0] += 1
    path = os.path.join(_PLOTS_DIR, f"{_plot_counter[0]:02d}_{filename}")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot saved] {path}")


def plot_demand(demand: np.ndarray) -> None:
    """Plots the normalized demand curve over time, giving a first look at what the reactor will be asked to follow."""
    plt.figure(figsize=(8, 8))
    plt.title("Evolution in the power demand")
    plt.plot(range(demand.shape[0]), demand, label='Demand')
    plt.xlabel("Time")
    plt.ylabel("Power demand (0 - 1)")
    plt.legend()
    plt.grid(True)
    _save("demand.png")


def plot_demand_response(demand: np.ndarray, response: np.ndarray) -> None:
    """Overlays demand and reactor response so it is easy to see how closely the reactor tracks the target."""
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
    Shows a scatter plot of demand vs. response with a linear regression line.
    Points clustered near the diagonal indicate good tracking; a dispersed cloud
    or a very different slope reveals systematic errors.
    """
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
    Draws the reactor's stochastic profile as a radar chart. Each axis represents
    one action and shows the probability of the intended outcome. A large polygon
    means a predictable reactor; a small or irregular one means it often does
    something other than what was asked.
    """
    labels = ['D', 'M', 'I']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values = probs[:, 1].tolist()
    ideal_values = [1.0, 1.0, 1.0]

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
    Plots the reactor response alongside the required control rod insertion. It helps
    understand the physical relationship between power output and rod position, and
    makes any anomalous behavior easy to spot visually.
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
    Bar chart comparing MAE and MSE at a glance. If MSE is much larger than MAE,
    there are isolated moments with big errors inflating the squared average.
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
    Bar chart comparing R² and Pearson correlation. R² measures absolute fit quality
    while Pearson measures synchrony, so seeing both together reveals whether the
    reactor fails in magnitude, in timing, or in both.
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
