# Import required dependencies
import numpy as np
import matplotlib.pyplot as plt
from Reactor import Reactor

def plot_demand(demand: np.ndarray) -> None:
    """ Plots the demand as a time-series (signal) """
    plt.figure(figsize=(8, 8))
    plt.title("Evolution in the power demand")
    plt.plot(range(demand.shape[0]), demand, label='Demand')
    plt.xlabel("Time")
    plt.ylabel("Power demand (0 - 1)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def plot_demand_response(demand: np.ndarray, response: np.ndarray) -> None:
    """ Plots the demand-response as two different time-series """
    plt.figure(figsize=(8, 8))
    plt.title("Power demand vs. Power response")
    plt.plot(range(demand.shape[0]), demand, label='Demand')
    plt.plot(range(response.shape[0]), response, label='Response')
    plt.xlabel("Time")
    plt.ylabel("Power value (0 - 1)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def plot_correlation(demand: np.ndarray, response: np.ndarray) -> None:
    """ Plots the demand-response correlation (scatter-plot) """
    # Compute the 2D Linear-Regression for the plot
    X       = np.ones(shape=(demand.shape[0], 2), dtype=np.float64)
    X[:, 1] = demand
    thetas  = np.linalg.inv(X.T @ X) @ X.T @ response
    x_reg   = np.array([ np.min(a=demand), np.max(a=demand) ], dtype=np.float64)
    y_reg   = thetas[0] + thetas[1] * x_reg

    # Perform the scatter-plot with the linear regression
    plt.figure(figsize=(8, 8))
    plt.title("Demand - Response correlation")
    plt.scatter(demand, response, label='Data', edgecolor='white', zorder=2)
    plt.plot(x_reg, y_reg, label='Linear Regression', color='black')
    plt.xlabel("Power value (0 - 1)")
    plt.ylabel("Power value (0 - 1)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def plot_reactor_as_radar(probs: np.ndarray) -> None:
    """ Plots the probabilities of the reactor in a radar-plot style """
    # Action labels (D: Decrease, M: Maintain, I: Increase)
    labels = ['D', 'M', 'I']

    # Compute the angles
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

    # Get the values from the Numpy Array (now a Python list)
    values = probs[:, 1].tolist()

    # Compute the ideal values for the perfect reactor (deterministic)
    ideal_values = [1.0, 1.0, 1.0]

    # Close the polygons
    values       += values[:1]
    angles       += angles[:1]
    labels       += labels[:1]
    ideal_values += ideal_values[:1]

    # Create the polar figure and show it
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
    plt.show()
    return

def plot_control_bars_usage(reactor: Reactor, response: np.ndarray) -> None:
    """ Plots the reactor's response and the corresponding control bar insertion (usage) """
    control_bar_usage = np.zeros_like(a=response, dtype=np.float64)
    for i in range(response.shape[0]):
        control_bar_usage[i] = reactor.compute_control_bars_insertion(power=response[i])
    plt.figure(figsize=(8, 8))
    plt.title("Response and control bar insertion plot")
    plt.plot(range(response.shape[0]), response, label='Response', color='gray')
    plt.plot(range(control_bar_usage.shape[0]), control_bar_usage, label='Control bar insertion', color='black')
    plt.xlabel("Time")
    plt.ylabel("Power (%) | Insertion (%)")
    plt.legend()
    plt.grid(True)
    plt.show()
    return

def plot_mae_and_mse(MAE: np.float64, MSE: np.float64) -> None:
    categories = ['MAE', 'MSE']
    values     = [MAE, MSE]
    plt.bar(categories, values, color=[ 'blue', 'orange' ], edgecolor='black', zorder=2)
    plt.title('MAE and MSE bar-plot')
    plt.xlabel('Regression error metric')
    plt.ylabel('Error')
    plt.grid(True)
    plt.show()

def plot_r2_and_pearson(R2: np.float64, Pearson: np.float64) -> None:
    categories = ['R2', 'Pearson']
    values     = [R2, Pearson]
    plt.bar(categories, values, color=[ 'blue', 'orange' ], edgecolor='black', zorder=2)
    plt.title("R2 and Pearson's Correlation bar-plot")
    plt.xlabel('Regression quality metric')
    plt.ylabel('Quality')
    plt.grid(True)
    plt.show()
