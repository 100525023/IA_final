import numpy as np
import argparse
import json
import os
from Reactor import Reactor
from ControlModule import ControlModule
from DemandGenerator import generate_demand
from Metrics import *
from Plotter import *


def get_args() -> tuple[Reactor, np.float64, int, str]:
    """
    Lee los argumentos de la línea de comandos y construye el objeto Reactor
    a partir del fichero JSON indicado.

    Se esperan tres argumentos:
    - --input-reactor (-i): ruta al JSON con la configuración del reactor.
    - --gamma (-g): factor de descuento del MDP.
    - --random-seed (-r): semilla para el generador de números aleatorios.

    Devuelve una tupla con el reactor ya construido, gamma, la semilla y la ruta
    al fichero del reactor.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-reactor", "-i", type=str,
                        help="Path to the reactor's JSON configuration file.")
    parser.add_argument("--gamma", "-g", type=float,
                        help="Discount factor used in the MDP (typically in (0, 1]).")
    parser.add_argument("--random-seed", "-r", type=int,
                        help="Seed for the pseudo-random number generator.")

    args = parser.parse_args()

    print(f"Loading reactor from:    {args.input_reactor}")
    print(f"Discount factor (gamma): {args.gamma}")
    print(f"Random seed:             {args.random_seed}")

    with open(args.input_reactor, 'r', encoding='utf-8') as file:
        json_data = json.load(fp=file)
        reactor = Reactor(
            model=json_data['model'],
            effective_section=float(json_data['effective_section']),
            neutron_flux=float(json_data['neutron_flux']),
            core_volume=float(json_data['core_volume']),
            fision_energy=float(json_data['fision_energy']),
            probabilities=dict(json_data['probabilities'])
        )

    print(reactor)

    return reactor, args.gamma, args.random_seed, args.input_reactor


def main() -> None:
    """
    Punto de entrada de la simulación del control del reactor nuclear.

    Carga el reactor desde su fichero de configuración.
    Genera una curva de demanda aleatoria.
    Ejecuta el bucle de control basado en MDP.
    Calcula las métricas de calidad del seguimiento.
    Guarda todas las gráficas en disco.

    Las gráficas se guardan en una subcarpeta con el nombre del modelo del reactor,
    así cada reactor tiene su propio directorio de resultados.
    """
    reactor, gamma, random_seed, reactor_path = get_args()

    np.random.seed(random_seed)

    # Redirigimos las gráficas a una subcarpeta con el nombre del reactor
    import Plotter
    Plotter._PLOTS_DIR = os.path.join("plots", reactor.model)
    os.makedirs(Plotter._PLOTS_DIR, exist_ok=True)
    Plotter._plot_counter[0] = 0
    print(f"Plots will be saved to: {Plotter._PLOTS_DIR}/")

    # Extraemos las probabilidades de transición como array numpy
    probs = np.array([
        reactor.probabilities['decrease'],
        reactor.probabilities['maintain'],
        reactor.probabilities['increase']
    ], dtype=np.float64)

    # Radar: muestra qué tan predecible es este reactor
    plot_reactor_as_radar(probs=probs)

    # Generamos la curva de demanda sintética
    demand = generate_demand(n_samples=512)

    # Configuración del MDP
    n_states  = 100
    n_actions = 3

    # Ejecutamos el bucle de control y obtenemos la respuesta del reactor
    response = ControlModule.control_loop(
        demand=demand,
        probs=probs,
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma
    )

    # Generamos y guardamos todas las gráficas
    plot_demand(demand=demand)
    plot_demand_response(demand=demand, response=response)
    plot_control_bars_usage(reactor=reactor, response=response)
    plot_correlation(demand=demand, response=response)

    # Calculamos y mostramos las métricas de calidad
    _MAE  = MAE(y_true=demand, y_pred=response)
    _MSE  = MSE(y_true=demand, y_pred=response)
    _R2   = R2(y_true=demand, y_pred=response)
    _Corr = Corr(y_true=demand, y_pred=response)

    print(f"MAE  = {_MAE:.6f}")
    print(f"MSE  = {_MSE:.6f}")
    print(f"R²   = {_R2:.6f}")
    print(f"Corr = {_Corr:.6f}")

    plot_mae_and_mse(MAE=_MAE, MSE=_MSE)
    plot_r2_and_pearson(R2=_R2, Pearson=_Corr)

    print(f"\nSimulation complete. Plots saved to: {Plotter._PLOTS_DIR}/")


if __name__ == '__main__':
    main()
