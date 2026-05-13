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
    Parses the command-line arguments and builds the Reactor object from the given JSON file.
    Expects --input-reactor, --gamma and --random-seed, and returns them together with the
    constructed reactor instance.
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
    Entry point of the simulation. Loads the reactor, generates a random demand curve,
    runs the MDP control loop, computes quality metrics and saves all plots to a
    subdirectory named after the reactor model.
    """
    reactor, gamma, random_seed, reactor_path = get_args()

    np.random.seed(random_seed)

    import Plotter
    Plotter._PLOTS_DIR = os.path.join("plots", reactor.model)
    os.makedirs(Plotter._PLOTS_DIR, exist_ok=True)
    Plotter._plot_counter[0] = 0
    print(f"Plots will be saved to: {Plotter._PLOTS_DIR}/")

    probs = np.array([
        reactor.probabilities['decrease'],
        reactor.probabilities['maintain'],
        reactor.probabilities['increase']
    ], dtype=np.float64)

    plot_reactor_as_radar(probs=probs)

    demand = generate_demand(n_samples=512)

    n_states  = 100
    n_actions = 3

    response = ControlModule.control_loop(
        demand=demand,
        probs=probs,
        n_states=n_states,
        n_actions=n_actions,
        gamma=gamma
    )

    plot_demand(demand=demand)
    plot_demand_response(demand=demand, response=response)
    plot_control_bars_usage(reactor=reactor, response=response)
    plot_correlation(demand=demand, response=response)

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
