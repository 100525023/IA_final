# Import required dependencies
import numpy as np
import argparse
import json
from Reactor import Reactor
from ControlModule import ControlModule
from DemandGenerator import generate_demand
from Metrics import *
from Plotter import *

def get_args() -> tuple[Reactor, np.float64]:
    # Define the parser object
    parser = argparse.ArgumentParser()

    # Define the expected arguments to parse and their data types
    parser.add_argument("--input-reactor", "-i", type=str, help="Path of the reactor's JSON file")
    parser.add_argument("--gamma", "-g", type=float, help="Discount factor used in the MDP")
    parser.add_argument("--random-seed", "-r", type=int, help="Pseudo-random number generator seed")

    # Parse the arguments
    args = parser.parse_args()
    
    # Some verbose to check the correct parsing of the input arguments
    print(f"Loading reactor from file: {args.input_reactor}")
    print(f"Using gamma (discount factor): {args.gamma}")
    print(f"Using {args.random_seed} as random seed")

    # Build the Reactor object by reading the reactor's JSON file
    with open(args.input_reactor, 'r', encoding='utf-8') as file:
        json_data = json.load(fp=file)
        reactor   = Reactor(model=json_data['model'],
                            effective_section=float(json_data['effective_section']),
                            neutron_flux=float(json_data['neutron_flux']),
                            core_volume=float(json_data['core_volume']),
                            fision_energy=float(json_data['fision_energy']),
                            probabilities=dict(json_data['probabilities']))
        
    # Some verbose of the reactor loaded
    print(reactor)  # Overloaded in the __str__ method of Reactor's class
    
    # Return the Reactor object, gamma and the random seed
    return reactor, args.gamma, args.random_seed

def main() -> None:
    # Parse the main arguments
    reactor, gamma, random_seed = get_args()

    # Set the random seed
    np.random.seed(random_seed)

    # Get the probabilities from the reactor's dynamics
    probs = np.array([reactor.probabilities['decrease'], 
                      reactor.probabilities['maintain'], 
                      reactor.probabilities['increase']], dtype=np.float64)
    
    # Make a radar-plot with the reactor probabilities
    plot_reactor_as_radar(probs=probs)
    
    # Generate a random power demand
    demand = generate_demand(n_samples=512)

    # Define the number of MDP's states, actions and the discount factor (gamma)
    n_states  = 100
    n_actions = 3

    # Get the response time-series (answer to the demand time-series)
    response  = ControlModule.control_loop(demand=demand, 
                                           probs=probs,
                                           n_states=n_states,
                                           n_actions=n_actions,
                                           gamma=gamma)
    
    # Plot the original power demand
    plot_demand(demand=demand)

    # Plot the original power demand and the corresponding response
    plot_demand_response(demand=demand, response=response)

    # Plot the power response and the control bar percentaje employed
    plot_control_bars_usage(reactor=reactor, response=response)

    # Plot the correlation scatter-plot of both the demand and the response time-series
    plot_correlation(demand=demand, response=response)

    # Print the four regression metrics for the current demand-response data
    _MAE  = MAE(y_true=demand, y_pred=response)
    _MSE  = MSE(y_true=demand, y_pred=response)
    _R2   = R2(y_true=demand, y_pred=response)
    _Corr = Corr(y_true=demand, y_pred=response)
    print(f"MAE={_MAE:.6f}")
    print(f"MSE={_MSE:.6f}")
    print(f"R2={_R2:.6f}")
    print(f"Corr={_Corr:.6f}")

    # Plot the MAE and the MSE in a bar-plot
    plot_mae_and_mse(MAE=_MAE, MSE=_MSE)

    # Plot the R2 and the Corr in a bar-plot
    plot_r2_and_pearson(R2=_R2, Pearson=_Corr)

if __name__ == '__main__':
    main()
