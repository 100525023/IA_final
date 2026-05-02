# Import required dependencies
import numpy as np
import mdptoolbox

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P() -> np.ndarray:
        """ Function that generates the probabilities (transition) matrix """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...

    @staticmethod
    def generate_R() -> np.ndarray:
        """ Function that generates the rewards (costs) matrix """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...

    @staticmethod
    def control_iteration() -> np.int32:
        """ Function that computes one control-iteration """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...

    @staticmethod
    def control_loop(demand: np.ndarray, 
                     probs: np.ndarray,
                     n_states: np.int32, 
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand """
        ### TO BE COMPLETED BY THE STUDENTS ###

        ### DUMMY BEHAVIOUR TO PREVENT CRASHING (MUST BE DELETED AFTER THE FULL IMPLEMENTATION) ###
        return np.zeros_like(a=demand, dtype=np.float64)
        ### ###
