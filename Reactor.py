# Import required dependencies
import numpy as np

class Reactor:
    def __init__(self,
                 model: str,
                 effective_section: np.float64,
                 neutron_flux: np.float64,
                 core_volume: np.float64,
                 fision_energy: np.float64,
                 probabilities: dict):
        """ Constructor of the Reactor class """
        self.model             = model
        self.effective_section = effective_section
        self.neutron_flux      = neutron_flux
        self.core_volume       = core_volume
        self.fision_energy     = fision_energy
        self.probabilities     = probabilities
        self.max_power         = self.compute_max_power()
        self.k                 = self.compute_k()

    def __str__(self) -> str:
        """ Overloading of the native __str__ function to print the class instances """
        _str  = f"Model: {self.model}\n"
        _str += f"Effective section: {self.effective_section} cm^-1\n"
        _str += f"Neutron flux: {self.neutron_flux} neutrons / (cm^2 · s)\n"
        _str += f"Core volume: {self.core_volume} cm^3\n"
        _str += f"Fision energy: {self.fision_energy} J\n"
        _str += f"Probabilities: {self.probabilities}"
        return _str

    def compute_max_power(self) -> np.float64:
        """ Computes the maximum power of a reactor based on its physical features """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...
    
    def compute_k(self) -> np.float64:
        """ Computes the value of the k-constant """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...
    
    def compute_power(self, control_bars_insertion: np.float64) -> np.float64:
        """ Computes the power delivered (%) by the reactor based on the % of control-bars inserted """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...
    
    def compute_control_bars_insertion(self, power: np.float64) -> np.float64:
        """ Computes the % of controls-bars inserted based on the % of power delivered by the reactor """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...
