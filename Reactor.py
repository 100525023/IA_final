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
        """
        Computes the maximum thermal power of a reactor based on its physical features.
        Formula: Pmax = Sigma_f * phi * V * E_f
        """
        return self.effective_section * self.neutron_flux * self.core_volume * self.fision_energy

    def compute_k(self) -> np.float64:
        """
        Computes the k-constant that governs the exponential decay of power with control rod insertion.
        Formula: k = -ln(10^-6 / Pmax)
        """
        return -np.log(1e-6 / self.max_power)

    def compute_power(self, control_bars_insertion: np.float64) -> np.float64:
        """
        Computes the fraction of power [0, 1] delivered by the reactor based on the
        percentage of control bars inserted (B in [0, 1]).
        Formula: P(B) = Pmax * e^(-k*B) / Pmax = e^(-k*B)
        """
        B = np.clip(control_bars_insertion, 0.0, 1.0)
        return np.exp(-self.k * B)

    def compute_control_bars_insertion(self, power: np.float64) -> np.float64:
        """
        Computes the % of control bars inserted given the power fraction [0, 1].
        Inverse of compute_power: B = -ln(power) / k
        """
        p = np.clip(power, 1e-10, 1.0)
        return -np.log(p) / self.k
