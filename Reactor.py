import numpy as np


class Reactor:
    """
    Physical model of a nuclear reactor. It holds the core parameters (fission cross-section,
    neutron flux, core volume and fission energy) and computes the maximum power and the
    decay constant k automatically on construction. It also carries the stochastic transition
    probabilities that describe how predictably the reactor responds to each control action.
    """

    def __init__(self,
                 model: str,
                 effective_section: np.float64,
                 neutron_flux: np.float64,
                 core_volume: np.float64,
                 fision_energy: np.float64,
                 probabilities: dict):
        """
        Sets up the reactor from its physical parameters and stochastic profile.
        The probabilities come from the reactor's JSON file and describe how well
        it obeys decrease, maintain and increase commands.
        """
        self.model             = model
        self.effective_section = effective_section
        self.neutron_flux      = neutron_flux
        self.core_volume       = core_volume
        self.fision_energy     = fision_energy
        self.probabilities     = probabilities
        self.max_power         = self.compute_max_power()
        self.k                 = self.compute_k()

    def __str__(self) -> str:
        """Returns a human-readable summary of the reactor's physical parameters."""
        lines = [
            f"Model:             {self.model}",
            f"Effective section: {self.effective_section} cm^-1",
            f"Neutron flux:      {self.neutron_flux} neutrons / (cm^2 · s)",
            f"Core volume:       {self.core_volume} cm^3",
            f"Fission energy:    {self.fision_energy} J",
            f"Probabilities:     {self.probabilities}",
        ]
        return "\n".join(lines)

    def compute_max_power(self) -> np.float64:
        """
        Computes the theoretical maximum thermal power using the standard neutronics
        formula P_max = Sigma_f · phi · V · E_f. This is the power the reactor would
        produce with no control rods inserted at all.
        """
        return self.effective_section * self.neutron_flux * self.core_volume * self.fision_energy

    def compute_k(self) -> np.float64:
        """
        Computes the decay constant k so that full rod insertion (B = 1) brings power
        down to nearly zero (1e-6 W) while no insertion (B = 0) gives maximum power.
        """
        return -np.log(1e-6 / self.max_power)

    def compute_power(self, control_bars_insertion: np.float64) -> np.float64:
        """
        Returns the normalized power fraction for a given rod insertion level, following
        an exponential decay P(B) = e^(-k·B). B = 0 means full power, B = 1 means nearly off.
        """
        B = np.clip(control_bars_insertion, 0.0, 1.0)
        return np.exp(-self.k * B)

    def compute_control_bars_insertion(self, power: np.float64) -> np.float64:
        """
        The inverse of compute_power: given a desired power level, returns how far
        the control rods need to be inserted to achieve it. Useful for the plots.
        """
        p = np.clip(power, 1e-10, 1.0)
        return -np.log(p) / self.k
