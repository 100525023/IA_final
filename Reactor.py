import numpy as np


class Reactor:
    """
    Modelo físico de un reactor nuclear.

    Esta clase representa un reactor concreto con sus parámetros físicos reales:
    sección eficaz de fisión, flujo de neutrones, volumen del núcleo y energía
    por fisión. A partir de ellos calcula automáticamente la potencia máxima y
    la constante de decaimiento que gobierna cómo responde el reactor a la
    inserción de las barras de control.

    También lleva consigo las probabilidades estocásticas de transición, que
    definen cuánto "obedece" el reactor cuando le pedimos subir, mantener o
    bajar potencia.
    """

    def __init__(self,
                 model: str,
                 effective_section: np.float64,
                 neutron_flux: np.float64,
                 core_volume: np.float64,
                 fision_energy: np.float64,
                 probabilities: dict):
        """
        Crea un reactor a partir de sus parámetros físicos y su perfil estocástico.

        'model' es simplemente el nombre del reactor (p. ej. "RBMK").
        Los parámetros físicos (sección eficaz, flujo neutrónico, volumen y energía
        de fisión) se usan para calcular la potencia máxima teórica y la constante k.
        'probabilities' viene del JSON del reactor y describe cómo de predecible
        es su comportamiento ante cada acción de control.
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
        """Devuelve un resumen legible con todos los parámetros físicos del reactor."""
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
        Calcula la potencia térmica máxima teórica del reactor en vatios.

        Usa la fórmula clásica de neutrónica:
            P_max = Sigma_f · phi · V · E_f

        Es la potencia que tendría el reactor si no hubiera ninguna barra de
        control insertada y todo funcionara a pleno rendimiento.
        """
        return self.effective_section * self.neutron_flux * self.core_volume * self.fision_energy

    def compute_k(self) -> np.float64:
        """
        Calcula la constante de decaimiento k para el modelo de inserción de barras.

        La idea es que con las barras completamente insertadas (B = 1), la potencia
        baja hasta casi cero (1e-6 W), y con las barras fuera (B = 0) tenemos
        la potencia máxima. k se elige para que esa transición sea suave y exponencial.
        """
        return -np.log(1e-6 / self.max_power)

    def compute_power(self, control_bars_insertion: np.float64) -> np.float64:
        """
        Devuelve la fracción de potencia normalizada [0, 1] para una inserción dada de barras.

        Cuanto más insertadas estén las barras (B → 1), menos potencia produce el reactor.
        Con B = 0 el reactor está a tope; con B = 1 está prácticamente parado.
        El modelo de decaimiento es exponencial: P(B) = e^(-k·B).
        """
        B = np.clip(control_bars_insertion, 0.0, 1.0)
        return np.exp(-self.k * B)

    def compute_control_bars_insertion(self, power: np.float64) -> np.float64:
        """
        Hace el camino inverso: dado un nivel de potencia, dice cuánto hay que
        insertar las barras para conseguirlo.

        Es simplemente la función inversa de compute_power. Útil para las
        visualizaciones, donde queremos saber qué posición física de las barras
        corresponde a la potencia que está produciendo el reactor en cada momento.
        """
        p = np.clip(power, 1e-10, 1.0)
        return -np.log(p) / self.k
