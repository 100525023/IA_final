import numpy as np

try:
    import mdptoolbox
    _USE_MDPTOOLBOX = True
except ImportError:
    _USE_MDPTOOLBOX = False


class _ValueIteration:
    """
    Implementación propia del algoritmo de Iteración de Valor, usada como fallback
    cuando la librería pymdptoolbox no está instalada en el entorno.

    El algoritmo funciona de forma iterativa: parte de una estimación inicial de
    cero para todos los estados y va refinando esa estimación hasta que los cambios
    entre una iteración y la siguiente son tan pequeños que podemos decir que ha
    convergido (por debajo de epsilon).

    La interfaz (transitions, reward, discount y run()) es idéntica a la de
    mdptoolbox.mdp.ValueIteration, así que intercambiarlos es transparente para
    el resto del código.
    """

    def __init__(self, transitions: np.ndarray, reward: np.ndarray,
                 discount: float, epsilon: float = 1e-6, max_iter: int = 1000):
        self.P = transitions        # Tensor de transiciones de forma (A, S, S)
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n_states = transitions.shape[1]
        self.n_actions = transitions.shape[0]

        # Normalizamos la matriz de recompensa a forma (S, A) independientemente
        # del formato en que llegue.
        if reward.ndim == 2 and reward.shape == (self.n_states, self.n_actions):
            self.R = reward
        elif reward.ndim == 3:
            # R(s, a) = suma sobre s' de P(a, s, s') * R(a, s, s')
            self.R = np.einsum('aij,aij->ia', self.P, reward).T
        else:
            raise ValueError(f"Unexpected reward shape: {reward.shape}")

        self.V = np.zeros(self.n_states, dtype=np.float64)
        self.policy = np.zeros(self.n_states, dtype=np.int32)

    def run(self):
        """
        Ejecuta la Iteración de Valor hasta que converge o se alcanza el límite
        de iteraciones.

        En cada paso calcula los valores Q(s, a) para todas las combinaciones
        de estado y acción, y se queda con el máximo. Cuando la función de valor
        apenas cambia entre dos iteraciones, para.
        """
        for _ in range(self.max_iter):
            V_prev = self.V.copy()

            # Q(s, a) = R(s, a) + gamma * suma_{s'} P(a, s, s') * V(s')
            Q = self.R + self.discount * np.einsum('aij,j->ia', self.P, self.V)

            self.V = np.max(Q, axis=1)
            self.policy = np.argmax(Q, axis=1)

            # Paramos si la función de valor ha convergido
            if np.max(np.abs(self.V - V_prev)) < self.epsilon:
                break


class ControlModule:
    """
    Sistema de control basado en MDP para un reactor nuclear.

    Aquí vive la lógica central de control: construir la matriz de transiciones P
    y la matriz de costes C a partir de la dinámica estocástica del reactor, resolver
    el MDP mediante Iteración de Valor, y ejecutar el bucle de control completo
    sobre una curva de demanda.

    Todo es estático porque no necesitamos estado propio: cada llamada recibe
    todo lo que necesita como argumento.
    """

    @staticmethod
    def build_transition_matrix(probs: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
        """
        Construye el tensor de probabilidades de transición P con forma (n_actions, n_states, n_states).

        Cada acción (bajar, mantener, subir) tiene tres posibles resultados estocásticos,
        con sus correspondientes probabilidades. Cuando el reactor está en los extremos
        (estado 0 o estado n_states-1), los saltos que se saldrían del rango se acumulan
        en el estado frontera, así el reactor nunca sale del espacio válido de potencias.

        Los offsets por acción son:
        - Bajar:     -2, -1, 0
        - Mantener:  -1,  0, +1
        - Subir:      0, +1, +2
        """
        P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        offsets_decrease = [-2, -1, 0]
        offsets_maintain = [-1, 0, +1]
        offsets_increase = [0, +1, +2]

        action_offsets = [offsets_decrease, offsets_maintain, offsets_increase]

        for a, offsets in enumerate(action_offsets):
            for s in range(n_states):
                for k, delta in enumerate(offsets):
                    s_next = int(np.clip(s + delta, 0, n_states - 1))
                    P[a, s, s_next] += probs[a, k]

        return P

    @staticmethod
    def build_cost_matrix(demand_t: float, n_states: int, n_actions: int,
                          P: np.ndarray) -> np.ndarray:
        """
        Construye la matriz de costes C con forma (n_states, n_actions) para un instante dado.

        El coste de ir del estado s al estado s' mediante la acción a es la distancia
        absoluta entre la demanda en ese momento y el nivel de potencia de s'. Si
        además la transición nos aleja de la demanda (en lugar de acercarnos a ella),
        ese coste se multiplica por 2 como penalización extra.

        De este modo el MDP aprende a preferir acciones que acerquen el reactor
        a lo que se le está pidiendo, y a evitar las que van en sentido contrario.
        """
        # Cada estado representa el límite inferior de su intervalo de potencia
        levels = np.arange(n_states, dtype=np.float64) / n_states

        C = np.zeros((n_states, n_actions), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                cost = 0.0
                for s_next in range(n_states):
                    if P[a, s, s_next] == 0.0:
                        continue

                    distance = abs(demand_t - levels[s_next])

                    current_level = levels[s]
                    next_level = levels[s_next]

                    # ¿Nos estamos alejando de la demanda?
                    moving_away = (
                        (next_level > demand_t and next_level > current_level) or
                        (next_level < demand_t and next_level < current_level)
                    )

                    if moving_away:
                        distance *= 2.0

                    cost += P[a, s, s_next] * distance

                C[s, a] = cost

        return C

    @staticmethod
    def solve_iteration(demand_t: float, current_state: int, P: np.ndarray,
                        n_states: int, n_actions: int, gamma: float) -> int:
        """
        Resuelve una iteración del MDP para la demanda actual y devuelve la mejor acción.

        El proceso es: construimos la matriz de costes para este instante de tiempo,
        la convertimos en recompensa (negamos el coste, porque el MDP maximiza),
        resolvemos con Iteración de Valor y consultamos qué acción recomienda
        la política resultante para el estado actual del reactor.
        """
        C = ControlModule.build_cost_matrix(demand_t, n_states, n_actions, P)

        # Negamos el coste para convertirlo en recompensa (VI maximiza, nosotros queremos minimizar coste)
        R = -C

        if _USE_MDPTOOLBOX:
            vi = mdptoolbox.mdp.ValueIteration(P, R, gamma)
        else:
            vi = _ValueIteration(P, R, gamma)

        vi.run()

        return int(vi.policy[current_state])

    @staticmethod
    def control_loop(demand: np.ndarray, probs: np.ndarray,
                     n_states: int, n_actions: int, gamma: float) -> np.ndarray:
        """
        Ejecuta el bucle de control completo sobre toda la curva de demanda.

        En cada paso de tiempo:
        1. Calculamos la mejor acción para la demanda y estado actuales.
        2. Muestreamos el siguiente estado de forma estocástica (el reactor no
           es perfectamente predecible, así que el resultado real puede diferir
           del deseado según las probabilidades del reactor).
        3. Registramos la potencia normalizada resultante.

        La matriz de transición P se construye una sola vez al principio, ya que
        no cambia durante la simulación.

        Devuelve un array con la respuesta de potencia del reactor en cada instante.
        """
        # Construimos P una sola vez: es fijo para todo el experimento
        P = ControlModule.build_transition_matrix(probs, n_states, n_actions)

        T = demand.shape[0]
        response = np.zeros(T, dtype=np.float64)

        # Arrancamos en el nivel de potencia más cercano al primer punto de demanda
        current_state = int(np.clip(demand[0] * n_states, 0, n_states - 1))

        action_offsets = [[-2, -1, 0], [-1, 0, +1], [0, +1, +2]]

        for t in range(T):
            demand_t = float(demand[t])

            # Decidimos la mejor acción para este momento
            action = ControlModule.solve_iteration(demand_t, current_state, P,
                                                   n_states, n_actions, gamma)

            # Simulamos la respuesta real del reactor: puede que no salga exactamente
            # lo que queríamos, según las probabilidades estocásticas
            offsets = action_offsets[action]
            outcome = np.random.choice(len(offsets), p=probs[action])
            delta = offsets[outcome]
            current_state = int(np.clip(current_state + delta, 0, n_states - 1))

            # Guardamos la potencia normalizada de este instante
            response[t] = current_state / n_states

        return response
