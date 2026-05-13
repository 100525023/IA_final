import numpy as np

try:
    import mdptoolbox
    _USE_MDPTOOLBOX = True
except ImportError:
    _USE_MDPTOOLBOX = False


class _ValueIteration:
    """
    A minimal Value Iteration implementation used as a fallback when pymdptoolbox
    is not available. It iteratively refines state values until the maximum change
    between iterations drops below epsilon, then reads the optimal policy from the
    resulting Q-values. Its interface mirrors mdptoolbox.mdp.ValueIteration so the
    rest of the code doesn't need to know which one is running.
    """

    def __init__(self, transitions: np.ndarray, reward: np.ndarray,
                 discount: float, epsilon: float = 1e-6, max_iter: int = 1000):
        self.P = transitions
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n_states = transitions.shape[1]
        self.n_actions = transitions.shape[0]

        # Normalize reward to shape (S, A) regardless of the input format
        if reward.ndim == 2 and reward.shape == (self.n_states, self.n_actions):
            self.R = reward
        elif reward.ndim == 3:
            self.R = np.einsum('aij,aij->ia', self.P, reward).T
        else:
            raise ValueError(f"Unexpected reward shape: {reward.shape}")

        self.V = np.zeros(self.n_states, dtype=np.float64)
        self.policy = np.zeros(self.n_states, dtype=np.int32)

    def run(self):
        """Runs the update loop until convergence or the iteration limit is reached."""
        for _ in range(self.max_iter):
            V_prev = self.V.copy()

            Q = self.R + self.discount * np.einsum('aij,j->ia', self.P, self.V)

            self.V = np.max(Q, axis=1)
            self.policy = np.argmax(Q, axis=1)

            if np.max(np.abs(self.V - V_prev)) < self.epsilon:
                break


class ControlModule:
    """
    MDP-based control system for a nuclear reactor. It builds the transition matrix
    and cost matrix from the reactor's stochastic dynamics, solves Value Iteration
    at each time step, and runs the full control loop over a demand curve.
    """

    @staticmethod
    def build_transition_matrix(probs: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
        """
        Builds the transition tensor P of shape (n_actions, n_states, n_states). Each action
        has three possible stochastic outcomes with offsets of [-2,-1,0] for decrease,
        [-1,0,+1] for maintain and [0,+1,+2] for increase. States at the boundaries absorb
        any probability that would fall outside the valid range.
        """
        P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        action_offsets = [[-2, -1, 0], [-1, 0, +1], [0, +1, +2]]

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
        Builds the cost matrix C of shape (n_states, n_actions) for the current demand value.
        The cost of each transition is the absolute distance between the demand and the next
        power level. Transitions that move the reactor further away from the target are
        penalized with a factor of 2.
        """
        levels = np.arange(n_states, dtype=np.float64) / n_states
        C = np.zeros((n_states, n_actions), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                cost = 0.0
                for s_next in range(n_states):
                    if P[a, s, s_next] == 0.0:
                        continue

                    distance = abs(demand_t - levels[s_next])

                    moving_away = (
                        (levels[s_next] > demand_t and levels[s_next] > levels[s]) or
                        (levels[s_next] < demand_t and levels[s_next] < levels[s])
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
        Solves one MDP iteration for the current demand and returns the best action for
        the current state. It builds the cost matrix, negates it into a reward matrix,
        runs Value Iteration and reads the policy at the current state.
        """
        C = ControlModule.build_cost_matrix(demand_t, n_states, n_actions, P)
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
        Runs the full control loop over the demand curve. At each time step it picks
        the best action via MDP, then samples the actual next state stochastically to
        simulate the reactor's inherent uncertainty. The transition matrix is built once
        at the start since it does not change over time.
        """
        P = ControlModule.build_transition_matrix(probs, n_states, n_actions)

        T = demand.shape[0]
        response = np.zeros(T, dtype=np.float64)
        current_state = int(np.clip(demand[0] * n_states, 0, n_states - 1))

        action_offsets = [[-2, -1, 0], [-1, 0, +1], [0, +1, +2]]

        for t in range(T):
            demand_t = float(demand[t])

            action = ControlModule.solve_iteration(demand_t, current_state, P,
                                                   n_states, n_actions, gamma)

            offsets = action_offsets[action]
            outcome = np.random.choice(len(offsets), p=probs[action])
            delta = offsets[outcome]
            current_state = int(np.clip(current_state + delta, 0, n_states - 1))

            response[t] = current_state / n_states

        return response
