import numpy as np

try:
    import mdptoolbox
    _USE_MDPTOOLBOX = True
except ImportError:
    _USE_MDPTOOLBOX = False


class _ValueIteration:
    """
    Custom implementation of the Value Iteration algorithm, used as a fallback
    when the pymdptoolbox library is not available in the environment.

    This class mirrors the interface expected by the rest of the system
    (transitions, reward, discount, and a run() method), so that replacing it
    with the real library only requires changing the import logic at the top of
    this file.

    The algorithm iteratively updates state values until the maximum change
    between consecutive iterations falls below a convergence threshold (epsilon).
    """

    def __init__(self, transitions: np.ndarray, reward: np.ndarray,
                 discount: float, epsilon: float = 1e-6, max_iter: int = 1000):
        self.P = transitions        # Transition tensor of shape (A, S, S)
        self.discount = discount
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.n_states = transitions.shape[1]
        self.n_actions = transitions.shape[0]

        # Normalize the reward matrix to shape (S, A), regardless of input format.
        if reward.ndim == 2 and reward.shape == (self.n_states, self.n_actions):
            self.R = reward
        elif reward.ndim == 3:
            # Compute the expected reward R(s, a) as a weighted average over next states:
            # R(s, a) = sum_{s'} P(a, s, s') * R(a, s, s')
            self.R = np.einsum('aij,aij->ia', self.P, reward).T
        else:
            raise ValueError(f"Unexpected reward shape: {reward.shape}")

        self.V = np.zeros(self.n_states, dtype=np.float64)
        self.policy = np.zeros(self.n_states, dtype=np.int32)

    def run(self):
        """Runs Value Iteration until convergence or until the maximum number of iterations is reached."""
        for _ in range(self.max_iter):
            V_prev = self.V.copy()

            # Q(s, a) = R(s, a) + gamma * sum_{s'} P(a, s, s') * V(s')
            Q = self.R + self.discount * np.einsum('aij,j->ia', self.P, self.V)

            self.V = np.max(Q, axis=1)
            self.policy = np.argmax(Q, axis=1)

            # Stop early if the value function has converged
            if np.max(np.abs(self.V - V_prev)) < self.epsilon:
                break


class ControlModule:
    """
    MDP-based control system for a nuclear reactor.

    This module implements the core control logic: building the transition matrix P
    and the cost matrix C from the reactor's stochastic dynamics, solving the MDP
    via Value Iteration, and executing the full control loop over a demand curve.
    """

    @staticmethod
    def build_transition_matrix(probs: np.ndarray, n_states: int, n_actions: int) -> np.ndarray:
        """
        Builds the transition probability tensor P of shape (n_actions, n_states, n_states).

        Each action (decrease, maintain, increase) has three possible outcomes with
        associated probabilities. Boundary states (0 and n_states - 1) are handled
        by reflecting the probability mass back, so that the reactor never leaves
        the valid power range.

        Parameters
        ----------
        probs : np.ndarray
            Array of shape (3, 3) where each row corresponds to one action and each
            column to one of its three possible stochastic outcomes.
        n_states : int
            Total number of discrete power levels in the MDP.
        n_actions : int
            Number of available actions (always 3: decrease, maintain, increase).

        Returns
        -------
        np.ndarray
            Transition tensor P of shape (n_actions, n_states, n_states).
        """
        P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        # Action 0: Decrease — possible outcomes are -2, -1, or 0 levels
        offsets_decrease = [-2, -1, 0]
        # Action 1: Maintain — possible outcomes are -1, 0, or +1 levels
        offsets_maintain = [-1, 0, +1]
        # Action 2: Increase — possible outcomes are 0, +1, or +2 levels
        offsets_increase = [0, +1, +2]

        action_offsets = [offsets_decrease, offsets_maintain, offsets_increase]

        for a, offsets in enumerate(action_offsets):
            for s in range(n_states):
                for k, delta in enumerate(offsets):
                    # Clip the destination state to keep it within valid bounds
                    s_next = int(np.clip(s + delta, 0, n_states - 1))
                    P[a, s, s_next] += probs[a, k]

        return P

    @staticmethod
    def build_cost_matrix(demand_t: float, n_states: int, n_actions: int,
                          P: np.ndarray) -> np.ndarray:
        """
        Builds the cost matrix C of shape (n_states, n_actions) for a single control iteration.

        The cost of transitioning from state s to state s' via action a is defined as the
        absolute distance between the demand at time t and the power level of s'. Actions
        that actively move the reactor away from the target demand are penalized with a
        factor of 2, as specified in the assignment.

        Parameters
        ----------
        demand_t : float
            Current demand value at time step t, in the range [0, 1].
        n_states : int
            Total number of discrete power levels.
        n_actions : int
            Number of available actions.
        P : np.ndarray
            Precomputed transition tensor of shape (n_actions, n_states, n_states).

        Returns
        -------
        np.ndarray
            Cost matrix C of shape (n_states, n_actions).
        """
        # Represent each state as the lower bound of its power interval
        levels = np.arange(n_states, dtype=np.float64) / n_states

        C = np.zeros((n_states, n_actions), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                cost = 0.0
                for s_next in range(n_states):
                    if P[a, s, s_next] == 0.0:
                        continue

                    distance = abs(demand_t - levels[s_next])

                    # Penalize transitions that move away from the target demand
                    current_level = levels[s]
                    next_level = levels[s_next]

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
        Solves a single MDP control iteration for the current demand value.

        Builds the cost matrix C, constructs the MDP using the precomputed transition
        matrix P, and runs Value Iteration to obtain the optimal policy. Returns the
        optimal action for the current power state.

        Parameters
        ----------
        demand_t : float
            Current demand value at time step t.
        current_state : int
            Current discrete power level of the reactor (index in [0, n_states - 1]).
        P : np.ndarray
            Precomputed transition tensor of shape (n_actions, n_states, n_states).
        n_states : int
            Total number of discrete power levels.
        n_actions : int
            Number of available actions.
        gamma : float
            Discount factor for the MDP (typically close to 1).

        Returns
        -------
        int
            Index of the optimal action (0 = decrease, 1 = maintain, 2 = increase).
        """
        C = ControlModule.build_cost_matrix(demand_t, n_states, n_actions, P)

        # Convert cost to reward (Value Iteration maximizes reward, so we negate the cost)
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
        Executes the full control loop over an entire demand curve.

        For each time step, the optimal action is determined by solving the MDP,
        and the next reactor state is sampled stochastically according to the
        transition probabilities of the chosen action. This simulates the inherent
        uncertainty of reactor dynamics.

        Parameters
        ----------
        demand : np.ndarray
            Normalized demand time series of shape (T,), with values in [0, 1].
        probs : np.ndarray
            Array of shape (3, 3) with the stochastic transition probabilities for
            each action (decrease, maintain, increase).
        n_states : int
            Total number of discrete power levels.
        n_actions : int
            Number of available actions.
        gamma : float
            Discount factor for the MDP.

        Returns
        -------
        np.ndarray
            Array of shape (T,) with the normalized power response of the reactor
            at each time step.
        """
        # Build the transition matrix once, since it does not change over time
        P = ControlModule.build_transition_matrix(probs, n_states, n_actions)

        T = demand.shape[0]
        response = np.zeros(T, dtype=np.float64)

        # Start at the power level closest to the first demand point
        current_state = int(np.clip(demand[0] * n_states, 0, n_states - 1))

        # Offsets for each action outcome (same structure as in build_transition_matrix)
        action_offsets = [[-2, -1, 0], [-1, 0, +1], [0, +1, +2]]

        for t in range(T):
            demand_t = float(demand[t])

            # Determine the best action for the current state and demand
            action = ControlModule.solve_iteration(demand_t, current_state, P,
                                                   n_states, n_actions, gamma)

            # Sample the next state stochastically, mimicking the reactor's uncertainty
            offsets = action_offsets[action]
            outcome = np.random.choice(len(offsets), p=probs[action])
            delta = offsets[outcome]
            current_state = int(np.clip(current_state + delta, 0, n_states - 1))

            # Record the normalized power level as the response at time t
            response[t] = current_state / n_states

        return response
