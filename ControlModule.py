# Import required dependencies
import numpy as np

# ---------------------------------------------------------------------------
# Lightweight Value Iteration solver compatible with the pymdptoolbox interface.
# This implementation mirrors the API of mdptoolbox.mdp.ValueIteration so that
# replacing it with the real library (once network access is available) only
# requires uncommenting two lines and removing this block.
# ---------------------------------------------------------------------------
try:
    import mdptoolbox
    _USE_MDPTOOLBOX = True
except ImportError:
    _USE_MDPTOOLBOX = False


class _ValueIteration:
    """
    Minimal implementation of the Value Iteration algorithm for finite MDPs.
    Matches the interface of mdptoolbox.mdp.ValueIteration.

    Parameters
    ----------
    transitions : np.ndarray, shape (A, S, S)
        Transition probability tensor. transitions[a, s, s'] is the probability
        of reaching state s' from state s under action a.
    reward : np.ndarray, shape (S, A)  —or—  (A, S, S)
        Reward (or negative cost) matrix. Interpreted as (S, A) if 2-D.
    discount : float
        Discount factor gamma in (0, 1].
    epsilon : float
        Convergence threshold (max-norm of the Bellman residual).
    max_iter : int
        Maximum number of iterations before stopping.
    """

    def __init__(self, transitions: np.ndarray, reward: np.ndarray,
                 discount: float, epsilon: float = 1e-6, max_iter: int = 1000):
        self.P        = transitions          # (A, S, S)
        self.discount = discount
        self.epsilon  = epsilon
        self.max_iter = max_iter
        self.n_states  = transitions.shape[1]
        self.n_actions = transitions.shape[0]

        # Normalise reward to shape (S, A)
        if reward.ndim == 2 and reward.shape == (self.n_states, self.n_actions):
            self.R = reward                  # (S, A)
        elif reward.ndim == 3:
            # Average over next states: R(s, a) = sum_{s'} P(a,s,s') * R(a,s,s')
            self.R = np.einsum('aij,aij->ia', self.P, reward).T  # (S, A)
        else:
            raise ValueError(f"Unexpected reward shape: {reward.shape}")

        self.V      = np.zeros(self.n_states, dtype=np.float64)
        self.policy = np.zeros(self.n_states, dtype=np.int32)

    def run(self):
        """ Execute Value Iteration until convergence or max_iter is reached. """
        for _ in range(self.max_iter):
            V_prev = self.V.copy()

            # Q(s, a) = R(s, a) + gamma * sum_{s'} P(a, s, s') * V(s')
            # Shape: (S, A)
            Q = self.R + self.discount * np.einsum('aij,j->ia', self.P, self.V)

            self.V      = np.max(Q,  axis=1)
            self.policy = np.argmax(Q, axis=1)

            # Check convergence
            if np.max(np.abs(self.V - V_prev)) < self.epsilon:
                break


class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(probs: np.ndarray, n_states: np.int32, n_actions: np.int32) -> np.ndarray:
        """
        Generates the transition probability tensor P of shape (A, S, S).

        P[a, s, s'] = probability of transitioning from state s to state s'
                      when action a is taken.

        Actions:
            0 — decrease  (d): intended outcome  -1 level;
                                side outcomes     -2 (undesired) and  0 (undesired).
            1 — maintain  (m): intended outcome   0 levels;
                                side outcomes     -1 (undesired) and +1 (undesired).
            2 — increase  (i): intended outcome  +1 level;
                                side outcomes      0 (undesired) and +2 (undesired).

        The probability tables follow the order given in the JSON files:
            decrease : [p(-2), p(-1), p(0)]
            maintain : [p(-1), p( 0), p(+1)]
            increase : [p( 0), p(+1), p(+2)]

        Boundary conditions: if a transition would take the reactor below level 0
        or above level 99, that probability mass is redirected to the boundary level
        (i.e., the reactor "clips" at the extremes instead of leaving the state space).

        Parameters
        ----------
        probs : np.ndarray, shape (3, 3)
            Row 0 — decrease probabilities [p(-2), p(-1), p(0)]
            Row 1 — maintain probabilities [p(-1), p(0),  p(+1)]
            Row 2 — increase probabilities [p(0),  p(+1), p(+2)]
        n_states  : int — number of power levels (100)
        n_actions : int — number of actions (3)

        Returns
        -------
        P : np.ndarray, shape (A, S, S)
        """
        P = np.zeros((n_actions, n_states, n_states), dtype=np.float64)

        # Offsets for each action: [undesired_low, desired, undesired_high]
        offsets = {
            0: [-2, -1,  0],   # decrease
            1: [-1,  0, +1],   # maintain
            2: [ 0, +1, +2],   # increase
        }

        for a in range(n_actions):
            p_low, p_mid, p_high = probs[a]
            delta_low, delta_mid, delta_high = offsets[a]

            for s in range(n_states):
                # Compute raw destination states
                raw = [s + delta_low, s + delta_mid, s + delta_high]
                p_vals = [p_low, p_mid, p_high]

                for prob, dest in zip(p_vals, raw):
                    # Clip to valid state range [0, n_states - 1]
                    dest_clipped = int(np.clip(dest, 0, n_states - 1))
                    P[a, s, dest_clipped] += prob

        return P

    @staticmethod
    def generate_R(demand_t: np.float64, n_states: np.int32, n_actions: np.int32,
                   P: np.ndarray) -> np.ndarray:
        """
        Generates the reward matrix R of shape (S, A) for a given demand point.

        The cost of reaching next state s' is the absolute distance |d_t - level(s')|
        where level(s') = s' / 100  (lower bound of each power interval).

        Actions that move the reactor *away* from the demand level incur a ×2 penalty.
        Because pymdptoolbox maximises reward, we negate the costs: R = -cost.

        Parameters
        ----------
        demand_t  : float — current demand value in [0, 1]
        n_states  : int
        n_actions : int
        P         : np.ndarray, shape (A, S, S) — transition tensor

        Returns
        -------
        R : np.ndarray, shape (S, A)
        """
        # Power level represented by each state (lower bound of interval)
        levels = np.arange(n_states, dtype=np.float64) / n_states  # shape (S,)

        # Base distances from demand to every level
        distances = np.abs(demand_t - levels)  # shape (S,)

        # Build R(S, A)
        R = np.zeros((n_states, n_actions), dtype=np.float64)

        for a in range(n_actions):
            for s in range(n_states):
                current_level = levels[s]

                # Compute expected cost over next states weighted by P(a, s, s')
                expected_cost = 0.0
                for s_next in range(n_states):
                    p_trans = P[a, s, s_next]
                    if p_trans == 0.0:
                        continue

                    next_level = levels[s_next]
                    base_cost  = distances[s_next]

                    # Penalise transitions that move away from demand
                    # Decrease action (a=0) moving below demand → penalty
                    # Increase action (a=2) moving above demand → penalty
                    moving_away = False
                    if a == 0 and next_level < demand_t and current_level >= demand_t:
                        moving_away = True
                    elif a == 0 and next_level < current_level and current_level < demand_t:
                        moving_away = True
                    elif a == 2 and next_level > demand_t and current_level <= demand_t:
                        moving_away = True
                    elif a == 2 and next_level > current_level and current_level > demand_t:
                        moving_away = True

                    cost = 2.0 * base_cost if moving_away else base_cost
                    expected_cost += p_trans * cost

                # Negate: Value Iteration maximises reward, we minimise cost
                R[s, a] = -expected_cost

        return R

    @staticmethod
    def control_iteration(demand_t: np.float64, current_state: np.int32,
                          P: np.ndarray, n_states: np.int32, n_actions: np.int32,
                          gamma: np.float64) -> np.int32:
        """
        Solves one control iteration using Value Iteration to determine the
        optimal action for the current reactor state given the current demand.

        Parameters
        ----------
        demand_t      : float — current demand value in [0, 1]
        current_state : int   — current power level (0–99)
        P             : np.ndarray — precomputed transition tensor (A, S, S)
        n_states      : int
        n_actions     : int
        gamma         : float — MDP discount factor

        Returns
        -------
        action : int — optimal action index (0=decrease, 1=maintain, 2=increase)
        """
        # Build cost/reward matrix for this demand point
        R = ControlModule.generate_R(demand_t=demand_t, n_states=n_states,
                                     n_actions=n_actions, P=P)

        # Solve the MDP with Value Iteration
        if _USE_MDPTOOLBOX:
            vi = mdptoolbox.mdp.ValueIteration(transitions=P, reward=R,
                                               discount=gamma)
        else:
            vi = _ValueIteration(transitions=P, reward=R, discount=gamma)

        vi.run()

        # Return the optimal action for the current state
        return np.int32(vi.policy[current_state])

    @staticmethod
    def control_loop(demand: np.ndarray,
                     probs: np.ndarray,
                     n_states: np.int32,
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """
        Executes the full control loop over all demand time-steps.

        For each time-step t:
            1. Solve the MDP for the current demand and state → optimal action.
            2. Simulate the stochastic reactor transition using the action probabilities.
            3. Update the current state and record the power level.

        Parameters
        ----------
        demand    : np.ndarray — power demand curve in [0, 1], shape (T,)
        probs     : np.ndarray — reactor probability tables, shape (3, 3)
        n_states  : int — number of power levels (100)
        n_actions : int — number of actions (3)
        gamma     : float — MDP discount factor

        Returns
        -------
        response : np.ndarray — reactor power output at each time-step, shape (T,)
        """
        T = demand.shape[0]
        response = np.zeros(T, dtype=np.float64)

        # Precompute the transition matrix P (constant across all iterations)
        P = ControlModule.generate_P(probs=probs, n_states=n_states, n_actions=n_actions)

        # Offsets for stochastic transitions per action [low, mid, high]
        offsets = {
            0: [-2, -1,  0],
            1: [-1,  0, +1],
            2: [ 0, +1, +2],
        }

        # Initialise the reactor at the power level closest to the first demand point
        current_state = np.int32(np.clip(int(demand[0] * n_states), 0, n_states - 1))

        for t in range(T):
            # Store current power level as a fraction in [0, 1]
            response[t] = current_state / n_states

            # Determine the optimal action for this demand point and state
            action = ControlModule.control_iteration(
                demand_t=demand[t],
                current_state=current_state,
                P=P,
                n_states=n_states,
                n_actions=n_actions,
                gamma=gamma
            )

            # Simulate the stochastic transition of the reactor
            p_action = probs[action]   # [p_low, p_mid, p_high]
            outcome_idx  = np.random.choice(a=3, p=p_action)
            delta        = offsets[action][outcome_idx]
            next_state   = int(np.clip(current_state + delta, 0, n_states - 1))

            current_state = np.int32(next_state)

        return response
