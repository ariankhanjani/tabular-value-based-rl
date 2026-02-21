import numpy as np

def policy(Q, state, num_actions, epsilon=0.2):
    """Epsilon-greedy policy.
    
    Args:
        Q (dict): State-action value table mapping states to action-value arrays.
        state (hashable): Current state.
        num_actions (int): Number of available actions.
        epsilon (float, optional): Exploration probability. Defaults to 0.2.

    Returns:
        int: Selected action index.
    """
    rng = np.random.default_rng()

    if rng.random() < epsilon:
        return rng.integers(num_actions)
    else:
        return np.argmax(Q[state])