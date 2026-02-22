import numpy as np

def extract_policy(Q, num_actions):
    """Extract a deterministic policy from a Q-table.

Selects the action with the highest value for each state.

Args:
    Q (dict): Mapping state -> np.array of action values.

Returns:
    dict: Mapping state -> optimal action (int).
"""

    policy = {}

    for state in Q.keys():
        policy[state] = int(np.argmax(Q[state]))

    return policy