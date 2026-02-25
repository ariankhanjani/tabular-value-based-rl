import numpy as np
from collections import defaultdict


def sarsa(env, num_episodes=5000, alpha=0.3, gamma=0.98, epsilon=0.1):
    """On-policy SARSA with epsilon-greedy exploration.
Args:
    env: Gymnasium-like environment with discrete action space.
    num_episodes (int): Number of training episodes.
    alpha (float): Learning rate.
    gamma (float): Discount factor.
    epsilon (float): Exploration rate for epsilon-greedy policy.

Returns:
    Q (defaultdict): Mapping state -> np.array of action values.
    stats (dict): Episode statistics with keys:
        - "episode_rewards": list of total reward per episode
        - "episode_lengths": list of steps per episode
    """

    num_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(num_actions))

    stats = {"episode_rewards": [], "episode_lengths": []}

    def policy(state):
        if np.random.rand() < epsilon:
            return np.random.randint(num_actions)
        return np.argmax(Q[state])

    for _ in range(num_episodes):
        state, _ = env.reset()
        action = policy(state)

        total_reward = 0
        steps = 0
        done, truncated = False, False

        while not (done or truncated):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = policy(next_state)

            Q[state][action] += alpha * (
                reward + gamma * Q[next_state][next_action] - Q[state][action]
            )

            state, action = next_state, next_action
            total_reward += reward
            steps += 1

        stats["episode_rewards"].append(total_reward)
        stats["episode_lengths"].append(steps)

    return Q, stats