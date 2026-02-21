import numpy as np
from collections import defaultdict


def sarsa(env, num_episode=10000, alpha=0.3, gamma=0.98, epsilon=0.2):
    """SARSA (on-policy TD control) algorithm.

    Args:
        env (gym.Env): Gymnasium-compatible environment.
        num_episode (int, optional): Number of training episodes.
        alpha (float, optional): Learning rate.
        gamma (float, optional): Discount factor.
        epsilon (float, optional): Exploration rate.

    Returns:
        tuple:
            Q (dict): Learned Q-table.
            rewards (list): Episode reward history.
    """
    
    num_actions = env.action_space.n
    Q = defaultdict(lambda: np.zeros(num_actions))
    rewards = []

    for _ in range(num_episode):
        state, _ = env.reset()
        state = int(state)

        action = policy(Q, state, num_actions, epsilon)
        done = False
        episode_reward = 0

        while not done:
            next_state, reward, done, truncated, _ = env.step(action)
            next_state = int(next_state)

            next_action = policy(Q, next_state, num_actions, epsilon)

            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]

            Q[state][action] += alpha * td_error

            state = next_state
            action = next_action
            episode_reward += reward

            if truncated:
                break

        rewards.append(episode_reward)

    return Q, rewards