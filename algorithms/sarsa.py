import numpy as np
from collections import defaultdict


def sarsa(env, num_episodes=10000, alpha=0.3, gamma=0.98, epsilon=0.1):
    """On-policy SARSA (State-Action-Reward-State-Action) for tabular control.
    
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
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        action = policy(Q, state, num_actions, epsilon)
        
        total_reward = 0
        steps = 0
        
        while True:
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = policy(Q, next_state, num_actions, epsilon)
            
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            
            Q[state][action] += alpha * td_error
            
            state = next_state
            action = next_action
            
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        stats["episode_rewards"].append(total_reward)
        stats["episode_lengths"].append(steps)
    
    return Q, stats