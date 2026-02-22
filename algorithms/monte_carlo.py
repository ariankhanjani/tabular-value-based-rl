import numpy as np
from collections import defaultdict


def monte_carlo(env, num_episodes=10000, gamma=0.98, epsilon=0.1):
    """First-visit Monte Carlo control with epsilon-greedy policy.

Args:
    env: Gymnasium-like environment with discrete action space.
    num_episodes (int): Number of training episodes.
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
    returns = defaultdict(lambda: np.zeros(num_actions))
    counts = defaultdict(lambda: np.zeros(num_actions))
    
    stats = {"episode_rewards": [], "episode_lengths": []}
    
    for _ in range(num_episodes):
        episode = []
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        
        # Generate episode
        while True:
            action = policy(Q, state, num_actions, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            
            episode.append((state, action, reward))
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done or truncated:
                break
        
        # First-visit MC update
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                
                returns[state][action] += G
                counts[state][action] += 1
                
                Q[state][action] = returns[state][action] / counts[state][action]
        
        stats["episode_rewards"].append(total_reward)
        stats["episode_lengths"].append(steps)
    
    return Q, stats