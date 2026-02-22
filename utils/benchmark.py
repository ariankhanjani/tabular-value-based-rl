import time
import numpy as np
from tabulate import tabulate

def evaluate_algorithm(algo_fn, env, name, num_episodes=5000):
    """Run a training algorithm and compute summary performance metrics.

Measures execution time and aggregates episode statistics such as
average and standard deviation of episode lengths.

Args:
    algo_fn (callable): RL algorithm function returning (Q, stats).
    env: Gymnasium-like environment.
    name (str): Name of the algorithm (for reporting).
    num_episodes (int): Number of training episodes.

Returns:
    dict: Summary metrics including algorithm name, episode count,
    average episode length, standard deviation, and total runtime.
"""

    start_time = time.time()
    
    Q, stats = algo_fn(env, num_episodes=num_episodes)
    
    end_time = time.time()
    
    episode_lengths = np.array(stats["episode_lengths"])
    
    result = {
        "Algorithm": name,
        "Episodes": num_episodes,
        "Avg Length": np.mean(episode_lengths),
        "Std Length": np.std(episode_lengths),
        "Time (s)": end_time - start_time
    }
    
    return result