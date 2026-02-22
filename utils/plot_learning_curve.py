import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(stats, window=100, label=None):
    """Plot smoothed episode rewards over time.

Applies a moving average to episode rewards for better visualization
of learning trends.

Args:
    stats (dict): Contains "episode_rewards" (list of rewards per episode).
    window (int): Smoothing window size for moving average.
    label (str, optional): Label for the plot (used in legend).
"""
    
    rewards = np.array(stats["episode_rewards"])
    
    # Moving average for smoothing
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    else:
        smoothed = rewards
    
    plt.plot(smoothed, label=label)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.title("Learning Curve")