#!/usr/bin/env python3
"""
View Gameplay Script

Plays a random agent episode and displays it using matplotlib.
This is more reliable than 'human' render mode on some systems.
"""

import gymnasium as gym
from gamer.agents.random_agent import RandomAgent
from gamer.utils.visualization import play_and_display


def main():
    print("=" * 60)
    print("Pacman Gameplay Viewer (Matplotlib)")
    print("=" * 60)
    print()
    print("Creating environment and agent...")

    # Create environment with rgb_array mode for matplotlib
    env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
    agent = RandomAgent(env, seed=42)

    print("Playing episode...")
    print("(A matplotlib window will appear showing the gameplay)")
    print()

    # Play and display
    results = play_and_display(env, agent, max_steps=500, interval=30)

    print()
    print("=" * 60)
    print("Episode Complete")
    print("=" * 60)
    print(f"Total Reward: {results['total_reward']:.1f}")
    print(f"Total Steps:  {results['steps']}")
    print(f"Frames:       {results['frames']}")
    print("=" * 60)

    env.close()


if __name__ == '__main__':
    main()
