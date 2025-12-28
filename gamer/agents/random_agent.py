"""
Random Agent for Pacman

This agent selects random actions at each timestep.
It serves as a baseline to compare more sophisticated agents against.
"""

import gymnasium as gym
import numpy as np
import time
from typing import Optional


class RandomAgent:
    """
    A simple agent that takes random actions in the environment.

    This agent is useful for:
    - Establishing baseline performance metrics
    - Testing environment setup
    - Understanding the action space
    """

    def __init__(self, env: gym.Env, seed: Optional[int] = None):
        """
        Initialize the Random Agent.

        Args:
            env: The Gymnasium environment
            seed: Random seed for reproducibility
        """
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)

    def select_action(self, observation) -> int:
        """
        Select a random action from the action space.

        Args:
            observation: Current observation from the environment (unused)

        Returns:
            A random valid action
        """
        return self.action_space.sample()

    def play_episode(self, render: bool = False, max_steps: Optional[int] = None) -> dict:
        """
        Play a single episode using random actions.

        Args:
            render: Whether to render the environment
            max_steps: Maximum number of steps per episode

        Returns:
            Dictionary containing episode statistics
        """
        observation, info = self.env.reset()
        total_reward = 0
        steps = 0
        done = False

        while not done:
            action = self.select_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)

            # Add small delay for human viewing (only when rendering)
            if render:
                time.sleep(0.02)  # 20ms delay to make gameplay visible

            total_reward += reward
            steps += 1
            done = terminated or truncated

            if max_steps and steps >= max_steps:
                break

        return {
            'total_reward': total_reward,
            'steps': steps,
            'terminated': terminated,
            'truncated': truncated
        }

    def evaluate(self, num_episodes: int = 10, render: bool = False,
                 max_steps: Optional[int] = None) -> dict:
        """
        Evaluate the agent over multiple episodes.

        Args:
            num_episodes: Number of episodes to run
            render: Whether to render the environment
            max_steps: Maximum steps per episode

        Returns:
            Dictionary containing evaluation statistics
        """
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            stats = self.play_episode(render=render, max_steps=max_steps)
            episode_rewards.append(stats['total_reward'])
            episode_lengths.append(stats['steps'])

            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Reward = {stats['total_reward']:.1f}, "
                  f"Steps = {stats['steps']}")

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_steps': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }


def main():
    """
    Main function to run the random agent.
    """
    print("=" * 60)
    print("Random Agent - Pacman AI")
    print("=" * 60)
    print()

    # Create the Pacman environment with visual rendering
    env = gym.make('ALE/MsPacman-v5', render_mode='human')

    print(f"Environment: {env.spec.id}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print()
    print("NOTE: A game window should appear showing Pacman gameplay")
    print()

    # Create and evaluate the random agent
    agent = RandomAgent(env, seed=42)

    print("Running evaluation (5 episodes)...")
    print("-" * 60)

    results = agent.evaluate(num_episodes=5, render=True)

    print()
    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"Min Reward:    {results['min_reward']:.2f}")
    print(f"Max Reward:    {results['max_reward']:.2f}")
    print(f"Mean Steps:    {results['mean_steps']:.2f}")
    print("=" * 60)

    env.close()


if __name__ == '__main__':
    main()
