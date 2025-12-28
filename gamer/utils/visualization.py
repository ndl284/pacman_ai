"""
Visualization utilities for rendering and displaying gameplay.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, clear_output
import numpy as np
import warnings


def play_and_display(env, agent, max_steps=500, interval=50):
    """
    Play an episode and display it using matplotlib animation.

    This works better than 'human' render mode on some systems (especially macOS).

    Args:
        env: Gymnasium environment (should use render_mode='rgb_array')
        agent: Agent with a select_action method
        max_steps: Maximum steps per episode
        interval: Milliseconds between frames (50 = 20 FPS)

    Returns:
        Dictionary with episode statistics
    """
    # Set render_fps in metadata if not already set (suppresses warning)
    if env.metadata.get('render_fps') is None:
        env.metadata['render_fps'] = 30

    frames = []
    observation, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    # Collect frames (suppress UserWarning about render_fps)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*render_fps.*')

        while not done and steps < max_steps:
            # Get current frame
            frame = env.render()
            frames.append(frame)

            # Take action
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

    # Display animation
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis('off')
    img = ax.imshow(frames[0])

    def update_frame(frame_idx):
        img.set_data(frames[frame_idx])
        ax.set_title(f'Step: {frame_idx + 1}/{len(frames)} | Reward: {total_reward:.1f}')
        return [img]

    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(frames),
        interval=interval, blit=False, repeat=False
    )

    # Calculate total duration in seconds
    duration_seconds = (len(frames) * interval) / 1000.0

    plt.tight_layout()
    plt.show(block=False)

    # Wait for animation to complete + 5 seconds
    plt.pause(duration_seconds + 5.0)
    plt.close(fig)

    return {
        'total_reward': total_reward,
        'steps': steps,
        'frames': len(frames)
    }


def play_episode_notebook(env, agent, max_steps=500):
    """
    Play an episode with live updates in Jupyter notebook.

    Args:
        env: Gymnasium environment (should use render_mode='rgb_array')
        agent: Agent with a select_action method
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with episode statistics
    """
    # Set render_fps in metadata if not already set (suppresses warning)
    if env.metadata.get('render_fps') is None:
        env.metadata['render_fps'] = 30

    observation, info = env.reset()
    total_reward = 0
    steps = 0
    done = False

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')

    # Suppress render_fps warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, message='.*render_fps.*')

        while not done and steps < max_steps:
            # Render and display current frame
            frame = env.render()

            clear_output(wait=True)
            ax.clear()
            ax.imshow(frame)
            ax.set_title(f'Step: {steps} | Total Reward: {total_reward:.1f}')
            ax.axis('off')
            display(fig)

            # Take action
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            steps += 1
            done = terminated or truncated

    plt.close(fig)

    return {
        'total_reward': total_reward,
        'steps': steps
    }
