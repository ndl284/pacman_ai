#!/usr/bin/env python3
"""
Setup Verification Script

This script tests that all dependencies are properly installed
and the environment is correctly configured.
"""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"✗ {package_name or module_name}: {e}")
        return False


def main():
    print("=" * 60)
    print("Pacman AI - Setup Verification")
    print("=" * 60)
    print()

    # Check Python version
    print("Python Version:")
    print(f"  {sys.version}")
    print()

    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required!")
        return False

    # Check required packages
    print("Checking Dependencies:")
    print("-" * 60)

    all_ok = True

    # Core dependencies
    all_ok &= check_import("gymnasium", "gymnasium")
    all_ok &= check_import("numpy", "numpy")
    all_ok &= check_import("torch", "PyTorch")

    # Atari support
    all_ok &= check_import("ale_py", "ale-py (Atari)")

    # Visualization
    all_ok &= check_import("matplotlib", "matplotlib")
    all_ok &= check_import("cv2", "opencv-python")

    # Utilities
    all_ok &= check_import("tqdm", "tqdm")
    all_ok &= check_import("yaml", "pyyaml")
    all_ok &= check_import("PIL", "pillow")

    print()

    if not all_ok:
        print("=" * 60)
        print("SETUP INCOMPLETE: Some dependencies are missing!")
        print("Please run: ./setup.sh")
        print("=" * 60)
        return False

    # Test Gymnasium environment
    print("Testing Gymnasium Environment:")
    print("-" * 60)

    try:
        import gymnasium as gym

        print("Creating ALE/MsPacman-v5 environment...")
        env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')

        print(f"✓ Environment created successfully")
        print(f"  Action Space: {env.action_space}")
        print(f"  Observation Space: {env.observation_space}")

        print("\nTesting environment reset...")
        observation, info = env.reset(seed=42)
        print(f"✓ Environment reset successful")
        print(f"  Observation shape: {observation.shape}")

        print("\nTesting single step...")
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Action taken: {action}")
        print(f"  Reward: {reward}")

        env.close()
        print()

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        print()
        return False

    # Test Random Agent with Visualization
    print("Testing Random Agent with Visualization:")
    print("-" * 60)

    try:
        from gamer.agents.random_agent import RandomAgent
        from gamer.utils.visualization import play_and_display

        env = gym.make('ALE/MsPacman-v5', render_mode='rgb_array')
        agent = RandomAgent(env, seed=42)

        print("Running 1 test episode with matplotlib visualization...")
        print("(A matplotlib window will show the gameplay animation)")
        print("Playing until Pacman loses all lives...")
        results = play_and_display(env, agent, max_steps=10000, interval=30)

        print(f"\n✓ Random Agent working correctly")
        print(f"  Total Reward: {results['total_reward']:.2f}")
        print(f"  Total Steps: {results['steps']}")

        env.close()

    except Exception as e:
        print(f"✗ Random Agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # All tests passed
    print()
    print("=" * 60)
    print("SUCCESS: All tests passed!")
    print("=" * 60)
    print()
    print("Your environment is ready. You can now:")
    print("  - Run the random agent: python -m gamer.agents.random_agent")
    print("  - Start implementing other agents")
    print("  - Begin training deep learning models")
    print()

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
