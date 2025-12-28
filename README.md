# Pacman AI Project

A project exploring both classical AI algorithms and deep learning approaches to play Pacman using OpenAI Gymnasium.

## Overview

This project implements and compares various AI techniques for playing Pacman:
- **Classical AI**: Random agents, rule-based systems, search algorithms (A*, BFS, MCTS)
- **Deep Learning**: DQN, Double DQN, Dueling DQN, PPO, A3C
- **Hybrid Approaches**: Combining classical pathfinding with deep reinforcement learning

## Project Setup

### Prerequisites

- Anaconda or Miniconda (recommended)
  - Download from: https://docs.conda.io/en/latest/miniconda.html
- Python 3.10 will be installed automatically via conda

### Quick Setup (Recommended)

Run the automated setup script:

```bash
./setup.sh
```

This script will:
1. Check for conda installation
2. Create a conda environment named `pacman_ai` with Python 3.10
3. Install all required dependencies
4. Set up the project directory structure
5. Prepare the environment for development

### Manual Setup

If you prefer to set up manually:

```bash
# Create conda environment
conda create -n pacman_ai python=3.10 -y

# Activate conda environment
conda activate pacman_ai

# Install dependencies
pip install -r requirements.txt

# Create project directories
mkdir -p gamer/agents gamer/models gamer/utils gamer/training
mkdir -p notebooks configs saved_models results
```

### Verify Installation

After setup, test that everything is working:

```bash
conda activate pacman_ai  # Activate environment first
python test_setup.py
```

This will run a random agent playing Pacman for a few episodes to verify your installation.

## Project Structure

```
pacman_ai/
├── gamer/                  # Main package
│   ├── agents/            # Agent implementations
│   ├── models/            # Neural network models
│   ├── utils/             # Utility functions and wrappers
│   └── training/          # Training scripts
├── notebooks/             # Jupyter notebooks for experiments
├── configs/               # Configuration files
├── saved_models/          # Trained model checkpoints
├── results/               # Evaluation results and logs
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
├── test_setup.py         # Installation verification script
├── Plan.md               # Detailed implementation plan
└── README.md             # This file
```

## Dependencies

Key dependencies include:
- **gymnasium[atari]**: OpenAI Gymnasium with Atari environments
- **torch**: PyTorch for deep learning
- **numpy**: Numerical computing
- **matplotlib**: Visualization
- **opencv-python**: Image processing
- **tqdm**: Progress bars

See [requirements.txt](requirements.txt) for the complete list.

## Usage

### Running the Random Agent

```bash
conda activate pacman_ai
python -m gamer.agents.random_agent
```

## Plan and Current Status
The goal of this project is to play around with search algorithms and then finally DL algorithms to play pacman. For fun and learning!

- [x] Project setup and structure
- [x] Random agent implementation
- [ ] Rule-based agent
- [ ] Search algorithms (A*, BFS, MCTS)
- [ ] DQN implementation
- [ ] Advanced RL algorithms
- [ ] Hybrid approaches
- [ ] Evaluation framework

## Contributing

This is a learning and experimentation project. Feel free to:
- Try different hyperparameters
- Implement new algorithms
- Improve existing implementations
- Add visualization tools

## License

This project is for educational purposes.

## Acknowledgments

- OpenAI Gymnasium for the Atari environments
- The reinforcement learning community for various algorithm implementations
