#!/bin/bash

# Pacman AI Project Setup Script
# This script sets up the entire development environment using Conda

set -e  # Exit on error

echo "========================================="
echo "Pacman AI Project Setup"
echo "========================================="
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda from:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Conda version:"
conda --version
echo ""

# Environment name
ENV_NAME="pacman_ai"

echo "Step 1: Checking for existing conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Using existing environment."
    fi
fi

echo ""
echo "Step 2: Creating conda environment..."
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n ${ENV_NAME} python=3.10 -y
    echo "Conda environment '${ENV_NAME}' created successfully."
fi

echo ""
echo "Step 3: Activating conda environment..."
conda activate ${ENV_NAME}

echo ""
echo "Step 4: Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Step 5: Creating project directories..."
mkdir -p gamer/agents
mkdir -p gamer/models
mkdir -p gamer/utils
mkdir -p gamer/training
mkdir -p notebooks
mkdir -p configs
mkdir -p saved_models
mkdir -p results

# Create __init__.py files if they don't exist
touch gamer/agents/__init__.py
touch gamer/models/__init__.py
touch gamer/utils/__init__.py
touch gamer/training/__init__.py

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "To activate the conda environment, run:"
echo "  conda activate pacman_ai"
echo ""
echo "To test the installation, run:"
echo "  python test_setup.py"
echo ""
echo "To deactivate the environment when done:"
echo "  conda deactivate"
echo ""
