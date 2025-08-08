# NeurIPS_Polymer_2025
Development project for the kaggle competition : NeurIPS - Open Polymer Prediction

## Installation

This project is designed to be used inside a [devcontainer](https://containers.dev/) for reproducible development environments. The devcontainer includes Python, Node.js, and other essential tools pre-installed.

We use [`uv`](https://github.com/astral-sh/uv) as the Python package manager for fast and reliable dependency management. All commands below assume you are working inside the devcontainer.

### Python Environment Setup

Dependency installation is managed by `uv`. The command `uv sync` should have been run automatically when the devcontainer started. If dependencies are missing, you can manually run:

```
uv sync
```

If PyTorch was not installed, we install it for CUDA 12.1 and up with the following command:

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

To install PyTorch Geometric and related packages, use:

```
uv pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

These commands will install GPU-enabled versions of PyTorch and the PyTorch Geometric ecosystem.

## Usage

1. Open the project in VS Code and connect to the devcontainer.  
   For a tutorial on using devcontainers, see [VS Code Dev Containers Guide](https://code.visualstudio.com/docs/devcontainers/containers).
2. Install dependencies as shown above.
3. Run your experiments or scripts using:
   ```
   uv run <script.py>
   ```
4. To add more dependencies, use:
   ```
   uv add <package>
   ```
