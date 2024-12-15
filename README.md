
## Getting Started

### Setup via Dev Container

1. Ensure Docker and Visual Studio Code with the Remote - Containers extension are installed on your machine.
2. Open the project in Visual Studio Code and when prompted, reopen the project in a container. This will build the Docker container as defined and install all necessary dependencies.

As `mcr.microsoft.com/devcontainers/miniconda:1-3` image is being used, the `environment.yml` file will automatically be installed
3. `pip install -r dust3r/requirements.txt --user`


### Setup by Hand

If you prefer to set up the environment manually or are unable to use Docker, follow these steps:

1. Install Conda or Mamba as your Python package manager.
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
2. Use the script provided below to create and activate a new environment with the necessary dependencies.
```
#!/bin/bash

# Define the environment name
ENV_NAME=${1:-your-env}

# Create a new environment named $ENV_NAME
mamba create -n $ENV_NAME -y

# Add conda-forge channel and set channel priority
mamba config --add channels conda-forge
mamba config --set channel_priority strict

# Activate the newly created environment
source activate $ENV_NAME

# Install needed packages for development
pip install pytest pytest-watch black ruff isort jupyterlab_code_formatter jupyterlab jupyterlab-vim mypy -y
```

## Contributing

Contributions to improve the development workflow or update dependencies are welcome. Please refer to the [`README.md`](README.md) for guidelines on contributing to this repository.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

# Setup via Dev Container
This should be done on container start up, note here that as `mcr.microsoft.com/devcontainers/miniconda:1-3` image is being used, the `environment.yml` file will automatically be installed

# TODO:
Look at https://medium.com/@jamiekt/vscode-devcontainer-with-zsh-oh-my-zsh-and-agnoster-theme-8adf884ad9f6 for setting up zsh in dev container and remember bash history