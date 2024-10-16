# Data Science Development Workflow Setup

This repository is designed to streamline the setup of a data science development environment, making it easy for data scientists to get up and running with a comprehensive suite of tools and libraries. Leveraging the power of containerization and dependency management, this setup ensures a consistent and reproducible development environment across different machines and platforms.

## Features

- **Containerized Development Environment**: Utilizes Docker and the Visual Studio Code Remote - Containers extension to create a consistent development environment inside a container. This approach isolates the development environment from the host system, reducing the "works on my machine" problem. The container is defined in the [`.devcontainer/Dockerfile`](.devcontainer/Dockerfile) and configured through [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json).

- **Pre-configured Python Environment**: The Python environment is pre-configured with a set of dependencies specified in [`environment.yml`](environment.yml), making it easy to get started with data science projects. This includes popular libraries and tools such as JupyterLab, pytest, black for code formatting, and mypy for type checking.

- **Automated Dependency Updates**: With Dependabot configuration in [`.github/dependabot.yml`](.github/dependabot.yml), the repository is set up to automatically receive updates for the development container and other dependencies, ensuring that the development environment stays up-to-date with the latest versions.

- **Pre-commit Hooks**: Utilizes pre-commit hooks configured in [`.pre-commit-config.yaml`](.pre-commit-config.yaml) to ensure code quality and consistency. This includes running linters and formatters on commit, helping to maintain a clean codebase.

- **Testing Support**: The setup includes pytest for running tests, making it easy to maintain high-quality code through automated testing. Test files can be placed in the [`tests/`](tests/) directory.

## Getting Started

### Setup via Dev Container

1. Ensure Docker and Visual Studio Code with the Remote - Containers extension are installed on your machine.
2. Open the project in Visual Studio Code and when prompted, reopen the project in a container. This will build the Docker container as defined and install all necessary dependencies.

As `mcr.microsoft.com/devcontainers/miniconda:1-3` image is being used, the `environment.yml` file will automatically be installed


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