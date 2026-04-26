# AICapstone
This repository relies mainly on IsaacSim, IsaacLab to perform simulations tasks

# "uv" as the main package manager
- uv run python to spawn python shell

# Workspace design
Workspaces organize large codebases by splitting them into multiple packages with common dependencies. Think: a FastAPI-based web application, alongside a series of libraries that are versioned and maintained as separate Python packages, all in the same Git repository.

In a workspace, each package defines its own pyproject.toml, but the workspace shares a single lockfile, ensuring that the workspace operates with a consistent set of dependencies.

As such, uv lock operates on the entire workspace at once, while uv run and uv sync operate on the workspace root by default, though both accept a --package argument, allowing you to run a command in a particular workspace member from any workspace directory.


## packages/umi
This package served as foundation UMI processing pipeline

## packages/simulator
This package served as an extension package on top of leisaac, it installs "leisaac" as it internal dependency
