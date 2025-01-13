#!/bin/bash

# Execute with `source create_env.sh` (but not `.create_env.sh`).

# Error message "/bin/bash^M: bad interpreter:" is fixed with:
# git config --global core.autocrlf input
# sed -i -e 's/\r$//' create_env.sh

# The first line above ensures that CRLF & LF changes are both
# handled correctly (rather than auto-replaced by Git).
# The second line above applies the fix to the file.

# Create virtual environment directory:
repo_dir=$(git rev-parse --show-toplevel)
path_to_all_environments=$repo_dir"/environments"
path_to_this_environment=$path_to_all_environments"/env_nucls"
path_to_env_requirements=$repo_dir"/requirements/"

mkdir -p $path_to_all_environments

# Create virtual environment (if it doesn't exist) & activate it:
if [ ! -d $path_to_this_environment ]; then
    python3 -m venv $path_to_this_environment
fi
source $path_to_this_environment"/bin/activate"

# Install dependencies:
pip install --upgrade pip
pip install -r $path_to_env_requirements"requirements.txt"


# Error message "pg_config executable not found" is fixed with:
# sudo apt install libpq-dev python3-dev

# Update PYTHONPATH to add repo root:
export PYTHONPATH=$PYTHONPATH:$repo_dir