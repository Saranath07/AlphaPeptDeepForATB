#!/bin/bash
# This script runs peptdeep by activating the conda environment
# and executing the peptdeep CLI with all arguments passed to this script

# Check if conda environment name is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <conda_env_name> [peptdeep_arguments]"
  echo "Example: $0 peptdeep gui"
  exit 1
fi

CONDA_ENV=$1
shift  # Remove the first argument (conda env name)

# Activate conda environment and run peptdeep with remaining arguments
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $CONDA_ENV

# Run peptdeep with all remaining arguments
peptdeep "$@"

# If peptdeep command is not found, try running through Python module
if [ $? -ne 0 ]; then
  echo "Trying to run peptdeep as a Python module..."
  python -m peptdeep.cli "$@"
fi