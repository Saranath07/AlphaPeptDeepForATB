#!/bin/bash
# This script runs peptdeep without requiring conda

# Run peptdeep with all arguments passed to this script
peptdeep "$@"

# If peptdeep command is not found, try running through Python module
if [ $? -ne 0 ]; then
  echo "Trying to run peptdeep as a Python module..."
  python -m peptdeep.cli "$@"
fi