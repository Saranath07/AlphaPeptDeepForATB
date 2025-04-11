#!/bin/bash
# This script installs peptdeep in a conda environment

# Default values
ENV_NAME="peptdeep"
PYTHON_VERSION="3.9"
INSTALL_TYPE="stable"  # Options: stable, loose, gui, hla

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
      ;;
    --type)
      INSTALL_TYPE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --env NAME       Conda environment name (default: peptdeep)"
      echo "  --python VERSION Python version (default: 3.9)"
      echo "  --type TYPE      Installation type: stable, loose, gui, hla (default: stable)"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Installing peptdeep with the following configuration:"
echo "  Conda environment: $ENV_NAME"
echo "  Python version: $PYTHON_VERSION"
echo "  Installation type: $INSTALL_TYPE"

# Check if conda is available
if ! command -v conda &> /dev/null; then
  echo "Error: conda is not installed or not in PATH"
  exit 1
fi

# Create conda environment if it doesn't exist
if ! conda env list | grep -q "^$ENV_NAME "; then
  echo "Creating conda environment: $ENV_NAME"
  conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
else
  echo "Conda environment $ENV_NAME already exists"
fi

# Prepare installation string
if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[$INSTALL_TYPE]"
fi

# Install peptdeep
echo "Installing peptdeep$INSTALL_STRING in environment $ENV_NAME"
conda run -n "$ENV_NAME" --no-capture-output pip install -e ".$INSTALL_STRING"

# Verify installation
echo "Verifying installation..."
conda run -n "$ENV_NAME" --no-capture-output peptdeep -v

echo ""
echo "Installation complete!"
echo "To use peptdeep, either:"
echo "1. Activate the conda environment: conda activate $ENV_NAME"
echo "   Then run: peptdeep [command]"
echo "2. Use the peptdeep-run.sh script: ./peptdeep-run.sh $ENV_NAME [command]"