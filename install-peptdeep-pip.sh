#!/bin/bash
# This script installs peptdeep using pip without conda

# Default values
INSTALL_TYPE="stable"  # Options: stable, loose, gui, hla
INSTALL_MODE="user"    # Options: user, system, venv

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --type)
      INSTALL_TYPE="$2"
      shift 2
      ;;
    --mode)
      INSTALL_MODE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --type TYPE      Installation type: stable, loose, gui, hla (default: stable)"
      echo "  --mode MODE      Installation mode: user, system, venv (default: user)"
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
echo "  Installation type: $INSTALL_TYPE"
echo "  Installation mode: $INSTALL_MODE"

# Prepare installation string
if [ "$INSTALL_TYPE" = "loose" ]; then
  INSTALL_STRING=""
else
  INSTALL_STRING="[$INSTALL_TYPE]"
fi

# Install peptdeep based on mode
if [ "$INSTALL_MODE" = "user" ]; then
  echo "Installing peptdeep for current user..."
  pip install --user -e ".$INSTALL_STRING"
elif [ "$INSTALL_MODE" = "system" ]; then
  echo "Installing peptdeep system-wide (may require sudo)..."
  pip install -e ".$INSTALL_STRING"
elif [ "$INSTALL_MODE" = "venv" ]; then
  echo "Creating a virtual environment and installing peptdeep..."
  python -m venv peptdeep-venv
  source peptdeep-venv/bin/activate
  pip install -e ".$INSTALL_STRING"
  echo "Virtual environment created at: $(pwd)/peptdeep-venv"
  echo "To activate: source peptdeep-venv/bin/activate"
else
  echo "Unknown installation mode: $INSTALL_MODE"
  exit 1
fi

# Verify installation
echo "Verifying installation..."
peptdeep -v

echo ""
echo "Installation complete!"
echo "You can now run peptdeep using the 'peptdeep' command."