# Making PeptDeep Executable

This guide explains how to make PeptDeep executable on your system without using conda.

## Option 1: Using the provided scripts

Two scripts have been created to help you install and run PeptDeep:

### 1. Installation Script

The `install-peptdeep-pip.sh` script installs PeptDeep using pip:

```bash
# Make the script executable (if not already)
chmod +x install-peptdeep-pip.sh

# Show help
./install-peptdeep-pip.sh --help

# Install with default settings (user installation)
./install-peptdeep-pip.sh

# Install with custom settings
./install-peptdeep-pip.sh --type gui --mode venv
```

Installation types:
- `stable`: Standard installation with fixed dependencies
- `loose`: Installation with flexible dependencies
- `gui`: Installation with GUI dependencies
- `hla`: Installation with HLA analysis dependencies

Installation modes:
- `user`: Install for current user only (uses `pip install --user`)
- `system`: Install system-wide (may require sudo)
- `venv`: Create a Python virtual environment and install there

### 2. Run Script

The `run-peptdeep.sh` script runs PeptDeep:

```bash
# Make the script executable (if not already)
chmod +x run-peptdeep.sh

# Run PeptDeep
./run-peptdeep.sh gui
./run-peptdeep.sh -v
./run-peptdeep.sh library settings.yaml
```

## Option 2: Manual Installation

You can also install PeptDeep manually:

```bash
# Install for current user
pip install --user peptdeep

# OR create a virtual environment
python -m venv peptdeep-venv
source peptdeep-venv/bin/activate
pip install peptdeep

# Run PeptDeep
peptdeep -v
peptdeep gui
```

## Option 3: Build a standalone executable

For advanced users, you can build a standalone executable using PyInstaller:

```bash
# Install PyInstaller
pip install pyinstaller

# Navigate to the PyInstaller directory
cd release/pyinstaller

# Build the executable
pyinstaller peptdeep.spec
```

This will create an executable in the `dist` directory that you can run without needing Python installed.

## Option 4: Create a symbolic link

If you've installed PeptDeep but the command isn't in your PATH, you can create a symbolic link:

```bash
# Find where peptdeep is installed
which peptdeep

# If not found, find the Python script
find /usr -name "peptdeep" 2>/dev/null
# OR
find ~/.local -name "peptdeep" 2>/dev/null

# Create a symbolic link (replace with actual path)
sudo ln -s /path/to/peptdeep /usr/local/bin/peptdeep