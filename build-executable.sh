#!/bin/bash
# This script builds a standalone executable for peptdeep using PyInstaller

# Check if PyInstaller is installed
if ! pip show pyinstaller &> /dev/null; then
  echo "PyInstaller is not installed. Installing now..."
  pip install pyinstaller
fi

# Navigate to the PyInstaller directory
cd release/pyinstaller || {
  echo "Error: PyInstaller directory not found at release/pyinstaller"
  exit 1
}

echo "Building peptdeep executable..."
pyinstaller peptdeep.spec

# Check if build was successful
if [ -d "dist" ]; then
  echo "Build successful!"
  echo "Executable created at: $(pwd)/dist"
  
  # Determine platform-specific executable location
  if [ "$(uname)" == "Darwin" ]; then
    # macOS
    EXEC_PATH="dist/peptdeep_gui/peptdeep_gui"
  elif [ "$(uname)" == "Linux" ]; then
    # Linux
    EXEC_PATH="dist/peptdeep"
  else
    # Windows (assuming)
    EXEC_PATH="dist/peptdeep/peptdeep.exe"
  fi
  
  # Check if executable exists
  if [ -f "$EXEC_PATH" ]; then
    echo "You can run the executable with: $EXEC_PATH"
  else
    echo "Executable was built but not found at expected location."
    echo "Check the dist/ directory for the executable."
  fi
else
  echo "Build failed. Check the output for errors."
  exit 1
fi