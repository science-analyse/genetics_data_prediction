#!/bin/bash

# Render.com build script for Gene Expression Classifier

set -e  # Exit on error

echo "=================================="
echo "Building Gene Expression Classifier"
echo "=================================="

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo ""
    echo "WARNING: Models directory is empty!"
    echo "Models must be committed to git or mounted as a disk."
    echo ""
    echo "To commit models:"
    echo "  1. Run notebooks/analyse.ipynb locally"
    echo "  2. git add models/"
    echo "  3. git commit -m 'Add trained models'"
    echo "  4. git push"
    echo ""
fi

echo ""
echo "Build completed successfully!"
echo "=================================="
