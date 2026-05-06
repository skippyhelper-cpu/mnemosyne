#!/bin/bash
set -e

echo "Installing Mnemosyne..."
pip install mnemosyne-memory[vec] --quiet
pip install kuzu --quiet

echo "Done. Run: mnemosyne stats"
