#!/bin/bash
# Setup script for image classification example

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/project"

echo "Setting up image classification example..."

# Create directories
mkdir -p "$PROJECT_DIR/data/train/cat"
mkdir -p "$PROJECT_DIR/data/train/dog"
mkdir -p "$PROJECT_DIR/models"
mkdir -p "$PROJECT_DIR/configs"
mkdir -p "$PROJECT_DIR/logs"

# Check if Python dependencies are available
if ! python -c "import torch" 2>/dev/null; then
    echo "Installing Python dependencies..."
    pip install -r "$PROJECT_DIR/requirements.txt"
fi

# Create synthetic data for demo
echo "Creating synthetic demo data..."
python -c "
import sys
sys.path.insert(0, '$PROJECT_DIR')
from dataset import create_synthetic_data
create_synthetic_data('$PROJECT_DIR/data', num_samples=200, num_classes=2)
print('Created 200 synthetic images (100 per class)')
"

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Run the agent: mlops-agent -i --project $PROJECT_DIR"
echo "  2. Or train directly: cd $PROJECT_DIR && python train.py --synthetic"
echo ""
