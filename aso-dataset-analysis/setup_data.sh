#!/bin/bash
# Setup script to download ASO datasets for analysis
# Usage: ./setup_data.sh              # clones to current directory
#        DATA_DIR=/tmp ./setup_data.sh # clones to /tmp

set -e

DATA_DIR="${DATA_DIR:-.}"
cd "$DATA_DIR"

echo "=== Downloading to $DATA_DIR ==="

echo "=== Cloning ASO Atlas ==="
if [ ! -d "aso_atlas" ]; then
    git clone --depth 1 https://github.com/barneyhill/aso_atlas.git
else
    echo "aso_atlas already exists, skipping..."
fi

echo ""
echo "=== Cloning ASOptimizer ==="
if [ ! -d "ASOptimizer" ]; then
    git clone --depth 1 https://github.com/Spidercores/ASOptimizer.git
else
    echo "ASOptimizer already exists, skipping..."
fi

echo ""
echo "=== Dataset Summary ==="
echo "ASO Atlas: aso_atlas/data/aso_atlas.pkl - 190,927 ASO records"
echo "ASOptimizer: ASOptimizer/dataset/experiments_with_smiles.csv - 36,890 ASO records"

echo ""
echo "Setup complete!"
if [ "$DATA_DIR" != "." ]; then
    echo "Data downloaded to: $DATA_DIR"
    echo "Update notebook paths or create symlinks:"
    echo "  ln -s $DATA_DIR/aso_atlas aso_atlas"
    echo "  ln -s $DATA_DIR/ASOptimizer ASOptimizer"
fi
