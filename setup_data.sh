#!/bin/bash
# Setup script to download ASO datasets for analysis

set -e

echo "=== Cloning ASO Atlas ==="
if [ ! -d "aso_atlas" ]; then
    git clone https://github.com/barneyhill/aso_atlas.git
else
    echo "aso_atlas already exists, skipping..."
fi

echo ""
echo "=== Cloning ASOptimizer ==="
if [ ! -d "ASOptimizer" ]; then
    git clone https://github.com/Spidercores/ASOptimizer.git
else
    echo "ASOptimizer already exists, skipping..."
fi

echo ""
echo "=== Dataset Summary ==="
echo "ASO Atlas: $(wc -l < aso_atlas/data/aso_atlas.pkl 2>/dev/null || echo 'pickle file') - 190,927 ASO records"
echo "ASOptimizer: $(wc -l < ASOptimizer/dataset/experiments_with_smiles.csv) lines - 36,890 ASO records"

echo ""
echo "Setup complete! Run 'python generate_report.py' to regenerate the analysis report."
