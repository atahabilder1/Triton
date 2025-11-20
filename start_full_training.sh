#!/bin/bash

# Full Training Script for RTX A6000
# Expected time: 8-12 hours
# Dataset: FORGE (4,540 train, 1,011 val, 1,024 test)

echo "========================================="
echo "Starting Triton Full Training"
echo "========================================="
echo "GPU: RTX A6000 (46GB VRAM)"
echo "Dataset: FORGE (6,575 total contracts)"
echo "Expected time: 8-12 hours"
echo "========================================="
echo ""

# Create log directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Activate environment and run training
source triton_env/bin/activate

# Run with optimal batch size for RTX A6000 (46GB VRAM)
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --num-epochs 20 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --skip-tests \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Log file: $LOG_FILE"
echo "Checkpoints: models/checkpoints/"
echo "TensorBoard: tensorboard --logdir runs/"
echo "========================================="
