#!/bin/bash

# Static Vulnerability Detection Training Script
# NOW WITH CONFIG SUPPORT - No need to specify paths!

echo "========================================="
echo "Static Vulnerability Detection Training"
echo "========================================="
echo "Using configuration from: config.yaml"
echo "========================================="
echo ""

# Create log directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/static_training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Run training - all settings loaded from config.yaml!
# No paths needed - they're in config.yaml
python scripts/train/static/train_static_optimized.py \
    2>&1 | tee "$LOG_FILE"

# You can still override specific settings if needed:
# python scripts/train/static/train_static_optimized.py \
#     --batch-size 32 \
#     --learning-rate 0.0005 \
#     2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Log file: $LOG_FILE"
echo "View results with: cat models/checkpoints/test_results_*.txt"
echo "TensorBoard: tensorboard --logdir runs/"
echo "========================================="
