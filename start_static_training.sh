#!/bin/bash

# Static Vulnerability Detection Training Script
# Trains ONLY the static encoder (PDG-based GAT model)

echo "========================================="
echo "Static Vulnerability Detection Training"
echo "========================================="
echo "Model: Graph Attention Network (GAT)"
echo "Input: Program Dependence Graphs (PDG)"
echo "Dataset: FORGE (6,575 contracts)"
echo "========================================="
echo ""

# Create log directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/static_training_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo ""

# Activate environment
source triton_env/bin/activate

# Run training with detailed metrics
python scripts/train_static_only.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --num-epochs 20 \
    --batch-size 8 \
    --learning-rate 0.001 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "Training Complete!"
echo "========================================="
echo "Log file: $LOG_FILE"
echo "Model checkpoint: models/checkpoints/static_encoder_best.pt"
echo "Results: models/checkpoints/test_results_*.txt"
echo "TensorBoard: tensorboard --logdir runs/"
echo "========================================="
