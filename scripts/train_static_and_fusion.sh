#!/bin/bash
# Train Static Encoder and Fusion Module with fixed Slither

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="training_logs"
mkdir -p "$LOG_DIR"

echo "=================================="
echo "RETRAINING WITH FIXES"
echo "=================================="
echo "Start Time: $(date)"
echo "Training Data: data/datasets/combined_labeled/train (155 contracts)"
echo "Models: Static Encoder + Fusion Module"
echo "Epochs: 20 per phase"
echo ""

# Phase 1: Train Static Encoder (with fixed Slither)
echo "PHASE 1: Training Static Encoder..."
python3 scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined_labeled/train \
    --num-epochs 20 \
    --batch-size 4 \
    --train-mode static \
    --skip-tests 2>&1 | tee "$LOG_DIR/static_retrain_${TIMESTAMP}.log"

echo ""
echo "Static encoder training complete!"
echo ""

# Phase 2: Train Fusion Module (using new static + existing semantic/dynamic)
echo "PHASE 2: Training Fusion Module..."
python3 scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined_labeled/train \
    --num-epochs 20 \
    --batch-size 4 \
    --train-mode fusion \
    --skip-tests 2>&1 | tee "$LOG_DIR/fusion_retrain_${TIMESTAMP}.log"

echo ""
echo "=================================="
echo "TRAINING COMPLETE!"
echo "=================================="
echo "End Time: $(date)"
echo "Logs saved to: $LOG_DIR/"
echo ""
echo "Next: Test the models with:"
echo "  python3 test_each_modality.py"
echo "=================================="
