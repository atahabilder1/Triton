#!/bin/bash
################################################################################
# Improved Static Training with Enhanced PDG Extraction
# - All Solidity compiler versions installed (77 versions)
# - Exact version matching for better PDG success rate
# - Failure logging enabled for debugging
################################################################################

echo "================================================================================"
echo "TRITON - IMPROVED STATIC VULNERABILITY DETECTION TRAINING"
echo "================================================================================"
echo "🔧 Improvements:"
echo "  - 77 Solidity compiler versions installed (0.4.11 to 0.8.28)"
echo "  - Exact version matching for better compatibility"
echo "  - Failure logging enabled (logs/pdg_failures.log)"
echo "  - Expected PDG success rate: 50-70% (previously 5.8%)"
echo "================================================================================"
echo ""

# Create log directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/improved_training_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/training.log"

echo "📁 Log directory: $LOG_DIR"
echo "📝 Log file: $LOG_FILE"
echo ""

# Ensure we're in the correct directory
cd /home/anik/code/Triton

# Activate environment
source triton_env/bin/activate

# Set PYTHONPATH to include project root
export PYTHONPATH=/home/anik/code/Triton:$PYTHONPATH

echo "🚀 Starting training..."
echo ""

# Run static training with improved PDG extraction
python3 scripts/train/static/train_static_optimized.py \
    --train-dir data/datasets/forge_reconstructed/train \
    --val-dir data/datasets/forge_reconstructed/val \
    --test-dir data/datasets/forge_reconstructed/test \
    --num-epochs 50 \
    --batch-size 16 \
    --learning-rate 0.001 \
    --early-stopping 10 \
    --num-workers 4 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "================================================================================"
echo "✅ TRAINING COMPLETE!"
echo "================================================================================"
echo "📊 Results:"
echo "  - Training log: $LOG_FILE"
echo "  - PDG failures: logs/pdg_failures.log"
echo "  - Model checkpoint: models/checkpoints/static_encoder_best.pt"
echo "  - Test results: models/checkpoints/test_results_*.txt"
echo "  - TensorBoard: runs/static_optimized_*"
echo ""
echo "🔍 To analyze PDG failures:"
echo "  cat logs/pdg_failures.log | wc -l  # Count failures"
echo "  cat logs/pdg_failures.log | head -20  # View first 20 failures"
echo ""
echo "📈 To view TensorBoard:"
echo "  tensorboard --logdir runs/"
echo "================================================================================"
