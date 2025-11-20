#!/bin/bash

# GPU-Optimized Static Vulnerability Detection Training
# Uses FILTERED dataset (interfaces/libraries removed)

echo "========================================="
echo "🚀 GPU-OPTIMIZED TRAINING (FILTERED DATA)"
echo "========================================="
echo "Model: Static Encoder (GAT)"
echo "Device: GPU (CUDA)"
echo "Dataset: FORGE FILTERED (2,596 train / 575 val / 575 test)"
echo "Filter: Removed interfaces, libraries, empty contracts"
echo "========================================="
echo ""

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available!"
    echo "Training will use CPU (much slower)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "📊 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Create directories
mkdir -p logs
mkdir -p models/checkpoints

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_filtered_${TIMESTAMP}.log"

echo "📝 Logging to: $LOG_FILE"
echo ""

# Activate environment
source triton_env/bin/activate

echo "⚙️  Training Configuration:"
echo "  - Dataset: FILTERED (quality contracts only)"
echo "  - Batch size: 16 (optimized for A6000)"
echo "  - Workers: 8 (parallel data loading)"
echo "  - Max epochs: 50 (with early stopping)"
echo "  - Learning rate: 0.0001 (with auto-decay)"
echo ""

python scripts/train_static_optimized.py \
    --train-dir data/datasets/forge_filtered/train \
    --val-dir data/datasets/forge_filtered/val \
    --test-dir data/datasets/forge_filtered/test \
    --batch-size 16 \
    --num-epochs 50 \
    --learning-rate 0.0001 \
    --num-workers 8 \
    --early-stopping 10 \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================="
echo "✅ Training Complete!"
echo "========================================="
echo "📁 Outputs:"
echo "  - Log: $LOG_FILE"
echo "  - Model: models/checkpoints/static_encoder_best.pt"
echo "  - Results: models/checkpoints/test_results_*.txt"
echo ""
echo "📈 View training progress:"
echo "  tensorboard --logdir runs/"
echo "========================================="
