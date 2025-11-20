#!/bin/bash

# GPU-Optimized Static Vulnerability Detection Training
# Uses full GPU power with optimal settings

echo "========================================="
echo "🚀 GPU-OPTIMIZED TRAINING"
echo "========================================="
echo "Model: Static Encoder (GAT)"
echo "Device: GPU (CUDA)"
echo "Dataset: FORGE (6,575 contracts)"
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
LOG_FILE="logs/training_gpu_${TIMESTAMP}.log"

echo "📝 Logging to: $LOG_FILE"
echo ""

# Activate environment
source triton_env/bin/activate

# Optimal settings for RTX A6000 (46GB VRAM)
# Adjust batch size based on your GPU memory:
# - RTX 3090 (24GB): --batch-size 12
# - RTX 4090 (24GB): --batch-size 14
# - A6000 (46GB):    --batch-size 16
# - H100 (80GB):     --batch-size 24

echo "⚙️  Training Configuration:"
echo "  - Batch size: 16 (optimized for A6000)"
echo "  - Workers: 8 (parallel data loading)"
echo "  - Max epochs: 50 (with early stopping)"
echo "  - Learning rate: 0.001 (with auto-decay)"
echo ""

python scripts/train_static_optimized.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
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
