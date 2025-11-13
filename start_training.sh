#!/bin/bash
# Start Triton Training with Monitoring

echo "=================================="
echo "TRITON TRAINING LAUNCHER"
echo "=================================="
echo ""
echo "Dataset: Combined Labeled (228 contracts)"
echo "  Train: 155 contracts"
echo "  Val:   29 contracts"
echo "  Test:  44 contracts (held out)"
echo ""
echo "Training Configuration:"
echo "  - Epochs: 20 per phase"
echo "  - Batch Size: 4"
echo "  - Early Stopping: 5 epochs patience"
echo "  - 4 Phases: Static → Dynamic → Semantic → Fusion"
echo ""
echo "Estimated Time: 1-2 hours total"
echo ""
echo "=================================="
echo ""

# Check if already running
if pgrep -f "train_complete_pipeline.py" > /dev/null; then
    echo "⚠️  Training already running!"
    echo ""
    echo "To monitor: tail -f training_log.txt"
    echo "To stop: pkill -f train_complete_pipeline.py"
    exit 1
fi

# Create models directory if needed
mkdir -p models/checkpoints

# Clean old log
if [ -f "training_log.txt" ]; then
    echo "📁 Archiving old training log..."
    mv training_log.txt "training_log_$(date +%Y%m%d_%H%M%S).txt"
fi

echo "🚀 Starting training..."
echo ""
echo "📊 Monitor progress with:"
echo "   ./monitor_training.sh"
echo ""
echo "Or manually:"
echo "   tail -f training_log.txt"
echo ""
echo "=================================="
echo ""

# Start training
python3 scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined_labeled/train \
    --num-epochs 20 \
    --batch-size 4 \
    --train-mode all \
    2>&1 | tee training_log.txt

echo ""
echo "=================================="
echo "✅ TRAINING COMPLETE!"
echo "=================================="
echo ""
echo "Check results:"
echo "  - Models: models/checkpoints/"
echo "  - Log: training_log.txt"
echo ""
echo "Next step:"
echo "  python scripts/test_dataset_performance.py \\"
echo "      --dataset custom \\"
echo "      --custom-dir data/datasets/combined_labeled/test"
echo ""
