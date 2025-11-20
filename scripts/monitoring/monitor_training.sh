#!/bin/bash

# Training Monitor Script
# Shows real-time training progress

echo "========================================="
echo "Triton Training Monitor"
echo "========================================="
echo ""

# Check if training is running
if pgrep -f "train_complete_pipeline" > /dev/null; then
    echo "✅ Training is RUNNING"
    echo ""

    # Show latest log file
    LATEST_LOG=$(ls -t logs/training_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        echo "📄 Log file: $LATEST_LOG"
        echo ""
        echo "========================================="
        echo "Last 30 lines of training log:"
        echo "========================================="
        tail -30 "$LATEST_LOG"
    else
        echo "📄 Log file: /tmp/triton_training_main.log"
        echo ""
        echo "========================================="
        echo "Last 30 lines of training log:"
        echo "========================================="
        tail -30 /tmp/triton_training_main.log
    fi

    echo ""
    echo "========================================="
    echo "GPU Status:"
    echo "========================================="
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv

    echo ""
    echo "========================================="
    echo "Commands:"
    echo "========================================="
    echo "  tail -f $LATEST_LOG        # Follow log in real-time"
    echo "  tensorboard --logdir runs/             # View TensorBoard"
    echo "  ./monitor_training.sh                  # Run this script again"
    echo "  pkill -f train_complete_pipeline       # Stop training"
    echo "========================================="
else
    echo "❌ Training is NOT running"
    echo ""
    echo "To start training:"
    echo "  ./start_full_training.sh"
fi
