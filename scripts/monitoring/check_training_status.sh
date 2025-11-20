#!/bin/bash
# Training Status Checker
# Run this script when you return to check training results

echo "========================================="
echo "TRITON TRAINING STATUS CHECKER"
echo "========================================="
echo ""

LOG_FILE="logs/training_20251113_050638.log"

# Check if training is still running
if pgrep -f "train_complete_pipeline.py" > /dev/null; then
    echo "⏳ STATUS: TRAINING IS STILL RUNNING"
    echo ""

    # Get current phase
    CURRENT_PHASE=$(tail -100 "$LOG_FILE" | grep -E "PHASE [1-4]|TESTING.*ENCODER" | tail -1)
    echo "Current Phase: $CURRENT_PHASE"
    echo ""

    # Get latest progress
    PROGRESS=$(tail -200 "$LOG_FILE" | grep "%" | tail -1)
    echo "Latest Progress: $PROGRESS"
    echo ""

    # GPU usage
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu --format=csv,noheader
    echo ""

else
    echo "✅ STATUS: TRAINING COMPLETED"
    echo ""

    # Check if completed successfully
    if tail -50 "$LOG_FILE" | grep -q "Training completed successfully"; then
        echo "🎉 TRAINING COMPLETED SUCCESSFULLY!"
    else
        echo "❌ TRAINING MAY HAVE FAILED - CHECK LOG"
    fi
    echo ""
fi

# Show last 30 lines of log
echo "========================================="
echo "LAST 30 LINES OF LOG:"
echo "========================================="
tail -30 "$LOG_FILE"
echo ""

# Check for errors
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE")
echo "========================================="
echo "Total Errors Found: $ERROR_COUNT"
echo "(Note: Slither errors are normal for contracts with missing dependencies)"
echo "========================================="
echo ""

# Show model checkpoints
echo "========================================="
echo "SAVED MODEL CHECKPOINTS:"
echo "========================================="
ls -lh models/checkpoints/ 2>/dev/null || echo "No checkpoints found yet"
echo ""

# Final results summary
if tail -100 "$LOG_FILE" | grep -q "Final Test Results\|Testing Results"; then
    echo "========================================="
    echo "FINAL TEST RESULTS:"
    echo "========================================="
    tail -100 "$LOG_FILE" | grep -A 20 "Final Test Results\|Testing Results"
fi

echo ""
echo "========================================="
echo "For detailed logs: tail -f $LOG_FILE"
echo "For TensorBoard: tensorboard --logdir runs/"
echo "========================================="
