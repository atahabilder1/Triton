#!/bin/bash
# Monitor training progress in real-time

echo "=================================="
echo "TRITON TRAINING MONITOR"
echo "=================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear
    echo "=================================="
    echo "TRAINING PROGRESS MONITOR"
    echo "=================================="
    date
    echo ""

    # Show latest log entries
    if [ -f "training_log.txt" ]; then
        echo "Latest Training Output:"
        echo "----------------------------------"
        tail -n 30 training_log.txt
    else
        echo "⏳ Waiting for training to start..."
        echo "   (training_log.txt not found yet)"
    fi

    echo ""
    echo "=================================="
    echo "Model Checkpoints:"
    echo "=================================="
    ls -lh models/checkpoints/ 2>/dev/null || echo "No checkpoints yet"

    sleep 10
done
