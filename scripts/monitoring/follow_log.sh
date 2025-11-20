#!/bin/bash
# Follow training log in real-time (filters out noise)

echo "========================================="
echo "Following training log (filtered)"
echo "Press Ctrl+C to stop"
echo "========================================="
echo ""

tail -f logs/training_20251113_050638.log | grep -E "(PHASE|Testing|Training|Epoch|Loss|Accuracy|completed|ERROR|WARNING|INFO.*Extracted|%)" --line-buffered
