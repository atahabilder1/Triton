#!/bin/bash
# Quick training status checker

LOG_FILE="logs/improved_training_20251120_014509/training.log"

echo "================================================================================"
echo "TRAINING STATUS - $(date)"
echo "================================================================================"
echo ""

# Check if training is running
if pgrep -f "train_static_optimized.py" > /dev/null; then
    PID=$(pgrep -f "train_static_optimized.py" | head -1)
    echo "✅ Training is RUNNING (PID: $PID)"
else
    echo "❌ Training is NOT running"
fi

echo ""
echo "================================================================================"
echo "PDG EXTRACTION PROGRESS"
echo "================================================================================"

# Count PDG extractions
PDG_COUNT=$(grep -c "Extracted PDG" "$LOG_FILE" 2>/dev/null || echo "0")
FAIL_COUNT=$(grep -c "Slither CLI analysis failed" "$LOG_FILE" 2>/dev/null || echo "0")
TOTAL=$((PDG_COUNT + FAIL_COUNT))

if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $PDG_COUNT * 100 / $TOTAL" | bc)
    echo "✅ Successfully extracted: $PDG_COUNT PDGs"
    echo "❌ Failed to extract: $FAIL_COUNT contracts"
    echo "📊 Success rate: $SUCCESS_RATE% ($PDG_COUNT/$TOTAL)"
else
    echo "⏳ Still loading datasets..."
fi

echo ""
echo "================================================================================"
echo "TRAINING PROGRESS"
echo "================================================================================"

# Check for epoch information
if grep -q "Epoch" "$LOG_FILE" 2>/dev/null; then
    echo "Latest epochs:"
    grep "Epoch \[" "$LOG_FILE" | tail -5
else
    echo "⏳ Still extracting PDGs, training hasn't started yet..."
fi

echo ""
echo "================================================================================"
echo "RECENT ACTIVITY (last 10 lines)"
echo "================================================================================"
tail -10 "$LOG_FILE" 2>/dev/null || echo "Log file not found"

echo ""
echo "================================================================================"
echo "To monitor live: tail -f $LOG_FILE"
echo "================================================================================"
