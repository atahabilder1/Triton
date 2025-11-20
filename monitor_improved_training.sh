#!/bin/bash
################################################################################
# Monitor Improved Training Progress
# Shows PDG extraction success rate and training metrics
################################################################################

LOG_FILE=$(ls -t logs/improved_training_*/training.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No training log found!"
    exit 1
fi

echo "================================================================================"
echo "IMPROVED TRAINING MONITOR"
echo "================================================================================"
echo "Log file: $LOG_FILE"
echo ""

echo "================================================================================"
echo "PDG EXTRACTION STATUS"
echo "================================================================================"

# Count PDG extractions
SUCCESS_COUNT=$(grep -c "Extracted PDG with" "$LOG_FILE" 2>/dev/null || echo "0")
EMPTY_PDG=$(grep -c "empty PDG" "$LOG_FILE" 2>/dev/null || echo "0")
FAILURE_COUNT=$(grep -c "Slither CLI analysis failed" "$LOG_FILE" 2>/dev/null || echo "0")
TOTAL=$((SUCCESS_COUNT + FAILURE_COUNT + EMPTY_PDG))

if [ $TOTAL -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=1; $SUCCESS_COUNT * 100 / $TOTAL" | bc)
    echo "✅ Successful PDG extractions: $SUCCESS_COUNT"
    echo "❌ Failed PDG extractions: $FAILURE_COUNT"
    echo "⚠️  Empty PDG extractions: $EMPTY_PDG"
    echo "📊 Total processed: $TOTAL"
    echo "📈 Success rate: $SUCCESS_RATE%"
else
    echo "⏳ PDG extraction starting..."
fi

echo ""
echo "================================================================================"
echo "TRAINING PROGRESS"
echo "================================================================================"

# Check if training has started
if grep -q "EPOCH" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "Latest epoch:"
    grep "EPOCH.*SUMMARY" "$LOG_FILE" | tail -1
    echo ""
    echo "Latest metrics:"
    grep -E "Train Loss:|Val Loss:|Val F1:" "$LOG_FILE" | tail -5
else
    echo "⏳ Training not started yet (loading dataset and extracting PDGs)"
    echo ""
    echo "Recent activity:"
    tail -10 "$LOG_FILE" | grep -v "solc\|Python API failed"
fi

echo ""
echo "================================================================================"
echo "PDG FAILURE ANALYSIS"
echo "================================================================================"

if [ -f "logs/pdg_failures.log" ]; then
    TOTAL_FAILURES=$(wc -l < logs/pdg_failures.log)
    echo "📝 Total failures logged: $TOTAL_FAILURES"
    echo ""
    echo "Top 10 failure types:"
    cut -d'|' -f2 logs/pdg_failures.log | sort | uniq -c | sort -rn | head -10
else
    echo "⏳ No failures logged yet"
fi

echo ""
echo "================================================================================"
echo "LIVE LOG (last 15 lines)"
echo "================================================================================"
tail -15 "$LOG_FILE" | grep -v "solc\|Python API failed"

echo ""
echo "================================================================================"
echo "To follow live: tail -f $LOG_FILE"
echo "================================================================================"
