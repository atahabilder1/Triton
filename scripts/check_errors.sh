#!/bin/bash
echo "=================================="
echo "ERROR CHECK - Training Status"
echo "=================================="
echo ""

# Check if running
if pgrep -f train_complete_pipeline > /dev/null; then
    echo "✅ Training is RUNNING"
else
    echo "❌ Training STOPPED"
fi

echo ""
echo "Latest 30 lines from log:"
echo "----------------------------------"
tail -30 retrain_output_*.log

echo ""
echo "=================================="
echo "Checking for CRITICAL errors:"
echo "=================================="

# Look for actual errors (not Slither warnings)
CRITICAL=$(tail -200 retrain_output_*.log | grep -iE "Exception|Traceback|Error:" | grep -v "WARNING" | grep -v "Slither")

if [ -n "$CRITICAL" ]; then
    echo "⚠️  CRITICAL ERRORS FOUND:"
    echo "$CRITICAL"
else
    echo "✅ No critical errors - training is healthy"
    echo "   (Slither warnings are normal and expected)"
fi

echo ""
