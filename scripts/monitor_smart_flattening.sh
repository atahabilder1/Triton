#!/bin/bash
# Monitor smart flattening V3 progress

LOG_FILE="logs/smart_flattening_v3.log"

echo "========================================"
echo "SMART FLATTENING V3 - LIVE MONITOR"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "SMART FLATTENING V3 - LIVE PROGRESS"
    echo "========================================"
    echo ""

    # Expected total (from forge_reconstructed)
    EXPECTED=1148

    # Count progress from log
    SUCCESS=$(grep -c "✅ SUCCESS" "$LOG_FILE" 2>/dev/null || echo "0")
    FAILED=$(grep -c "❌ FAILED" "$LOG_FILE" 2>/dev/null || echo "0")
    PROCESSED=$((SUCCESS + FAILED))

    # Count first-try successes
    FIRST_TRY=$(grep -c "success_first_try" "$LOG_FILE" 2>/dev/null || echo "0")

    # Calculate percentages
    if [ $PROCESSED -gt 0 ]; then
        PROGRESS_PCT=$(echo "scale=1; $PROCESSED * 100 / $EXPECTED" | bc)
        SUCCESS_RATE=$(echo "scale=1; $SUCCESS * 100 / $PROCESSED" | bc)
    else
        PROGRESS_PCT=0
        SUCCESS_RATE=0
    fi

    echo "Contracts: $PROCESSED / $EXPECTED ($PROGRESS_PCT%)"
    echo "Successful: $SUCCESS"
    echo "Failed: $FAILED"
    echo "Success Rate: $SUCCESS_RATE%"
    echo ""

    # Show recent activity
    echo "Recent Activity:"
    echo "----------------------------------------"
    tail -20 "$LOG_FILE" 2>/dev/null | grep -E "^\[.*\]|SUCCESS|FAILED|Installing" | tail -10
    echo ""

    # Estimate completion time
    if [ $PROCESSED -gt 10 ]; then
        START_TIME=$(stat -c %Y "$LOG_FILE" 2>/dev/null || echo "0")
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - START_TIME))

        if [ $ELAPSED -gt 0 ]; then
            AVG_TIME=$(echo "scale=2; $ELAPSED / $PROCESSED" | bc)
            REMAINING=$((EXPECTED - PROCESSED))
            ETA_SEC=$(echo "$AVG_TIME * $REMAINING" | bc | cut -d. -f1)
            ETA_MIN=$((ETA_SEC / 60))

            echo "Estimated Time Remaining: $ETA_MIN minutes"
        fi
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "========================================"

    sleep 5
done
