#!/bin/bash
# Real-time monitoring of perfect flattening progress

LOG_FILE="logs/perfect_flattening.log"

echo "========================================"
echo "FLATTENING PROGRESS MONITOR"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "PERFECT FLATTENING - LIVE PROGRESS"
    echo "========================================"
    echo ""

    # Count progress
    TOTAL=6616
    PROCESSED=$(grep -c "Processing:" "$LOG_FILE" 2>/dev/null || echo "0")
    SUCCESS=$(grep -c "✅ SUCCESS" "$LOG_FILE" 2>/dev/null || echo "0")
    FAILED=$(grep -c "❌ FAILED" "$LOG_FILE" 2>/dev/null || echo "0")

    # Calculate percentages
    if [ $PROCESSED -gt 0 ]; then
        PROGRESS_PCT=$(echo "scale=1; $PROCESSED * 100 / $TOTAL" | bc)
        SUCCESS_RATE=$(echo "scale=1; $SUCCESS * 100 / ($SUCCESS + $FAILED)" | bc 2>/dev/null || echo "0")
    else
        PROGRESS_PCT=0
        SUCCESS_RATE=0
    fi

    echo "Projects: $PROCESSED / $TOTAL ($PROGRESS_PCT%)"
    echo "Successful: $SUCCESS"
    echo "Failed: $FAILED"
    echo "Success Rate: $SUCCESS_RATE%"
    echo ""

    # Show recent activity
    echo "Recent Activity:"
    echo "----------------------------------------"
    tail -15 "$LOG_FILE" 2>/dev/null | grep -E "Processing:|SUCCESS|FAILED" | tail -10
    echo ""

    # Estimate time remaining
    if [ $PROCESSED -gt 0 ]; then
        ELAPSED=$(stat -c %Y "$LOG_FILE" 2>/dev/null)
        START=$(head -1 "$LOG_FILE" | grep -oP '\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}' | head -1)
        if [ ! -z "$START" ]; then
            START_SEC=$(date -d "$START" +%s 2>/dev/null || echo "0")
            CURRENT_SEC=$(date +%s)
            ELAPSED_SEC=$((CURRENT_SEC - START_SEC))

            if [ $ELAPSED_SEC -gt 0 ] && [ $PROCESSED -gt 0 ]; then
                AVG_TIME=$(echo "scale=2; $ELAPSED_SEC / $PROCESSED" | bc)
                REMAINING=$((TOTAL - PROCESSED))
                ETA_SEC=$(echo "$AVG_TIME * $REMAINING" | bc | cut -d. -f1)
                ETA_MIN=$((ETA_SEC / 60))

                echo "Estimated Time Remaining: $ETA_MIN minutes"
            fi
        fi
    fi

    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "========================================"

    sleep 5
done
