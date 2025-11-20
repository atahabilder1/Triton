#!/bin/bash

# Detailed Training Monitor
# Shows comprehensive training statistics

# Find the latest log file automatically
LOG_FILE=$(ls -t logs/training_*.log 2>/dev/null | head -1)

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

if [ ! -f "$LOG_FILE" ]; then
    echo "No training log file found!"
    exit 1
fi

clear

echo -e "${BOLD}${CYAN}================================================================================"
echo "                    TRITON TRAINING DETAILED MONITOR"
echo -e "================================================================================${NC}"
echo ""
echo -e "${BLUE}Log file:${NC} $LOG_FILE"
echo ""

# Check if training process is running
if pgrep -f "train_complete_pipeline.py" > /dev/null; then
    echo -e "${GREEN}✓ Training process: RUNNING${NC}"
else
    echo -e "${RED}✗ Training process: STOPPED${NC}"
fi

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}TRAINING PHASES${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

# Show training phases
grep -E "(TESTING|PHASE|Training|complete)" "$LOG_FILE" | grep -E "(TESTING ALL|PHASE [0-9]|training complete)" | tail -5

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}CURRENT TRAINING METRICS${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

# Get the latest epoch metrics
LATEST_METRICS=$(grep -E "Epoch [0-9]+/[0-9]+|Train Loss:|Train Acc:|Val Loss:|Val Acc:|Saved best" "$LOG_FILE" | tail -20)

if [ -z "$LATEST_METRICS" ]; then
    echo -e "${YELLOW}No training metrics yet (still in testing phase)${NC}"
else
    echo "$LATEST_METRICS"
fi

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}EPOCH SUMMARY${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

# Create epoch summary table
echo -e "${BOLD}Epoch   Train Loss   Train Acc   Val Loss   Val Acc   Status${NC}"
echo "─────────────────────────────────────────────────────────────────"

grep -B1 -A3 "Epoch [0-9]\+/[0-9]\+" "$LOG_FILE" | \
    awk '/Epoch [0-9]+\/[0-9]+/ {epoch=$2}
         /Train Loss:/ {train_loss=$3; train_acc=$6}
         /Val Loss:/ {val_loss=$3; val_acc=$6;
                      printf "%-7s %-12s %-11s %-10s %-9s ", epoch, train_loss, train_acc, val_loss, val_acc}
         /Saved best/ {print "✓ BEST"; next}
         /Val Loss:/ {print ""; next}' | tail -10

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}BEST CHECKPOINTS SAVED${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

grep "Saved best" "$LOG_FILE" | tail -5

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}TRAINING ERRORS & WARNINGS${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

# Count errors and warnings
ERROR_COUNT=$(grep -c "Training error:" "$LOG_FILE" 2>/dev/null || echo "0")
SLITHER_FAIL=$(grep -c "Slither CLI analysis failed" "$LOG_FILE" 2>/dev/null || echo "0")
MYTHRIL_FAIL=$(grep -c "Mythril.*failed" "$LOG_FILE" 2>/dev/null || echo "0")

# Clean up counts (remove extra newlines)
ERROR_COUNT=$(echo "$ERROR_COUNT" | head -1)
SLITHER_FAIL=$(echo "$SLITHER_FAIL" | head -1)
MYTHRIL_FAIL=$(echo "$MYTHRIL_FAIL" | head -1)

echo -e "${BOLD}Training errors:${NC} $ERROR_COUNT"
echo -e "${BOLD}Slither failures:${NC} $SLITHER_FAIL (expected - returns empty PDGs)"
echo -e "${BOLD}Mythril failures:${NC} $MYTHRIL_FAIL"

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo ""
    echo -e "${RED}Recent training errors:${NC}"
    grep "Training error:" "$LOG_FILE" | tail -3
fi

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}GPU STATUS${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits | \
        awk -F', ' '{printf "GPU %s: %s\n  Utilization: %s%%\n  Memory: %s/%s MB\n  Temperature: %s°C\n",
                     $1, $2, $3, $4, $5, $6}'
else
    echo "nvidia-smi not available"
fi

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}PROGRESS INDICATORS${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

# Show latest progress bar
LATEST_PROGRESS=$(grep -E "Training Static Encoder:|Training Dynamic Encoder:|Training Semantic Encoder:|Training Fusion Module:" "$LOG_FILE" | tail -1)
if [ -n "$LATEST_PROGRESS" ]; then
    echo "$LATEST_PROGRESS"
else
    echo "Waiting for training to start..."
fi

echo ""
echo -e "${BOLD}${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"
echo -e "${BOLD}TRAINING HEALTH CHECK${NC}"
echo -e "${MAGENTA}───────────────────────────────────────────────────────────────────────────────${NC}"

# Analyze training health
if grep -q "Train Acc:" "$LOG_FILE"; then
    LATEST_TRAIN_ACC=$(grep "Train Acc:" "$LOG_FILE" | tail -1 | awk '{print $6}' | sed 's/%//')
    LATEST_VAL_ACC=$(grep "Val Acc:" "$LOG_FILE" | tail -1 | awk '{print $6}' | sed 's/%//')

    echo -e "${BOLD}Latest Training Accuracy:${NC} ${LATEST_TRAIN_ACC}%"
    echo -e "${BOLD}Latest Validation Accuracy:${NC} ${LATEST_VAL_ACC}%"
    echo ""

    # Health assessment
    if (( $(echo "$LATEST_TRAIN_ACC < 10" | bc -l) )); then
        echo -e "${RED}⚠️  WARNING: Training accuracy is very low (< 10%)${NC}"
        echo -e "${RED}   This suggests the model is not learning properly!${NC}"
    elif (( $(echo "$LATEST_TRAIN_ACC < 15" | bc -l) )); then
        echo -e "${YELLOW}⚠️  CAUTION: Training accuracy is low (< 15%)${NC}"
        echo -e "${YELLOW}   Monitor next few epochs carefully.${NC}"
    elif (( $(echo "$LATEST_TRAIN_ACC > 50" | bc -l) )) && (( $(echo "$LATEST_VAL_ACC < 30" | bc -l) )); then
        echo -e "${YELLOW}⚠️  CAUTION: Possible overfitting detected${NC}"
        echo -e "${YELLOW}   Training accuracy much higher than validation.${NC}"
    else
        echo -e "${GREEN}✓ Training appears healthy${NC}"
    fi

    # Check if accuracy is improving
    EPOCHS_WITH_METRICS=$(grep "Train Acc:" "$LOG_FILE" | wc -l)
    if [ "$EPOCHS_WITH_METRICS" -ge 3 ]; then
        FIRST_3_ACCS=$(grep "Train Acc:" "$LOG_FILE" | head -3 | awk '{print $6}' | sed 's/%//')
        LAST_3_ACCS=$(grep "Train Acc:" "$LOG_FILE" | tail -3 | awk '{print $6}' | sed 's/%//')

        AVG_FIRST=$(echo "$FIRST_3_ACCS" | awk '{s+=$1} END {print s/NR}')
        AVG_LAST=$(echo "$LAST_3_ACCS" | awk '{s+=$1} END {print s/NR}')

        if (( $(echo "$AVG_LAST > $AVG_FIRST" | bc -l) )); then
            echo -e "${GREEN}✓ Training accuracy is improving${NC}"
        else
            echo -e "${RED}⚠️  WARNING: Training accuracy not improving${NC}"
        fi
    fi
else
    echo -e "${YELLOW}Waiting for first epoch to complete...${NC}"
fi

echo ""
echo -e "${BOLD}${CYAN}================================================================================"
echo "                    END OF TRAINING REPORT"
echo -e "================================================================================${NC}"
echo ""
echo -e "${BOLD}Commands:${NC}"
echo "  Monitor live: tail -f $LOG_FILE | grep -E 'Epoch|Train|Val'"
echo "  GPU status:   watch -n 1 nvidia-smi"
echo "  Quick status: ./quick_status.sh"
echo ""
