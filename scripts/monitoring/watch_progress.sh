#!/bin/bash
# Live Training Progress Monitor
# Shows real-time updates with progress bar and GPU stats

LOG_FILE="logs/training_20251113_050638.log"

echo "========================================="
echo "   TRITON TRAINING LIVE MONITOR"
echo "========================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    clear

    echo "========================================="
    echo "   TRITON TRAINING LIVE MONITOR"
    echo "========================================="
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Check if training is running
    if pgrep -f "train_complete_pipeline.py" > /dev/null; then
        echo "✅ Status: TRAINING IS RUNNING"
    else
        echo "⛔ Status: TRAINING STOPPED"
    fi

    echo ""
    echo "========================================="
    echo "CURRENT PHASE"
    echo "========================================="

    # Get current phase
    PHASE=$(tail -200 "$LOG_FILE" | grep -E "PHASE [1-4]|TESTING.*ENCODER" | tail -1)
    if [ -z "$PHASE" ]; then
        PHASE="Initializing..."
    fi
    echo "$PHASE"

    echo ""
    echo "========================================="
    echo "PROGRESS"
    echo "========================================="

    # Get latest progress bar
    PROGRESS=$(tail -300 "$LOG_FILE" | grep "%" | tail -1)
    if [ -z "$PROGRESS" ]; then
        PROGRESS="Initializing..."
    fi
    echo "$PROGRESS"

    echo ""
    echo "========================================="
    echo "GPU STATUS"
    echo "========================================="

    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)
    GPU_POWER=$(nvidia-smi --query-gpu=power.draw,power.limit --format=csv,noheader)

    echo "🔥 GPU Utilization: ${GPU_UTIL}%"
    echo "💾 GPU Memory: ${GPU_MEM}"
    echo "🌡️  Temperature: ${GPU_TEMP}°C"
    echo "⚡ Power Draw: ${GPU_POWER}"

    echo ""
    echo "========================================="
    echo "RECENT LOG (Last 10 lines)"
    echo "========================================="
    tail -10 "$LOG_FILE" | grep -v "solc\|Compilation warnings"

    echo ""
    echo "========================================="
    echo "Refreshing every 5 seconds..."
    echo "Press Ctrl+C to stop"
    echo "========================================="

    sleep 5
done
