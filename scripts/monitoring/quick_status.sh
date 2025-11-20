#!/bin/bash
# Quick status check - shows just the essentials

if pgrep -f "train_complete_pipeline.py" > /dev/null; then
    echo "✅ Training: RUNNING"
else
    echo "⛔ Training: STOPPED"
fi

echo "📊 Progress: $(tail -300 logs/training_20251113_050638.log | grep -E 'Testing.*%|Training.*%' | tail -1)"
echo "🔥 GPU Usage: $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader)"
echo "💾 GPU Memory: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo "🌡️  Temperature: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)°C"
echo ""
echo "📝 Log: logs/training_20251113_050638.log"
echo "🔍 Full status: ./check_training_status.sh"
