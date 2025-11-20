#!/bin/bash
################################################################################
# QUICK START TRAINING
# Run this when you're back - everything is ready!
################################################################################

echo "================================================================================"
echo "🚀 STARTING TRITON TRAINING"
echo "================================================================================"
echo ""
echo "GPU: NVIDIA RTX A6000 (44GB VRAM)"
echo "Dataset: 1,172 contracts, 7 classes"
echo "Expected time: 6-8 hours"
echo "Expected accuracy: 55-70%"
echo ""
echo "================================================================================"
echo ""

# Change to project directory
cd /home/anik/code/Triton

# Create timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "📝 Logs will be saved to: $LOG_FILE"
echo ""
echo "Starting training in 3 seconds..."
sleep 3

# Activate environment and run training
source triton_env/bin/activate
nohup ./scripts/overnight_training.sh > "$LOG_FILE" 2>&1 &

PID=$!

echo ""
echo "================================================================================"
echo "✅ TRAINING STARTED!"
echo "================================================================================"
echo ""
echo "Process ID: $PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Or use TensorBoard (after training starts):"
echo "  source triton_env/bin/activate"
echo "  tensorboard --logdir runs/"
echo ""
echo "Expected completion: $(date -d '+8 hours' '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "================================================================================"
echo "🎯 Go get some rest! Results will be ready when you wake up!"
echo "================================================================================"
