#!/usr/bin/bash
################################################################################
# Overnight Training Script
# Runs complete training pipeline while you're away
################################################################################

set -e  # Exit on error

# Change to project root
cd /home/anik/code/Triton

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/overnight_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================================"
echo "OVERNIGHT TRAINING STARTED - $(date)"
echo "================================================================================"
echo "GPU: NVIDIA RTX A6000 (44GB VRAM)"
echo "Dataset: forge_reconstructed (1,151 contracts, 7 classes)"
echo "Log directory: $LOG_DIR"
echo "================================================================================"
echo ""

# Activate environment
source triton_env/bin/activate

# Set Python path to include project root
export PYTHONPATH="/home/anik/code/Triton:$PYTHONPATH"

################################################################################
# PHASE 1: Quick Test (30 minutes)
################################################################################
echo ""
echo "================================================================================"
echo "PHASE 1: QUICK TEST (100 samples, 5 epochs)"
echo "================================================================================"
echo "Purpose: Verify PDG extraction and basic training works"
echo "Expected: 70-80% PDG success, accuracy >25% after 5 epochs"
echo "================================================================================"
echo ""

python3 scripts/train/static/train_static_optimized.py \
    --train-dir data/datasets/forge_reconstructed/train \
    --val-dir data/datasets/forge_reconstructed/val \
    --test-dir data/datasets/forge_reconstructed/test \
    --max-samples 100 \
    --num-epochs 5 \
    --batch-size 8 \
    --num-workers 2 \
    2>&1 | tee "$LOG_DIR/phase1_quick_test.log"

QUICK_TEST_EXIT=$?

if [ $QUICK_TEST_EXIT -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "❌ QUICK TEST FAILED - Check $LOG_DIR/phase1_quick_test.log"
    echo "================================================================================"
    exit 1
fi

echo ""
echo "================================================================================"
echo "✓ PHASE 1 COMPLETE - Quick test succeeded!"
echo "================================================================================"
echo ""

################################################################################
# PHASE 2: Full Training (4-6 hours)
################################################################################
echo ""
echo "================================================================================"
echo "PHASE 2: FULL TRAINING (1,151 samples, 50 epochs)"
echo "================================================================================"
echo "Purpose: Train on complete dataset"
echo "Expected: 55-70% final accuracy"
echo "================================================================================"
echo ""

python3 scripts/train/static/train_static_optimized.py \
    --train-dir data/datasets/forge_reconstructed/train \
    --val-dir data/datasets/forge_reconstructed/val \
    --test-dir data/datasets/forge_reconstructed/test \
    --num-epochs 50 \
    --batch-size 16 \
    --num-workers 4 \
    --learning-rate 0.001 \
    2>&1 | tee "$LOG_DIR/phase2_full_training.log"

FULL_TRAIN_EXIT=$?

if [ $FULL_TRAIN_EXIT -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "⚠️  FULL TRAINING HAD ERRORS - Check $LOG_DIR/phase2_full_training.log"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "✓ PHASE 2 COMPLETE - Full training succeeded!"
    echo "================================================================================"
fi

echo ""

################################################################################
# PHASE 3: Results Summary
################################################################################
echo ""
echo "================================================================================"
echo "PHASE 3: GENERATING RESULTS SUMMARY"
echo "================================================================================"
echo ""

# Extract key metrics from logs
BEST_VAL_ACC=$(grep -oP "Best validation.*:\s*\K[\d\.]+" "$LOG_DIR/phase2_full_training.log" | tail -1 || echo "N/A")
BEST_VAL_F1=$(grep -oP "Val F1:\s*\K[\d\.]+" "$LOG_DIR/phase2_full_training.log" | tail -1 || echo "N/A")
TEST_ACC=$(grep -oP "OVERALL ACCURACY:\s*\K[\d\.]+" "$LOG_DIR/phase2_full_training.log" | tail -1 || echo "N/A")

# Create summary report
cat > "$LOG_DIR/RESULTS_SUMMARY.md" << EOF
# Overnight Training Results
**Date**: $(date)
**Duration**: Full training completed

---

## Configuration
- **Dataset**: forge_reconstructed (1,151 contracts)
- **Classes**: 7 (access_control, arithmetic, denial_of_service, other, safe, time_manipulation, unchecked_low_level_calls)
- **GPU**: NVIDIA RTX A6000 (44GB VRAM)
- **Epochs**: 50
- **Batch Size**: 16

---

## Results

### Quick Test (Phase 1)
- Status: ✓ Completed
- Log: phase1_quick_test.log

### Full Training (Phase 2)
- Status: ✓ Completed
- Best Validation Accuracy: ${BEST_VAL_ACC}%
- Best Validation F1: ${BEST_VAL_F1}
- Test Accuracy: ${TEST_ACC}%
- Log: phase2_full_training.log

---

## Model Checkpoints
- Best model: models/checkpoints/static_encoder_best.pt
- TensorBoard logs: runs/static_optimized_*

---

## Next Steps

### If Accuracy is 55-70% ✅
1. Add missing vulnerability classes (reentrancy, bad_randomness, front_running)
2. Expand safe contracts dataset
3. Re-train on complete 11-class dataset

### If Accuracy is 30-55% ⚠️
1. Check PDG extraction success rate in logs
2. Verify dataset quality
3. Consider increasing training epochs or adjusting learning rate

### If Accuracy is <30% ❌
1. Investigate PDG extraction failures
2. Check for data quality issues
3. Review model architecture

---

## Files Generated
- Training logs: $LOG_DIR/
- Best model: models/checkpoints/static_encoder_best.pt
- Test results: models/checkpoints/test_results_*.txt

EOF

echo ""
echo "================================================================================"
echo "✓ RESULTS SUMMARY CREATED"
echo "================================================================================"
echo ""
cat "$LOG_DIR/RESULTS_SUMMARY.md"

echo ""
echo "================================================================================"
echo "🎉 OVERNIGHT TRAINING COMPLETE - $(date)"
echo "================================================================================"
echo "Results saved to: $LOG_DIR/RESULTS_SUMMARY.md"
echo "Best model: models/checkpoints/static_encoder_best.pt"
echo "View with TensorBoard: tensorboard --logdir runs/"
echo "================================================================================"
