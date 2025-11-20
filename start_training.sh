#!/bin/bash

#===============================================================================
# Triton - Smart Contract Vulnerability Detection Training
# Unified training launcher for all modalities
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${BLUE}ℹ${NC}  $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC}  $1"
}

# Function to show usage
show_usage() {
    cat << EOF
${BLUE}════════════════════════════════════════════════════════════════════════${NC}
  Triton - Smart Contract Vulnerability Detection Training
${BLUE}════════════════════════════════════════════════════════════════════════${NC}

Usage: $0 <training_type> [options]

${GREEN}Training Types:${NC}
  static      Train static analyzer (PDG-based GAT)
  dynamic     Train dynamic analyzer (Execution trace-based)
  semantic    Train semantic analyzer (CodeBERT-based)
  full        Train full multi-modal system (All encoders + fusion)

${GREEN}Options:${NC}
  --help              Show this help message
  --config FILE       Use custom config file (default: config.yaml)
  --batch-size N      Override batch size
  --epochs N          Override number of epochs
  --lr RATE           Override learning rate

${GREEN}Examples:${NC}
  # Train static analyzer with defaults from config.yaml
  $0 static

  # Train semantic analyzer with custom batch size
  $0 semantic --batch-size 16

  # Train full system with custom config
  $0 full --config custom_config.yaml

  # Quick test with limited epochs
  $0 static --epochs 5

${BLUE}════════════════════════════════════════════════════════════════════════${NC}
All configurations are loaded from config.yaml
Edit config.yaml to change default paths and hyperparameters
${BLUE}════════════════════════════════════════════════════════════════════════${NC}
EOF
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]] || [[ -z "$1" ]]; then
    show_usage
    exit 0
fi

TRAINING_TYPE=$1
shift  # Remove first argument

# Validate training type
case $TRAINING_TYPE in
    static|dynamic|semantic|full)
        ;;
    *)
        print_error "Invalid training type: $TRAINING_TYPE"
        echo ""
        show_usage
        exit 1
        ;;
esac

# Print header
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}Triton Training - ${TRAINING_TYPE^^} MODE${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Create directories
mkdir -p logs models/checkpoints

# Get timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/${TRAINING_TYPE}_training_${TIMESTAMP}.log"

print_info "Training type: ${TRAINING_TYPE}"
print_info "Logging to: $LOG_FILE"
print_info "Configuration: config.yaml"
echo ""

# Select the appropriate training script
case $TRAINING_TYPE in
    static)
        SCRIPT="scripts/train/static/train_static_optimized.py"
        print_info "Model: Graph Attention Network (GAT)"
        print_info "Input: Program Dependence Graphs (PDG)"
        ;;
    dynamic)
        SCRIPT="scripts/train/dynamic/train_dynamic.py"
        print_info "Model: Trace-based Neural Network"
        print_info "Input: Execution Traces"
        ;;
    semantic)
        SCRIPT="scripts/train/semantic/train_semantic.py"
        print_info "Model: CodeBERT Transformer"
        print_info "Input: Source Code Tokens"
        ;;
    full)
        SCRIPT="scripts/train/full/train_complete_pipeline.py"
        print_info "Model: Multi-Modal Fusion System"
        print_info "Input: PDG + Traces + Source Code"
        ;;
esac

# Check if script exists
if [ ! -f "$SCRIPT" ]; then
    print_error "Training script not found: $SCRIPT"
    print_warning "Please ensure the script exists or create it first"
    exit 1
fi

echo ""
print_success "Starting training..."
echo ""

# Run training with all arguments passed through
python "$SCRIPT" "$@" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    print_success "Training Complete!"
else
    print_error "Training failed with exit code $EXIT_CODE"
fi
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
print_info "Log file: $LOG_FILE"
print_info "Model checkpoints: models/checkpoints/"
print_info "TensorBoard: tensorboard --logdir runs/"
echo -e "${BLUE}════════════════════════════════════════════════════════════════════════${NC}"
echo ""

exit $EXIT_CODE
