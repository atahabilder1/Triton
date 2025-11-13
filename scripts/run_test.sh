#!/bin/bash
# Simple script to run model testing

echo "================================"
echo "TRITON MODEL TESTING"
echo "================================"
echo ""

# Default values
TEST_DIR="data/datasets/combined_labeled/test"
MODEL="semantic"
OUTPUT_DIR="results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test-dir)
            TEST_DIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_test.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test-dir DIR      Test directory (default: data/datasets/combined_labeled/test)"
            echo "  --model MODEL       Model to test: semantic, fusion, or all (default: semantic)"
            echo "  --output-dir DIR    Output directory (default: results)"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  ./run_test.sh"
            echo "  ./run_test.sh --model semantic"
            echo "  ./run_test.sh --test-dir /path/to/test --model all"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Test Directory: $TEST_DIR"
echo "Model: $MODEL"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Create timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Run the test
echo "Running tests..."
echo ""

python3 test_all_models.py \
    --test-dir "$TEST_DIR" \
    --models "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee "test_run_${TIMESTAMP}.log"

echo ""
echo "================================"
echo "Testing complete!"
echo "Log saved to: test_run_${TIMESTAMP}.log"
echo "Results in: $OUTPUT_DIR/"
echo "================================"
