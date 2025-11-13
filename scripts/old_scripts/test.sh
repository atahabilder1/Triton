#!/bin/bash

# Comprehensive Test Script - Tests All Modalities
# Uses the working test_models.py script

echo "================================================================================"
echo "                    TRITON COMPREHENSIVE TEST SUITE                            "
echo "================================================================================"
echo ""
echo "Testing all modalities on 44 held-out test contracts"
echo "Dataset: data/datasets/combined_labeled/test/"
echo ""
echo "Modalities to test:"
echo "  1. Static Encoder (PDG-based graph analysis)"
echo "  2. Dynamic Encoder (Execution trace analysis)"
echo "  3. Semantic Encoder (CodeBERT semantic understanding)"
echo "  4. Fusion Model (All 3 encoders combined)"
echo ""
echo "Estimated time: 15-20 minutes total"
echo "================================================================================"
echo ""

# Store results
RESULTS_FILE="test_results_$(date +%Y%m%d_%H%M%S).txt"
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Start timestamp
START_TIME=$(date +%s)

# Run comprehensive test (including fusion)
echo "Running tests..."
echo ""

python3 test_models.py \
    --test-dir data/datasets/combined_labeled/test \
    2>&1 | tee "$RESULTS_FILE"

EXIT_CODE=${PIPESTATUS[0]}

# End timestamp
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Final summary
echo ""
echo "================================================================================"
echo "                          TESTS COMPLETED                                       "
echo "================================================================================"
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo "Results saved to: $RESULTS_FILE"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ All tests completed successfully"
    echo ""
    echo "Performance Summary:"
    echo "-------------------"
    grep -A 5 "SUMMARY COMPARISON" "$RESULTS_FILE" | tail -4
    echo ""
else
    echo "❌ Some tests failed. Check the log file for details."
fi

echo ""
echo "To view detailed per-class tables:"
echo "  cat $RESULTS_FILE"
echo ""
echo "To see combined comparison table:"
echo "  grep -A 15 'COMBINED PERFORMANCE TABLE' $RESULTS_FILE"
echo ""
echo "================================================================================"
