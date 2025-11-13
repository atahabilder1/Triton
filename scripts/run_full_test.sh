#!/bin/bash

# Comprehensive Test Script - Tests all 4 modalities
# Usage: ./run_full_test.sh

echo "=================================="
echo "TRITON COMPREHENSIVE TEST SUITE"
echo "=================================="
echo ""
echo "This will test all contracts in the test directory using:"
echo "  1. Static Encoder Only"
echo "  2. Dynamic Encoder Only"
echo "  3. Semantic Encoder Only"
echo "  4. Fusion Model (All 3 Combined)"
echo ""
echo "Test Dataset: data/datasets/combined_labeled/test (44 contracts)"
echo ""
echo "Estimated time: 10-15 minutes"
echo "=================================="
echo ""

# Run comprehensive test
python3 test_comprehensive_report.py \
    --test-dir data/datasets/combined_labeled/test \
    --output COMPREHENSIVE_TEST_REPORT.md

echo ""
echo "=================================="
echo "✓ Testing Complete!"
echo "=================================="
echo ""
echo "Report saved to: COMPREHENSIVE_TEST_REPORT.md"
echo ""
echo "View report:"
echo "  cat COMPREHENSIVE_TEST_REPORT.md"
echo ""
