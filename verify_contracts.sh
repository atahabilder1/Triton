#!/bin/bash

#===============================================================================
# Contract Extraction Verification
# Verifies PDG and AST extraction for smart contracts
#===============================================================================

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
    echo -e "  ${GREEN}Triton - Contract Extraction Verification${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════${NC}"
}

show_usage() {
    cat << EOF

${GREEN}Usage:${NC}
  $0 <directory> [options]

${GREEN}Arguments:${NC}
  directory              Directory containing Solidity contracts

${GREEN}Options:${NC}
  --max N               Maximum number of contracts to verify (default: 100)
  --output FILE         Save JSON report to FILE
  --help                Show this help message

${GREEN}Examples:${NC}
  # Verify 100 contracts from train directory
  $0 data/datasets/forge_balanced_accurate/train/reentrancy

  # Verify specific number of contracts
  $0 data/datasets/forge_balanced_accurate/train/safe --max 50

  # Save detailed report
  $0 data/datasets/forge_balanced_accurate/train --max 100 --output report.json

${GREEN}What it does:${NC}
  1. Flattens each Solidity contract
  2. Extracts PDG (Program Dependence Graph) using Slither
  3. Extracts AST (Abstract Syntax Tree) using solc
  4. Reports success/failure statistics
  5. Saves detailed JSON report

EOF
}

# Check for help
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]] || [[ -z "$1" ]]; then
    print_header
    show_usage
    exit 0
fi

DIR="$1"
shift

# Validate directory
if [ ! -d "$DIR" ]; then
    echo -e "${YELLOW}⚠️  Directory not found: $DIR${NC}"
    exit 1
fi

print_header
echo ""
echo -e "${GREEN}Directory:${NC} $DIR"
echo ""

# Run verification
python scripts/utils/verify_extraction.py --dir "$DIR" "$@"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ Verification completed successfully!${NC}"
else
    echo -e "${YELLOW}⚠️  Verification completed with warnings${NC}"
fi
echo ""

exit $EXIT_CODE
