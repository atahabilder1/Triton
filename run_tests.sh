#!/bin/bash
# Triton Testing Helper Script
# Quick commands for common testing workflows

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Triton Testing Helper ===${NC}"
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Virtual environment not activated. Activating...${NC}"
    source triton_env/bin/activate
fi

# Check dataset status
if [ -d "data/datasets/smartbugs-curated/dataset" ]; then
    DATASET_COUNT=$(find data/datasets/smartbugs-curated/dataset -name "*.sol" 2>/dev/null | wc -l)
    echo -e "${GREEN}✓ SmartBugs Curated found: ${DATASET_COUNT} contracts${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠ SmartBugs Curated not found. Run option 1 to download.${NC}"
    echo ""
fi

# Show menu
echo "Choose an option:"
echo ""
echo "=== TESTING (Recommended - Start Here) ==="
echo "1. Test ALL SmartBugs Curated (143 contracts) ⭐ START HERE"
echo "2. Test reentrancy only (31 contracts)"
echo "3. Test arithmetic only (15 contracts)"
echo "4. Test access control only (18 contracts)"
echo "5. Test single contract"
echo ""
echo "=== DATASET MANAGEMENT ==="
echo "6. Download SmartBugs Curated dataset"
echo "7. Download all datasets (FORGE, SolidiFI, etc.)"
echo ""
echo "=== RESULTS ==="
echo "8. View latest results"
echo "9. Clean results directory"
echo ""
echo "0. Exit"
echo ""
read -p "Enter choice [0-9]: " choice

case $choice in
    1)
        echo -e "${GREEN}Testing on ALL SmartBugs Curated (143 contracts)...${NC}"
        if [ ! -d "data/datasets/smartbugs-curated/dataset" ]; then
            echo -e "${RED}Dataset not found. Please run option 6 first to download.${NC}"
            exit 1
        fi
        echo -e "${YELLOW}This will test all 10 vulnerability categories (143 contracts)${NC}"
        echo -e "${YELLOW}Estimated time: 30-60 minutes depending on your hardware${NC}"
        echo ""
        python scripts/test_triton.py --dataset smartbugs --output-dir results/smartbugs
        echo ""
        echo -e "${GREEN}Done! Results saved to results/smartbugs/${NC}"
        echo -e "${GREEN}Check the table below for detection breakdown:${NC}"
        echo ""
        ;;
    2)
        echo -e "${GREEN}Testing on reentrancy contracts (31 contracts)...${NC}"
        if [ ! -d "data/datasets/smartbugs-curated/dataset/reentrancy" ]; then
            echo -e "${RED}Dataset not found. Please run option 6 first to download.${NC}"
            exit 1
        fi
        python scripts/test_triton.py --dataset custom \
            --custom-dir data/datasets/smartbugs-curated/dataset/reentrancy \
            --output-dir results/reentrancy
        echo ""
        echo -e "${GREEN}Done! Results saved to results/reentrancy/${NC}"
        ;;
    3)
        echo -e "${GREEN}Testing on arithmetic contracts (15 contracts)...${NC}"
        if [ ! -d "data/datasets/smartbugs-curated/dataset/arithmetic" ]; then
            echo -e "${RED}Dataset not found. Please run option 6 first to download.${NC}"
            exit 1
        fi
        python scripts/test_triton.py --dataset custom \
            --custom-dir data/datasets/smartbugs-curated/dataset/arithmetic \
            --output-dir results/arithmetic
        echo ""
        echo -e "${GREEN}Done! Results saved to results/arithmetic/${NC}"
        ;;
    4)
        echo -e "${GREEN}Testing on access control contracts (18 contracts)...${NC}"
        if [ ! -d "data/datasets/smartbugs-curated/dataset/access_control" ]; then
            echo -e "${RED}Dataset not found. Please run option 6 first to download.${NC}"
            exit 1
        fi
        python scripts/test_triton.py --dataset custom \
            --custom-dir data/datasets/smartbugs-curated/dataset/access_control \
            --output-dir results/access_control
        echo ""
        echo -e "${GREEN}Done! Results saved to results/access_control/${NC}"
        ;;
    5)
        read -p "Enter path to contract file: " contract_path
        if [ ! -f "$contract_path" ]; then
            echo -e "${RED}File not found: $contract_path${NC}"
            exit 1
        fi
        echo -e "${GREEN}Analyzing $contract_path...${NC}"
        python main.py "$contract_path" --verbose
        ;;
    6)
        echo -e "${GREEN}Downloading SmartBugs Curated dataset (143 contracts)...${NC}"
        cd data/datasets
        if [ -d "smartbugs-curated" ]; then
            echo -e "${YELLOW}SmartBugs Curated already exists. Updating...${NC}"
            cd smartbugs-curated
            git pull
            cd ../..
        else
            git clone https://github.com/smartbugs/smartbugs-curated.git
            cd ../..
        fi
        echo -e "${GREEN}Done! Check data/datasets/smartbugs-curated/${NC}"
        echo -e "${GREEN}Found $(find data/datasets/smartbugs-curated/dataset -name '*.sol' 2>/dev/null | wc -l) contracts${NC}"
        ;;
    7)
        echo -e "${GREEN}Downloading all datasets (this may take 10-15 minutes)...${NC}"
        python scripts/download_datasets.py --dataset all
        echo -e "${GREEN}Done! Check data/datasets/${NC}"
        ;;
    8)
        echo -e "${GREEN}Latest test summary:${NC}"
        if [ -f "$(ls -t results/*/triton_test_summary_*.txt 2>/dev/null | head -1)" ]; then
            cat $(ls -t results/*/triton_test_summary_*.txt 2>/dev/null | head -1)
        else
            echo -e "${YELLOW}No results found. Run a test first.${NC}"
        fi
        ;;
    9)
        read -p "Are you sure you want to clean results directory? (y/n): " confirm
        if [ "$confirm" == "y" ]; then
            echo -e "${YELLOW}Cleaning results directory...${NC}"
            rm -rf results/*
            echo -e "${GREEN}Done!${NC}"
        else
            echo -e "${YELLOW}Cancelled.${NC}"
        fi
        ;;
    0)
        echo -e "${GREEN}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}=== Complete ===${NC}"
