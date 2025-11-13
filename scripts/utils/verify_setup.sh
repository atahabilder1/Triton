#!/bin/bash
# Verify training setup before starting

cd /home/anik/code/Triton

echo "=================================================="
echo "Triton Training Setup Verification"
echo "=================================================="
echo ""

# Check virtual environment
echo "1. Checking virtual environment..."
if [ -d "triton_env" ]; then
    echo "   ✓ Virtual environment found: triton_env"
else
    echo "   ❌ Virtual environment not found!"
    exit 1
fi

# Check Python packages
echo ""
echo "2. Checking Python packages..."
source triton_env/bin/activate

python << 'EOF'
import sys
missing = []

try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except:
    missing.append("torch")
    print("   ❌ PyTorch not found")

try:
    import transformers
    print(f"   ✓ Transformers {transformers.__version__}")
except:
    missing.append("transformers")
    print("   ❌ Transformers not found")

try:
    import networkx
    print(f"   ✓ NetworkX")
except:
    missing.append("networkx")
    print("   ❌ NetworkX not found")

if missing:
    print(f"\n   Install missing packages with:")
    print(f"   pip install {' '.join(missing)}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

# Check GPU
echo ""
echo "3. Checking GPU availability..."
python -c "import torch; print(f'   ✓ CUDA available: {torch.cuda.is_available()}'); print(f'   ✓ GPU count: {torch.cuda.device_count()}') if torch.cuda.is_available() else print('   ⚠ No GPU - training will use CPU (very slow!)')"

# Check datasets
echo ""
echo "4. Checking datasets..."
if [ -L "data/datasets/FORGE-Artifacts" ]; then
    echo "   ✓ FORGE-Artifacts dataset (symlink to /data)"
    du -sh data/datasets/FORGE-Artifacts 2>/dev/null | awk '{print "   Size: " $1}'
else
    echo "   ❌ FORGE-Artifacts dataset not found!"
fi

# Check cache directory
echo ""
echo "5. Checking cache directory..."
if [ -L "data/cache" ]; then
    echo "   ✓ Cache directory (symlink to /data)"
else
    echo "   ❌ Cache directory not properly linked!"
fi

# Check disk space
echo ""
echo "6. Checking disk space..."
df -h /data | tail -1 | awk '{print "   /data partition: " $4 " available (" $5 " used)"}'

AVAIL=$(df /data | tail -1 | awk '{print $4}')
if [ $AVAIL -lt 100000000 ]; then
    echo "   ⚠ Warning: Less than 100GB free - training needs ~75GB for cache"
else
    echo "   ✓ Sufficient space for training"
fi

# Check training script
echo ""
echo "7. Checking training script..."
if [ -f "scripts/train_complete_pipeline.py" ]; then
    echo "   ✓ Training script found"
else
    echo "   ❌ Training script not found!"
    exit 1
fi

echo ""
echo "=================================================="
echo "✓ Setup verification complete!"
echo "=================================================="
echo ""
echo "Ready to start training with:"
echo "  ./start_training.sh"
echo ""
