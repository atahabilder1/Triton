#!/bin/bash
# Train fusion module WITHOUT caching to avoid compatibility issues

echo "Starting fusion training (no cache)..."
date

# Temporarily modify the script to disable cache
python3 -c "
import sys
sys.argv = [
    'train_complete_pipeline.py',
    '--train-dir', 'data/datasets/combined_labeled/train',
    '--num-epochs', '20',
    '--batch-size', '4',
    '--train-mode', 'fusion',
    '--skip-tests'
]

# Monkey-patch to disable caching
import scripts.train_complete_pipeline as train_module
original_init = train_module.MultiModalDataset.__init__

def new_init(self, *args, **kwargs):
    kwargs['use_cache'] = False  # Force disable cache
    return original_init(self, *args, **kwargs)

train_module.MultiModalDataset.__init__ = new_init

# Run main
train_module.main()
"

echo ""
echo "Training complete!"
date
