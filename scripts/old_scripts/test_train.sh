#!/bin/bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --num-epochs 1 \
    --batch-size 2 \
    --max-samples 5
