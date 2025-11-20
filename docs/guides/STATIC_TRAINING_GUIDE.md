# Static Vulnerability Detection Training Guide

## Quick Start

To train the static-only model:

```bash
./start_static_training.sh
```

This will train for 20 epochs using the FORGE dataset and provide detailed per-vulnerability metrics.

---

## Understanding the Loss Function

### What is a Loss Function?

The **loss function** measures how wrong the model's predictions are. It's the KEY component that enables learning!

**Formula**: `Loss = CrossEntropyLoss(predictions, ground_truth)`

### How it Works

1. **Forward Pass**: Model predicts vulnerability type
   ```
   Smart Contract → PDG → GAT → Prediction: "reentrancy" (confidence: 0.85)
   Ground Truth: "reentrancy"
   ```

2. **Loss Calculation**:
   - If prediction is CORRECT → Low loss (e.g., 0.05)
   - If prediction is WRONG → High loss (e.g., 2.3)

3. **Backward Pass**: Calculate how to adjust weights
   ```
   Loss → Gradients → Weight Updates
   ```

4. **Optimizer Step**: Update model to reduce loss
   ```
   New Weights = Old Weights - Learning_Rate × Gradients
   ```

### Cross-Entropy Loss Explained

For a vulnerability with 11 classes:

```python
# Model outputs probability distribution
predictions = [0.1, 0.05, 0.02, 0.7, 0.03, 0.05, 0.01, 0.02, 0.01, 0.01, 0.0]
             #[AC,  Arith, Bad, DoS, FR,  Reen, SA,  Time, ULC, Oth, Safe]

# Ground truth (one-hot encoded)
ground_truth = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # Label: DoS (index 3)

# Cross-entropy loss
loss = -log(predictions[3]) = -log(0.7) = 0.36
```

If model predicted wrong class (e.g., index 0 with 0.1):
```
loss = -log(0.1) = 2.30  # Much higher!
```

### Class Weighting

Since the dataset is **imbalanced** (e.g., 666 arithmetic vs. 30 short_addresses), we use **class weights**:

```python
weight = 1 / (class_count + epsilon)

# Example weights:
# short_addresses (30 samples): weight = 11.0
# arithmetic (663 samples):     weight = 0.5
```

This ensures rare vulnerabilities like `short_addresses` contribute more to the loss, forcing the model to learn them.

---

## Training Process Step-by-Step

### Phase 1: Data Loading (Caching)

```
Contract.sol → Slither → PDG → Cache (data/cache/)
                  ↓
              Graph with:
              - Nodes: Functions, Variables, Modifiers
              - Edges: Calls, Reads, Writes, Uses_Modifier
```

**Caching saves time**: Slither analysis takes 10-60 seconds per contract, so we cache PDGs and reuse them for all 20 epochs!

### Phase 2: Model Forward Pass

```
PDG → Node Encoding → GAT Layer 1 → GAT Layer 2 → GAT Layer 3 → Pooling → Projection → Scores
```

**Details**:
1. **Node Encoding**: Convert node features (5-dim) to 128-dim embeddings
2. **GAT Layers**: 3 Graph Attention layers learn relationships
3. **Pooling**: Aggregate node embeddings into graph-level embedding
4. **Projection**: Map to 768-dim feature space
5. **Classification Heads**: 11 binary classifiers (one per vulnerability type)

### Phase 3: Loss Calculation

```python
# Get predictions for all 11 vulnerability types
scores = [score_AC, score_Arith, ..., score_Safe]  # Shape: [batch_size, 11]

# Calculate cross-entropy loss with class weights
loss = CrossEntropyLoss(scores, labels, weight=class_weights)
```

### Phase 4: Backpropagation

```
Loss → ∂Loss/∂weights → Update GAT weights → Update projection weights
```

The gradient tells each layer "how to change" to reduce loss.

### Phase 5: Optimization

```python
optimizer = Adam(learning_rate=0.001)
optimizer.step()  # Update all model weights
```

Adam optimizer adjusts weights intelligently using momentum and adaptive learning rates.

---

## Detailed Metrics Output

### What You'll See After Training

```
================================================================================
TEST SET - DETAILED VULNERABILITY DETECTION METRICS
================================================================================

OVERALL ACCURACY: 67.34%

Vulnerability Type              Precision     Recall   F1-Score  Detected/Total
--------------------------------------------------------------------------------
access_control                     0.7234     0.6891     0.7058     102/148   (68.9%)
arithmetic                         0.6912     0.7397     0.7146     108/146   (74.0%)
bad_randomness                     0.5417     0.5208     0.5310      13/24    (52.1%)
denial_of_service                  0.6486     0.6081     0.6277      45/74    (60.8%)
front_running                      0.5938     0.5938     0.5938      19/32    (59.4%)
reentrancy                         0.8319     0.7899     0.8103      94/119   (79.0%)
short_addresses                    0.4286     0.4286     0.4286       3/7     (42.9%)
time_manipulation                  0.6222     0.6222     0.6222      28/45    (62.2%)
unchecked_low_level_calls          0.7260     0.7260     0.7260     106/146   (72.6%)
other                              0.5594     0.5874     0.5731      84/143   (58.7%)
safe                               0.6929     0.7000     0.6964      98/140   (70.0%)
--------------------------------------------------------------------------------
MACRO AVERAGE                      0.6418     0.6369     0.6390
WEIGHTED AVERAGE                                        0.6712
TOTAL DETECTION RATE                                    700/1024  (68.36%)
================================================================================
```

### Understanding the Metrics

**Precision**: Of all contracts predicted as vulnerability X, how many actually had X?
```
Precision = True Positives / (True Positives + False Positives)

Example: Reentrancy precision = 0.8319
→ 83.19% of contracts predicted as reentrancy were actually reentrancy
```

**Recall**: Of all contracts with vulnerability X, how many did we detect?
```
Recall = True Positives / (True Positives + False Negatives)

Example: Reentrancy recall = 0.7899
→ We detected 78.99% (94/119) of all reentrancy vulnerabilities
```

**F1-Score**: Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Support**: Number of actual instances of each vulnerability in test set

---

## Key Insights

### Best Detected Vulnerabilities (High Recall)

1. **Reentrancy**: 79.0% detection rate
   - Why: Clear PDG patterns (external calls + state changes)
   - PDG captures call sequences well

2. **Arithmetic**: 74.0% detection rate
   - Why: Arithmetic operations visible in PDG structure

3. **Unchecked Low-Level Calls**: 72.6% detection rate
   - Why: Call patterns without require/assert checks

### Challenging Vulnerabilities (Low Recall)

1. **Short Addresses**: 42.9% detection rate
   - Why: Very few training samples (only 30!)
   - PDG doesn't capture low-level byte operations well

2. **Bad Randomness**: 52.1% detection rate
   - Why: Semantic issue (using block.timestamp), not structural

3. **Front Running**: 59.4% detection rate
   - Why: Transaction ordering issue, not visible in PDG

### Class Imbalance Impact

Notice how **short_addresses** has only 7 test samples vs. 148 for access_control!

This is why we use **class weighting** - to prevent the model from ignoring rare classes.

---

## Training Time Estimates

**RTX A6000 GPU (46GB VRAM)**:
- Batch size 8: ~2-3 hours
- Batch size 16: ~1.5-2 hours

**CPU Only**:
- Batch size 4: ~12-15 hours

**Bottleneck**: Slither PDG extraction (first epoch only, then cached)

---

## File Outputs

After training, you'll get:

1. **Model Checkpoint**: `models/checkpoints/static_encoder_best.pt`
   - Contains trained weights
   - Can be loaded for inference

2. **Test Results**: `models/checkpoints/test_results_YYYYMMDD_HHMMSS.txt`
   - Detailed metrics saved to text file

3. **Training Log**: `logs/static_training_YYYYMMDD_HHMMSS.log`
   - Full console output

4. **TensorBoard Logs**: `runs/static_only_YYYYMMDD_HHMMSS/`
   - View with: `tensorboard --logdir runs/`

---

## Customization Options

### Adjust Batch Size (for GPU memory)

```bash
python train_static_only.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --batch-size 16  # Increase for faster training (if GPU has memory)
```

### Train on Subset (for quick testing)

```bash
python train_static_only.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --max-samples 500  # Use only 500 training samples
    --num-epochs 5     # Train for fewer epochs
```

### Disable Caching (if PDGs are corrupted)

```bash
python train_static_only.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --no-cache  # Re-extract all PDGs
```

---

## Monitoring Training

### TensorBoard (Real-time)

```bash
tensorboard --logdir runs/
```

Open browser: http://localhost:6006

You'll see:
- Training/Validation Loss curves
- Training/Validation Accuracy curves
- F1 score progression

### Console Output

```
Training Static Encoder: 100%|████████| 568/568 [12:34<00:00, 1.33s/it, loss=0.89]

Epoch 1/20
Train Loss: 1.2345, Train Acc: 45.67%
Val Loss: 1.0987, Val Acc: 52.34%, Val F1: 0.4912
✓ Saved best model (F1: 0.4912)
```

---

## Expected Results

Based on the FORGE dataset:

**Overall Accuracy**: 60-75% (depends on training quality)

**Per-Vulnerability F1 Scores**:
- Reentrancy: 0.75-0.85 (GOOD - clear PDG patterns)
- Arithmetic: 0.65-0.75 (GOOD - structural patterns)
- Access Control: 0.60-0.70 (MEDIUM - modifier patterns)
- Unchecked Calls: 0.65-0.75 (GOOD - missing checks)
- DoS: 0.55-0.65 (MEDIUM - loop patterns)
- Time Manipulation: 0.55-0.65 (MEDIUM - timestamp usage)
- Front Running: 0.50-0.60 (CHALLENGING - transaction ordering)
- Bad Randomness: 0.45-0.55 (CHALLENGING - semantic issue)
- Short Addresses: 0.35-0.50 (POOR - very few samples)
- Other: 0.50-0.60 (MIXED - various patterns)
- Safe: 0.65-0.75 (GOOD - absence of patterns)

---

## Next Steps

1. **Analyze which vulnerabilities are easiest to detect**
   - Look at recall scores in test results

2. **Compare static vs. dynamic vs. semantic**
   - Train all three models separately
   - See which modality works best for each vulnerability

3. **Try fusion model**
   - Combine static + dynamic + semantic
   - Often achieves 5-15% better accuracy

4. **Experiment with hyperparameters**
   - Learning rate: Try 0.0001 or 0.01
   - Hidden dimensions: Increase to 512
   - Number of GAT layers: Try 4-5 layers

---

## Troubleshooting

### Out of Memory Error

Reduce batch size:
```bash
--batch-size 4
```

### Slither Errors

Check Solidity compiler versions:
```bash
solc-select install 0.8.0
solc-select use 0.8.0
```

### Poor Performance on Rare Classes

Increase class weights manually or collect more data.

---

## Questions?

- Check training logs in `logs/`
- View TensorBoard for loss curves
- Read test results file for detailed metrics
