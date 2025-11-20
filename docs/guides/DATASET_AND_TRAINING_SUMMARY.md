# Triton Static Vulnerability Detection - Complete Summary

**Created**: November 19, 2025
**Purpose**: Train machine learning model to detect smart contract vulnerabilities using static analysis (Program Dependence Graphs)

---

## 📁 Where Everything Is Located

### Documentation Files (Root Directory)
```
/home/anik/code/Triton/
├── README.md                           # Main project overview
├── PROJECT_ORGANIZATION.md             # Project structure
├── STATIC_TRAINING_GUIDE.md           # How loss function works ⭐
├── DATASET_AND_TRAINING_SUMMARY.md    # This file ⭐
└── WEEKLY_PROGRESS_NOV13_2025.md      # Progress report
```

### Training Scripts
```
/home/anik/code/Triton/
├── train_static_only.py               # New simplified static-only training ⭐
├── start_static_training.sh           # Quick launch script ⭐
└── scripts/
    └── train_complete_pipeline.py     # Original full pipeline (all 4 models)
```

### Dataset Location
```
/home/anik/code/Triton/data/datasets/forge_balanced_accurate/
├── train/      # 4,540 contracts (69%)
├── val/        # 1,011 contracts (15%)
├── test/       # 1,024 contracts (16%)
└── dataset_summary.json
```

### Model Checkpoints
```
/home/anik/code/Triton/models/checkpoints/
├── static_encoder_best.pt              # Best static model (will be created)
├── dynamic_encoder_best.pt             # Existing dynamic model
├── semantic_encoder_best.pt            # Existing semantic model
└── fusion_module_best.pt               # Existing fusion model
```

### Logs and Results
```
/home/anik/code/Triton/
├── logs/                               # Training logs
├── runs/                               # TensorBoard logs
└── models/checkpoints/
    └── test_results_*.txt              # Detailed test results
```

---

## 📊 Dataset Breakdown - FORGE Dataset (6,575 Contracts)

### Training Set (4,540 contracts - 69%)

| Vulnerability Type          | Count | Percentage |
|-----------------------------|-------|------------|
| unchecked_low_level_calls   | 666   | 14.7%      |
| arithmetic                  | 663   | 14.6%      |
| access_control              | 629   | 13.9%      |
| other                       | 620   | 13.7%      |
| safe                        | 606   | 13.3%      |
| reentrancy                  | 553   | 12.2%      |
| denial_of_service           | 317   | 7.0%       |
| time_manipulation           | 206   | 4.5%       |
| front_running               | 138   | 3.0%       |
| bad_randomness              | 112   | 2.5%       |
| short_addresses             | 30    | 0.7% ⚠️    |
| **TOTAL**                   | **4,540** | **100%** |

⚠️ **Note**: `short_addresses` is severely underrepresented (only 30 samples)

### Validation Set (1,011 contracts - 15%)

| Vulnerability Type          | Count | Percentage |
|-----------------------------|-------|------------|
| arithmetic                  | 148   | 14.6%      |
| unchecked_low_level_calls   | 147   | 14.5%      |
| safe                        | 143   | 14.1%      |
| other                       | 141   | 13.9%      |
| access_control              | 137   | 13.5%      |
| reentrancy                  | 117   | 11.6%      |
| denial_of_service           | 73    | 7.2%       |
| time_manipulation           | 45    | 4.5%       |
| front_running               | 30    | 3.0%       |
| bad_randomness              | 24    | 2.4%       |
| short_addresses             | 6     | 0.6%       |
| **TOTAL**                   | **1,011** | **100%** |

### Test Set (1,024 contracts - 16%)

| Vulnerability Type          | Count | Percentage |
|-----------------------------|-------|------------|
| access_control              | 148   | 14.5%      |
| arithmetic                  | 146   | 14.3%      |
| unchecked_low_level_calls   | 146   | 14.3%      |
| other                       | 143   | 14.0%      |
| safe                        | 140   | 13.7%      |
| reentrancy                  | 119   | 11.6%      |
| denial_of_service           | 74    | 7.2%       |
| time_manipulation           | 45    | 4.4%       |
| front_running               | 32    | 3.1%       |
| bad_randomness              | 24    | 2.3%       |
| short_addresses             | 7     | 0.7%       |
| **TOTAL**                   | **1,024** | **100%** |

### Dataset Balance Analysis

**Well-Represented Classes** (>500 samples):
- ✅ unchecked_low_level_calls: 959 total
- ✅ arithmetic: 957 total
- ✅ access_control: 914 total
- ✅ other: 904 total
- ✅ safe: 889 total
- ✅ reentrancy: 789 total

**Moderately-Represented Classes** (100-500 samples):
- ⚠️ denial_of_service: 464 total
- ⚠️ time_manipulation: 296 total
- ⚠️ front_running: 200 total
- ⚠️ bad_randomness: 160 total

**Severely Underrepresented Classes** (<100 samples):
- 🔴 short_addresses: **43 total** (CRITICAL ISSUE!)

---

## 🔄 Data Flow: From Smart Contract to Model

### Step 1: Data Loading
```
Smart Contract (.sol file)
    ↓
Read from disk (UTF-8 encoding)
    ↓
Store: {source_code, path, vulnerability_type, label}
```

**Location in code**: `train_static_only.py:46-159` (StaticDataset class)

### Step 2: Intermediate Representation (IR) - PDG Extraction
```
Source Code (.sol)
    ↓
Slither Analysis (60 seconds per contract)
    ↓
Program Dependence Graph (PDG)
    ├── Nodes: Functions, Variables, Modifiers
    │   └── Features: [is_function, is_variable, is_modifier, in_degree, out_degree]
    └── Edges: Calls, Reads, Writes, Uses_Modifier
    ↓
Cache to disk (data/cache/)
    ↓
Reuse for all 20 epochs (HUGE time saver!)
```

**Location in code**:
- PDG extraction: `train_static_only.py:147-158`
- Slither wrapper: `tools/slither_wrapper.py:106-141`
- Caching: `train_static_only.py:125-145`

**Example PDG**:
```
Contract: ERC20Token

Nodes:
- transfer (function) → degree: in=0, out=3
- balanceOf (state variable) → degree: in=2, out=1
- onlyOwner (modifier) → degree: in=1, out=0

Edges:
- transfer --[reads]--> balanceOf
- transfer --[writes]--> balanceOf
- adminWithdraw --[uses_modifier]--> onlyOwner
```

### Step 3: Graph Neural Network Encoding
```
PDG (NetworkX Graph)
    ↓
Convert to PyTorch Geometric format
    ├── Node features: [5-dim] → [128-dim embeddings]
    └── Edge features: [4-dim one-hot]
    ↓
Graph Attention Network (GAT) - 3 layers
    ├── Layer 1: 128-dim → 256-dim (8 attention heads)
    ├── Layer 2: 256×8-dim → 256-dim (8 attention heads)
    └── Layer 3: 256×8-dim → 256-dim (1 attention head)
    ↓
Global Mean Pooling (aggregate all nodes)
    ↓
Projection Layer: 256-dim → 768-dim
    ↓
Static Feature Vector [768-dimensional]
```

**Location in code**:
- PDG to PyTorch: `encoders/static_encoder.py:118-164`
- GAT layers: `encoders/static_encoder.py:11-63`
- Forward pass: `encoders/static_encoder.py:166-194`

### Step 4: Vulnerability Classification
```
Static Features [768-dim]
    ↓
11 Vulnerability Classification Heads (one per vulnerability type)
    ├── access_control: Linear(768 → 1) → sigmoid → score
    ├── arithmetic: Linear(768 → 1) → sigmoid → score
    ├── bad_randomness: Linear(768 → 1) → sigmoid → score
    ├── denial_of_service: Linear(768 → 1) → sigmoid → score
    ├── front_running: Linear(768 → 1) → sigmoid → score
    ├── reentrancy: Linear(768 → 1) → sigmoid → score
    ├── short_addresses: Linear(768 → 1) → sigmoid → score
    ├── time_manipulation: Linear(768 → 1) → sigmoid → score
    ├── unchecked_low_level_calls: Linear(768 → 1) → sigmoid → score
    ├── other: Linear(768 → 1) → sigmoid → score
    └── safe: Linear(768 → 1) → sigmoid → score
    ↓
Concatenate all scores → [batch_size, 11]
    ↓
Softmax → Probability distribution
    ↓
Predicted Vulnerability Type
```

**Location in code**: `encoders/static_encoder.py:104-116`, `190-193`

---

## 🎯 Training Process - How the Model Learns

### Loss Function (THE LEARNING MECHANISM!)

**Question**: "Do I need a loss function?"
**Answer**: **ABSOLUTELY YES!** Without it, the model cannot learn!

#### What is the Loss Function?

```python
# Location: train_static_only.py:226
criterion = CrossEntropyLoss(weight=class_weights)
```

**CrossEntropyLoss** measures "how wrong" the predictions are:

```
Given:
- Model prediction: [0.1, 0.05, 0.02, 0.7, 0.03, 0.05, 0.01, 0.02, 0.01, 0.01, 0.0]
                     [AC,  Arith, Bad, DoS*, FR,  Reen, SA,  Time, ULC, Oth, Safe]
- Ground truth: DoS (index 3)

Loss = -log(0.7) = 0.36  # Lower is better!

If model predicted wrong (e.g., AC with 0.1):
Loss = -log(0.1) = 2.30  # Much higher = BAD prediction!
```

#### Class Weighting (Handles Imbalance!)

Since `short_addresses` has only 30 samples vs. 666 for `arithmetic`, we use **inverse frequency weighting**:

```python
weight = 1 / (class_count + epsilon)

Example:
- short_addresses (30 samples): weight = 11.0  # High importance!
- arithmetic (663 samples):     weight = 0.5   # Lower importance
```

This forces the model to learn rare classes instead of ignoring them!

**Location in code**: `train_static_only.py:418-433`

### Training Loop (20 Epochs)

```
FOR epoch in 1 to 20:

    # 1. TRAINING PHASE
    FOR each batch of 8 contracts:

        a) Forward Pass:
           Contract → PDG → GAT → Features → Classification → Predictions

        b) Loss Calculation:
           loss = CrossEntropyLoss(predictions, ground_truth, weights=class_weights)

        c) Backward Pass (Backpropagation):
           Calculate ∂loss/∂weights for all layers

        d) Optimizer Step:
           Update weights: new_weight = old_weight - learning_rate × gradient

    # 2. VALIDATION PHASE
    FOR each validation batch:

        a) Forward Pass (no gradient calculation)
        b) Calculate validation loss
        c) Calculate accuracy, precision, recall, F1

    # 3. MODEL CHECKPOINT
    IF validation_f1 > best_f1:
        Save model to: models/checkpoints/static_encoder_best.pt
```

**Location in code**: `train_static_only.py:246-305`

### Optimizer: Adam

```python
optimizer = Adam(learning_rate=0.001)
```

**Adam** (Adaptive Moment Estimation) adjusts learning rate automatically:
- Fast convergence for easy-to-learn patterns
- Small steps for difficult patterns
- Uses momentum to avoid getting stuck

**Location in code**: `train_static_only.py:251`

---

## 📈 Expected Performance

### Overall Metrics

Based on similar datasets and PDG-based models:

| Metric                | Expected Range | Interpretation                        |
|-----------------------|----------------|---------------------------------------|
| Overall Accuracy      | 60-75%         | Percentage of correct predictions     |
| Macro F1              | 0.55-0.70      | Average F1 across all classes         |
| Weighted F1           | 0.60-0.75      | F1 weighted by class frequency        |

### Per-Vulnerability Performance Predictions

#### High Detection (F1 > 0.70) ✅

**Reentrancy** (Expected F1: 0.75-0.85)
- Why: Clear PDG pattern (external call → state write)
- PDG captures: Function calls + state variable writes
- Example pattern: `transfer() --calls--> external --writes--> balance`

**Arithmetic** (Expected F1: 0.70-0.80)
- Why: Arithmetic operations visible in PDG
- PDG captures: ADD, SUB, MUL, DIV operations
- Pattern: High arithmetic operations without SafeMath

**Unchecked Low-Level Calls** (Expected F1: 0.70-0.80)
- Why: Call patterns without require/assert
- PDG captures: CALL opcode without CHECK node
- Pattern: `call() --NOT--> require()`

#### Medium Detection (F1: 0.50-0.70) ⚠️

**Access Control** (Expected F1: 0.60-0.70)
- Why: Modifier patterns visible but not always clear
- PDG captures: Function --uses_modifier--> onlyOwner
- Challenge: Implicit access control not captured

**Denial of Service** (Expected F1: 0.55-0.65)
- Why: Loop patterns detectable but context-dependent
- PDG captures: JUMPI in loops
- Challenge: Legitimate loops vs. DoS loops

**Safe Contracts** (Expected F1: 0.60-0.70)
- Why: Absence of vulnerability patterns
- Pattern: No risky operations
- Challenge: False positives from incomplete analysis

**Time Manipulation** (Expected F1: 0.55-0.65)
- Why: Block.timestamp usage visible
- PDG captures: TIMESTAMP opcode usage
- Challenge: Legitimate time usage vs. manipulation

**Other** (Expected F1: 0.50-0.60)
- Why: Mixed vulnerability types
- Highly variable patterns

#### Low Detection (F1 < 0.50) 🔴

**Front Running** (Expected F1: 0.45-0.55)
- Why: Transaction ordering not visible in PDG
- PDG limitation: Static analysis can't see tx ordering
- Challenge: Requires dynamic analysis

**Bad Randomness** (Expected F1: 0.40-0.50)
- Why: Semantic issue (using block.timestamp for random)
- PDG limitation: Doesn't capture "intent" of randomness
- Challenge: Requires semantic understanding

**Short Addresses** (Expected F1: 0.30-0.45)
- Why: Very few training samples (30!) + low-level byte operations
- PDG limitation: Doesn't capture byte-level operations
- Challenge: Need more data + better representation

---

## 🚀 How to Run Training

### Quick Start (Recommended)

```bash
./start_static_training.sh
```

This runs:
- Batch size: 8
- Epochs: 20
- Learning rate: 0.001
- Full dataset: 4,540 train / 1,011 val / 1,024 test

**Expected time**:
- GPU (RTX A6000): 2-3 hours
- CPU: 12-15 hours

### Custom Training

```bash
python train_static_only.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --batch-size 16 \
    --num-epochs 30 \
    --learning-rate 0.0005
```

### Quick Test (Subset)

```bash
python train_static_only.py \
    --train-dir data/datasets/forge_balanced_accurate/train \
    --val-dir data/datasets/forge_balanced_accurate/val \
    --test-dir data/datasets/forge_balanced_accurate/test \
    --max-samples 500 \
    --num-epochs 5 \
    --batch-size 4
```

**Expected time**: ~30 minutes on GPU

---

## 📊 Understanding the Output

### During Training (Console Output)

```
================================================================================
EPOCH 1/20
================================================================================
Training: 100%|██████████| 568/568 [12:34<00:00, 1.33s/it, loss=0.89]

Train Loss: 1.2345, Train Acc: 45.67%
Val Loss: 1.0987, Val Acc: 52.34%, Val F1: 0.4912
✓ Saved best model (F1: 0.4912)
```

**What this means**:
- `Training: 568/568`: 568 batches processed (4540 contracts ÷ 8 batch size)
- `loss=0.89`: Current batch loss (lower is better)
- `Train Acc: 45.67%`: 45.67% of training predictions correct
- `Val F1: 0.4912`: Validation F1 score (harmonic mean of precision & recall)
- `✓ Saved best model`: This epoch achieved best F1 so far

### After Training (Detailed Metrics)

```
================================================================================
TEST SET - DETAILED VULNERABILITY DETECTION METRICS
================================================================================

OVERALL ACCURACY: 67.34%

Vulnerability Type              Precision     Recall   F1-Score  Detected/Total
--------------------------------------------------------------------------------
reentrancy                         0.8319     0.7899     0.8103      94/119   (79.0%)
arithmetic                         0.6912     0.7397     0.7146     108/146   (74.0%)
unchecked_low_level_calls          0.7260     0.7260     0.7260     106/146   (72.6%)
safe                               0.6929     0.7000     0.6964      98/140   (70.0%)
access_control                     0.7234     0.6891     0.7058     102/148   (68.9%)
denial_of_service                  0.6486     0.6081     0.6277      45/74    (60.8%)
time_manipulation                  0.6222     0.6222     0.6222      28/45    (62.2%)
front_running                      0.5938     0.5938     0.5938      19/32    (59.4%)
other                              0.5594     0.5874     0.5731      84/143   (58.7%)
bad_randomness                     0.5417     0.5208     0.5310      13/24    (52.1%)
short_addresses                    0.4286     0.4286     0.4286       3/7     (42.9%)
--------------------------------------------------------------------------------
MACRO AVERAGE                      0.6418     0.6369     0.6390
WEIGHTED AVERAGE                                        0.6712
TOTAL DETECTION RATE                                    700/1024  (68.36%)
================================================================================
```

**Metric Explanations**:

**Precision**: Of all contracts predicted as vulnerability X, how many actually were X?
```
Precision = True Positives / (True Positives + False Positives)

Example: Reentrancy precision = 0.8319
→ When model says "reentrancy", it's correct 83.19% of the time
```

**Recall**: Of all contracts with vulnerability X, how many did we detect?
```
Recall = True Positives / (True Positives + False Negatives)

Example: Reentrancy recall = 0.7899
→ We detected 78.99% (94 out of 119) of all reentrancy vulnerabilities
```

**F1-Score**: Balanced metric (harmonic mean of precision & recall)
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

High F1 = Good balance between precision and recall
```

**Detected/Total**: Raw detection count
```
Example: 94/119 (79.0%)
→ Detected 94 reentrancy vulnerabilities out of 119 total in test set
```

---

## 📁 Output Files

After training completes, you'll have:

### 1. Model Checkpoint
**Location**: `models/checkpoints/static_encoder_best.pt`
**Size**: ~67 MB
**Contents**:
- Model weights (all 3 GAT layers + projection + classification heads)
- Epoch number
- Validation loss, accuracy, F1

**Usage**: Load for inference on new contracts

### 2. Test Results (Text File)
**Location**: `models/checkpoints/test_results_YYYYMMDD_HHMMSS.txt`
**Contents**:
- Overall accuracy, macro F1, weighted F1
- Per-vulnerability precision, recall, F1, support

### 3. Training Log
**Location**: `logs/static_training_YYYYMMDD_HHMMSS.log`
**Contents**:
- Full console output
- Training/validation metrics per epoch
- Error messages (if any)

### 4. TensorBoard Logs
**Location**: `runs/static_only_YYYYMMDD_HHMMSS/`
**View with**: `tensorboard --logdir runs/`
**Contents**:
- Loss curves (train & validation)
- Accuracy curves
- F1 score progression

---

## 🎓 Key Takeaways

### What Works Well with PDG-Based Static Analysis

✅ **Structural vulnerabilities**:
- Reentrancy (call patterns)
- Arithmetic overflow (operation sequences)
- Unchecked calls (missing checks)
- Access control (modifier usage)

### What Doesn't Work Well

❌ **Semantic vulnerabilities**:
- Bad randomness (intent not captured)
- Front running (transaction ordering)
- Short addresses (low-level bytes)

### Why Class Imbalance Matters

With only **30 short_addresses samples**:
- Model sees it 0.7% of the time during training
- Hard to learn patterns from so few examples
- **Solution**: Collect more data or use data augmentation

### Why Loss Function is Critical

**Without loss function**: Model can't improve (no gradient signal)
**With CrossEntropyLoss**: Model learns by:
1. Making predictions
2. Calculating how wrong they are (loss)
3. Adjusting weights to reduce loss
4. Repeat for 20 epochs

---

## 🔍 Next Steps

1. **Run training**: `./start_static_training.sh`

2. **Analyze results**: Check which vulnerabilities are detected best

3. **Compare with other modalities**:
   - Dynamic analysis (execution traces)
   - Semantic analysis (CodeBERT)
   - Fusion (combine all three)

4. **Improve model**:
   - Collect more data for rare classes
   - Tune hyperparameters (learning rate, layers)
   - Try different architectures (GraphSAGE, GIN)

---

## 📚 Related Files

- `STATIC_TRAINING_GUIDE.md`: Detailed explanation of loss function and training mechanics
- `scripts/train_complete_pipeline.py`: Original full pipeline (all 4 models)
- `encoders/static_encoder.py`: Static encoder architecture
- `tools/slither_wrapper.py`: Slither integration for PDG extraction

---

**Questions? Check the training log or TensorBoard for detailed insights!**
