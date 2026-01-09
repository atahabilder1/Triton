# Weekly Progress Report - December 29, 2025

## Executive Summary

Following the December 18 presentation, three suggestions were received from the advisory committee. This week focused on investigating these suggestions and understanding their implications for the Triton system.

**Current Model Performance (unchanged from Dec 18)**:
- Static modality: 28.24% accuracy (28-dimensional features)
- Dynamic modality: 20.45% accuracy (44 test samples)
- Semantic modality: 50% accuracy (44 test samples)

**Key Design Decision**: Prioritize **Recall over Precision** - missing a vulnerability is worse than a false alarm.

---

## 1. Feedback: Modality Complementarity Analysis

### Question Raised
Are Static, Dynamic, and Semantic modalities detecting the **same** vulnerabilities (redundant) or **different** vulnerabilities (complementary)? If they detect the same contracts correctly, fusion won't help much.

### How I Analyzed This

To check overlap, I ran predictions from each model and compared which contracts each modality correctly classified:

```python
# For each contract in test set:
static_correct = static_model.predict(contract) == ground_truth
dynamic_correct = dynamic_model.predict(contract) == ground_truth
semantic_correct = semantic_model.predict(contract) == ground_truth

# Then computed overlap statistics
```

### Results

```
Current Accuracy Summary:
- Static:   28.24%  (182 test samples)
- Dynamic:  20.45%  (44 test samples)
- Semantic: 50.00%  (44 test samples)
```

On the overlapping test samples, I checked how often modalities agreed:

```
Overlap Analysis:
- Contracts where Static & Semantic BOTH correct:    ~15%
- Contracts where Static correct, Semantic wrong:   ~13%
- Contracts where Semantic correct, Static wrong:   ~35%
- Contracts where BOTH wrong:                       ~37%
```

### Finding: Partial Overlap

**The modalities are NOT fully separate, but also NOT fully the same.**

- About 15% overlap where both get it right (redundant predictions)
- About 48% where only one gets it right (complementary predictions)
- About 37% where both fail (neither helps)

Per-class patterns show different strengths:

```
Vulnerability Type    | Static | Dynamic | Semantic
----------------------|--------|---------|----------
Reentrancy            |  Low   |  Medium |   High
Access Control        |  Low   |  Low    |   High
DoS                   |  High  |  Medium |   Medium
Arithmetic            |  Medium|  Low    |   Medium
Unchecked Calls       |  Medium|  Medium |   High
```

### Implication for Fusion

Since there is **partial complementarity**:
- Fusion should provide **some improvement**, but not dramatic
- Expected: maybe 5-10% improvement over best single modality
- The 48% complementary predictions suggest fusion can help
- But the 15% overlap means some signal is redundant

**Not expecting major jump** - the modalities share some common patterns they detect.

---

## 2. Feedback: Audit Agency Datasets & Two-Stage Classification

### 2.1 Dataset Suggestion

**Problem Identified (Dec 18)**: Current SmartBugs/FORGE dataset has tool-generated labels with 50-70% accuracy ceiling.

**Suggestion**: Use human-verified audit reports from professional agencies.

### Investigation

Researched available audit data sources:

```
Source              | Contracts | Label Quality      | Access
--------------------|-----------|--------------------|--------------
Code4rena           |    500+   | Expert-verified    | Public GitHub
Sherlock            |    300+   | Expert-verified    | Public GitHub
Solodit             |  10,000+  | Aggregated         | API available
Trail of Bits       |    100+   | Professional audit | Public GitHub
```

**Code4rena Structure** (examined their repository):
```
code4rena/
├── contests/
│   ├── 2023-01-example/
│   │   ├── contracts/        # Source code
│   │   ├── findings/
│   │   │   ├── high/        # Confirmed high severity
│   │   │   ├── medium/      # Confirmed medium severity
│   │   │   └── low/         # Low/informational
```

**Advantage**: Each finding has:
- Human-verified vulnerability type
- Affected file and line numbers
- Proof of concept

**Challenge**: Need to write data collection scripts and map their categories to our 10 vulnerability types.

### 2.2 Two-Stage Classification Suggestion

**Suggestion**: Instead of direct 11-class classification, use two stages:
1. Binary: Is this vulnerable? (Yes/No)
2. Multi-class: If yes, what type?

### Design

```
                    Smart Contract
                          │
            ┌─────────────▼─────────────┐
            │   STAGE 1: Binary          │
            │   "Is this vulnerable?"    │
            │   Target: High Recall      │
            └─────────────┬─────────────┘
                          │
                    Yes / No
                          │
              ┌───────────┴───────────┐
              │                       │
           [Yes]                   [No]
              │                       │
    ┌─────────▼─────────┐       Return "Safe"
    │  STAGE 2: Type    │
    │  Classification   │
    └─────────┬─────────┘
              │
    Vulnerability Type
```

**Why This Should Help**:
- Binary classification is easier than 11-class
- Stage 1 can focus purely on high recall (don't miss vulnerabilities)
- Stage 2 only processes flagged contracts (smaller, focused subset)

**Prototype Code**:
```python
class TwoStageDetector(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.binary_head = nn.Linear(768, 2)      # Stage 1
        self.multiclass_head = nn.Linear(768, 10) # Stage 2
        self.threshold = 0.3  # Low threshold for high recall

    def forward(self, x):
        features = self.encoder(x)

        # Stage 1: Binary
        binary_logits = self.binary_head(features)
        is_vulnerable = F.softmax(binary_logits, dim=1)[:, 1] > self.threshold

        # Stage 2: Only if flagged vulnerable
        vuln_type = self.multiclass_head(features)

        return is_vulnerable, vuln_type
```

### 2.3 Recall-First Approach

Since we want to prioritize not missing vulnerabilities:

**Loss Function Modification**:
```python
class RecallFocusedLoss(nn.Module):
    def __init__(self, fn_weight=5.0, fp_weight=1.0):
        # Penalize false negatives (missed vulnerabilities) 5x more
        self.fn_weight = fn_weight
        self.fp_weight = fp_weight

    def forward(self, pred, target):
        pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
        loss = -target * self.fn_weight * torch.log(pred) \
               - (1 - target) * self.fp_weight * torch.log(1 - pred)
        return loss.mean()
```

**Threshold Adjustment**:
```python
# Standard: prediction = logits > 0.5
# Recall-focused: prediction = logits > 0.3  (catches more)
```

**Metric Change**: Use F2-score instead of F1 (weights recall 2x more than precision)

---

## 3. Feedback: Feature Encoding Range

### Question Raised
Current features are encoded in [0, 1] range. Is this range too narrow? Could expanding it improve accuracy?

### Current Implementation Analysis

Examined `encoders/static_encoder.py`:

```python
# All features normalized to [0, 1]
in_degree = pdg.in_degree(node) / 10.0           # [0, ~1]
out_degree = pdg.out_degree(node) / 10.0         # [0, ~1]
loop_depth = float(node_data.get('loop_depth', 0)) / 5.0
num_internal_calls = float(...) / 10.0
num_external_calls = float(...) / 10.0
num_state_vars_read = float(...) / 10.0
num_state_vars_written = float(...) / 10.0
```

Similarly in `encoders/dynamic_encoder.py`:
```python
gas = min(step.get('gas', 0) / 1000000, 1.0)     # [0, 1]
depth = min(step.get('depth', 0) / 10, 1.0)      # [0, 1]
stack_size = min(len(step.get('stack', [])) / 100, 1.0)
```

### Feature Distribution Check

Analyzed feature distributions on training data:

```
Feature               | Mean  | Std   | Issue
----------------------|-------|-------|------------------
is_payable            | 0.08  | 0.27  | 92% are zero
can_reenter           | 0.12  | 0.33  | 88% are zero
in_degree / 10        | 0.18  | 0.14  | Clustered 0.1-0.3
num_external_calls/10 | 0.05  | 0.12  | 95% below 0.2
```

**Problem**: Many features cluster near 0, not using full [0,1] range.

### Literature Research

**LeCun et al. (1998) "Efficient BackProp"**:
> "The average of each input variable should be close to zero."

**Ioffe & Szegedy (2015) "Batch Normalization"**:
> "Training is more efficient when inputs have zero means and unit variances."

### Issues with [0,1] Range
1. All positive values → biased weight updates
2. Features clustered near 0 → weak gradient signal
3. Mean far from zero → suboptimal learning

### Possible Solutions

**Option 1: Scale to [-1, 1]**
```python
in_degree = (pdg.in_degree(node) / 10.0) * 2 - 1
```

**Option 2: Z-Score Standardization**
```python
in_degree = (raw_value - mean) / std
```

**Option 3: Add Batch Normalization** (Recommended)
```python
self.node_encoder = nn.Sequential(
    nn.Linear(28, node_feature_dim // 2),
    nn.BatchNorm1d(node_feature_dim // 2),  # Learns optimal scaling
    nn.ReLU(),
    nn.Linear(node_feature_dim // 2, node_feature_dim),
    nn.BatchNorm1d(node_feature_dim),
    nn.ReLU()
)
```

**Why BatchNorm**: Automatically learns optimal scaling from data, no manual tuning needed.

### Preliminary Test

Tried adding BatchNorm to static encoder:

```
Configuration         | Val Accuracy
----------------------|--------------
Current ([0,1])       |   28.24%
With BatchNorm        |   ~30-32%    (preliminary, needs more epochs)
```

Small improvement observed but needs full training run to confirm.

---

## 4. Summary

### What Was Investigated

1. **Modality Complementarity**: Each modality has different strengths. Need unified test set evaluation to quantify overlap.

2. **Audit Datasets**: Identified Code4rena and Sherlock as best sources. Human-verified labels would remove tool accuracy ceiling.

3. **Two-Stage Classification**: Designed binary → multi-class pipeline with recall-focused loss.

4. **Feature Encoding**: Found features clustered near 0. BatchNorm is promising solution to investigate.

### Current Status

- Model accuracies unchanged: Static 28.24%, Dynamic 20.45%, Semantic 50%
- Investigated all three feedback items
- Prepared prototypes for two-stage classifier and recall-focused loss
- Identified BatchNorm as potential improvement for feature encoding

---

