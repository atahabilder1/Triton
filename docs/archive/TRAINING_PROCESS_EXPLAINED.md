# 🎓 Triton Training Process Explained

## 📝 Overview

**Total Training Phases: 4**

The training happens in **sequential order** with each phase building on the previous ones:

```
1. Static Encoder   (CFG analysis)
        ↓
2. Dynamic Encoder  (Execution traces)
        ↓
3. Semantic Encoder (Code understanding)
        ↓
4. Fusion Module    (Combine all three)
```

---

## 🔍 Detailed Process

### **Script Location:**
```
/home/anik/code/Triton/scripts/train_complete_pipeline.py
```

This ONE script handles ALL 4 training phases!

---

## 📊 Training Order & Details

### **PHASE 1: Static Encoder** (Lines 421-497)

**What it does:**
- Trains Graph Attention Network (GAT) on Control Flow Graphs (CFG)
- Learns patterns from program structure
- Uses Slither to extract CFGs

**Function:** `train_static_encoder()`

**Input:** PDG (Program Dependence Graph) from Slither

**Process:**
```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        pdgs = batch['pdg']  # Control flow graphs
        labels = batch['label']

        # Forward pass
        static_features, vuln_scores = static_encoder(pdgs)

        # Calculate loss
        loss = criterion(vuln_scores, labels)

        # Update weights
        loss.backward()
        optimizer.step()

    # Validation phase
    val_loss, val_acc = validate_static(val_loader)

    # Save best model
    if val_loss < best_val_loss:
        save_model("static_encoder_best.pt")
```

**Output:**
- `models/checkpoints/static_encoder_best.pt` (22 MB)

**Training Time:** ~15-30 minutes (155 contracts, 10 epochs)

---

### **PHASE 2: Dynamic Encoder** (Lines 499-582)

**What it does:**
- Trains LSTM on execution traces
- Learns patterns from runtime behavior
- Uses Mythril to generate execution traces

**Function:** `train_dynamic_encoder()`

**Input:** Execution traces from Mythril

**Process:**
```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        traces = batch['execution_traces']  # Runtime traces
        labels = batch['label']

        # Forward pass
        dynamic_features, vuln_scores = dynamic_encoder(traces)

        # Calculate loss
        loss = criterion(vuln_scores, labels)

        # Update weights
        loss.backward()
        optimizer.step()

    # Validation phase
    val_loss, val_acc = validate_dynamic(val_loader)

    # Save best model
    if val_loss < best_val_loss:
        save_model("dynamic_encoder_best.pt")
```

**Output:**
- `models/checkpoints/dynamic_encoder_best.pt` (30 MB)

**Training Time:** ~15-30 minutes (155 contracts, 10 epochs)

---

### **PHASE 3: Semantic Encoder** (Lines 584-663)

**What it does:**
- Fine-tunes pre-trained CodeBERT
- Learns semantic meaning of code
- Understands variable names, function purposes

**Function:** `train_semantic_encoder()`

**Input:** Source code (raw Solidity)

**Process:**
```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        source_codes = batch['source_code']  # Raw code
        labels = batch['label']

        # Forward pass (fine-tuning CodeBERT)
        semantic_features, vuln_scores = semantic_encoder(source_codes)

        # Calculate loss
        loss = criterion(vuln_scores, labels)

        # Update weights (lower learning rate for fine-tuning)
        loss.backward()
        optimizer.step()  # LR = 0.0001 (10x lower)

    # Validation phase
    val_loss, val_acc = validate_semantic(val_loader)

    # Save best model
    if val_loss < best_val_loss:
        save_model("semantic_encoder_best.pt")
```

**Output:**
- `models/checkpoints/semantic_encoder_best.pt` (517 MB)

**Training Time:** ~15-30 minutes (155 contracts, 10 epochs)

---

### **PHASE 4: Fusion Module** (Lines 665-753)

**What it does:**
- Combines all three encoders
- Learns how to weight each modality
- Trains end-to-end (all components together)

**Function:** `train_fusion_module()`

**Input:** PDGs + Execution Traces + Source Code (all three!)

**Process:**
```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        source_codes = batch['source_code']
        pdgs = batch['pdg']
        traces = batch['execution_traces']
        labels = batch['label']

        # Get features from ALL encoders
        static_features = static_encoder(pdgs)
        dynamic_features = dynamic_encoder(traces)
        semantic_features = semantic_encoder(source_codes)

        # FUSION - Combine all three
        fusion_output = fusion_module(
            static_features,
            dynamic_features,
            semantic_features
        )

        vulnerability_logits = fusion_output['vulnerability_logits']

        # Calculate loss
        loss = criterion(vulnerability_logits, labels)

        # Update ALL components (end-to-end)
        loss.backward()
        optimizer.step()  # Updates all 4 models!

    # Validation phase
    val_loss, val_acc = validate_fusion(val_loader)

    # Save best models (all 4!)
    if val_loss < best_val_loss:
        save_model("static_encoder_fusion_best.pt")
        save_model("dynamic_encoder_fusion_best.pt")
        save_model("semantic_encoder_fusion_best.pt")
        save_model("fusion_module_best.pt")
```

**Output:**
- `models/checkpoints/static_encoder_fusion_best.pt`
- `models/checkpoints/dynamic_encoder_fusion_best.pt`
- `models/checkpoints/semantic_encoder_fusion_best.pt`
- `models/checkpoints/fusion_module_best.pt` (39 MB)

**Training Time:** ~20-40 minutes (155 contracts, 10 epochs)

---

## 📁 Where is Each Component Defined?

### **1. Static Encoder**
- **File:** `/home/anik/code/Triton/encoders/static_encoder.py`
- **Architecture:** GAT (Graph Attention Network)
- **Input:** Control Flow Graphs (PDGs)
- **Output:** 768-dim features + vulnerability scores

### **2. Dynamic Encoder**
- **File:** `/home/anik/code/Triton/encoders/dynamic_encoder.py`
- **Architecture:** LSTM (Long Short-Term Memory)
- **Input:** Execution traces
- **Output:** 512-dim features + vulnerability scores

### **3. Semantic Encoder**
- **File:** `/home/anik/code/Triton/encoders/semantic_encoder.py`
- **Architecture:** CodeBERT (pre-trained transformer)
- **Input:** Source code
- **Output:** 768-dim features + vulnerability scores

### **4. Fusion Module**
- **File:** `/home/anik/code/Triton/fusion/cross_modal_fusion.py`
- **Architecture:** Attention-based fusion
- **Input:** Static + Dynamic + Semantic features
- **Output:** Combined 768-dim features + final predictions

### **5. Analysis Tools**
- **Slither Wrapper:** `/home/anik/code/Triton/tools/slither_wrapper.py`
- **Mythril Wrapper:** `/home/anik/code/Triton/tools/mythril_wrapper.py`

---

## ⏱️ Total Training Time

| Phase | Time | Contracts | Epochs |
|-------|------|-----------|--------|
| **Phase 1: Static** | 15-30 min | 155 | 10 |
| **Phase 2: Dynamic** | 15-30 min | 155 | 10 |
| **Phase 3: Semantic** | 15-30 min | 155 | 10 |
| **Phase 4: Fusion** | 20-40 min | 155 | 10 |
| **TOTAL** | **65-130 min** | **155** | **10** |

**Estimated:** 1-2 hours for complete training

---

## 🎯 Why This Order?

### **1. Individual Training First (Phases 1-3)**
Each encoder is trained independently to learn:
- Static: CFG patterns
- Dynamic: Execution patterns
- Semantic: Code semantics

**Benefit:** Each encoder becomes good at its own modality

### **2. Fusion Training Last (Phase 4)**
After encoders are trained, fusion module learns:
- How to combine them
- Which modality to trust for which vulnerability type
- End-to-end optimization

**Benefit:** Best overall performance

---

## 📊 What Gets Saved?

After training completes, you'll have:

```
models/checkpoints/
├── static_encoder_best.pt              (22 MB)  - Best from Phase 1
├── dynamic_encoder_best.pt             (30 MB)  - Best from Phase 2
├── semantic_encoder_best.pt            (517 MB) - Best from Phase 3
├── static_encoder_fusion_best.pt       (22 MB)  - Updated in Phase 4
├── dynamic_encoder_fusion_best.pt      (30 MB)  - Updated in Phase 4
├── semantic_encoder_fusion_best.pt     (517 MB) - Updated in Phase 4
└── fusion_module_best.pt               (39 MB)  - From Phase 4
```

**Total:** ~1.2 GB of model weights

---

## 🚀 How to Run Training

### **Train ALL phases (recommended):**
```bash
python scripts/train_complete_pipeline.py \
    --train-dir data/datasets/combined_labeled/train \
    --num-epochs 20 \
    --batch-size 4 \
    --train-mode all
```

### **Train only specific phase:**
```bash
# Only static encoder
python scripts/train_complete_pipeline.py \
    --train-mode static

# Only dynamic encoder
python scripts/train_complete_pipeline.py \
    --train-mode dynamic

# Only semantic encoder
python scripts/train_complete_pipeline.py \
    --train-mode semantic

# Only fusion module (requires pre-trained encoders)
python scripts/train_complete_pipeline.py \
    --train-mode fusion
```

---

## 🔍 Validation Usage

**Every phase uses validation:**

```python
# Phase 1: Static Encoder
for epoch in range(10):
    train_on_train_set()          # 155 contracts
    val_loss = validate_on_val()  # 29 contracts
    if val_loss improved:
        save_model()
    else:
        patience_counter += 1

    if patience_counter >= 5:
        early_stopping()

# Same for Phases 2, 3, 4
```

**Validation prevents overfitting in each phase!**

---

## 📈 Expected Output During Training

```
================================================================================
COMPLETE TRITON TRAINING PIPELINE
================================================================================
Training directory: data/datasets/combined_labeled/train
Batch size: 4
Epochs: 20
Training samples: 155
Validation samples: 29

================================================================================
PHASE 1: Training Static Encoder
================================================================================

Epoch 1/20
Training Static Encoder: 100%|████████| 39/39 [00:45<00:00, loss=2.34]
Train Loss: 2.3421, Train Acc: 25.60%
Val Loss: 2.2150, Val Acc: 31.03%
✓ Saved best static encoder (val_loss: 2.2150)

Epoch 2/20
Training Static Encoder: 100%|████████| 39/39 [00:43<00:00, loss=2.10]
Train Loss: 2.1052, Train Acc: 32.10%
Val Loss: 2.0899, Val Acc: 34.48%
✓ Saved best static encoder (val_loss: 2.0899)

... (continues for 20 epochs)

Static Encoder training complete! Best val_loss: 1.8234

================================================================================
PHASE 2: Training Dynamic Encoder
================================================================================

... (similar output)

================================================================================
PHASE 3: Training Semantic Encoder
================================================================================

... (similar output)

================================================================================
PHASE 4: Training Fusion Module End-to-End
================================================================================

... (similar output)

================================================================================
TRAINING COMPLETE!
================================================================================

Model checkpoints saved to: models/checkpoints
```

---

## 🎯 Summary

- **Script:** `scripts/train_complete_pipeline.py` (ONE file for everything!)
- **Phases:** 4 (Static → Dynamic → Semantic → Fusion)
- **Order:** Sequential (each builds on previous)
- **Validation:** Used in every phase (prevents overfitting)
- **Time:** 1-2 hours total
- **Output:** 7 model files (~1.2 GB)

**Ready to train!** 🚀
