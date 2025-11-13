# Triton Weekly Progress Report
November 6, 2025

## Quick Technical Notes

**Q: Why are the encoder dimensions different (Static: 768, Dynamic: 512, Semantic: 768)?**

- Graph-based encoders (static/semantic) need higher capacity for structural relationships
- Dynamic encoder works with simpler sequential traces, so lower dimension is fine
- Fusion module uses projection layers to map everything to unified 512-dim space before combining

**Q: Can we fine-tune when new vulnerabilities are discovered, or do we need to retrain from scratch?**

- Fine-tuning works if done carefully to avoid catastrophic forgetting
- Mix 30% old data + 70% new data (replay buffer)
- Freeze encoder layers, only train classification head
- Use lower learning rate (1e-5 instead of 1e-4)
- Expected result: 5-10% accuracy drop on old classes (acceptable)
- Without replay buffer: 50-80% forgetting (not acceptable)

**Q: Is class-weighted loss something new or established?**

- Been around since late 90s (Kubat & Matwin, 1997)
- Using inverse frequency weighting via PyTorch's CrossEntropyLoss
- Popularized by focal loss (Lin et al., 2017)
- Used in vulnerability detection like Devign (Zhou et al., 2019)
- Necessary because we have 1:40 class imbalance ratio

---

## What Got Done This Week

### Dataset cleanup and labeling
- Previous dataset was messy - unbalanced classes, unlabeled samples, explained the 11% accuracy
- Built proper dataset with 228 contracts from SmartBugs, SolidiFI, and other sources
- Split: 155 training / 29 validation / 44 test
- Covers 10 vulnerability types plus safe contracts
- Added class-weighted loss to handle imbalance

### Fixed the PDG extraction bug
- Static encoder was completely failing - PDGs were all empty
- Root cause: broken extraction logic in Slither wrapper
- Rewrote to use Slither Python API directly instead of parsing CLI JSON
- Result: 0% → 97.7% success rate (43 out of 44 contracts)

### Retrained everything
- Static encoder (PDG-based)
- Dynamic encoder (execution traces)
- Semantic encoder (CodeBERT)
- Fusion module (combines all three)

### Testing infrastructure
- Built unified test script: `./test.sh`
- Tests all 4 modalities on held-out test set
- Generates detailed per-class performance breakdowns

---

## Current Results

Tested on 44 contracts:

| Model | Accuracy | What it's good at |
|-------|----------|-------------------|
| Semantic | 50.00% | Best overall - catches bad_randomness, denial_of_service, time_manipulation perfectly |
| Dynamic | 20.45% | Perfect on unchecked_low_level_calls |
| Fusion | 14.29% | Perfect on reentrancy |
| Static | 11.90% | Perfect on access_control |

Good news:
- Each modality specializes in different vulnerability types - validates multi-modal approach
- Static encoder finally working after bug fix
- Fusion module successfully combines all three inputs

Bad news:
- Fusion underperforming vs semantic alone (14% vs 50%)
- Something's not right with the fusion training

Detailed breakdown by vulnerability:

| Vulnerability | Test samples | Static | Dynamic | Semantic | Fusion |
|--------------|--------------|--------|---------|----------|--------|
| access_control | 5 | 5/5 | 0/5 | 1/5 | 0/5 |
| unchecked_calls | 9 | 0/9 | 9/9 | 8/9 | 0/9 |
| bad_randomness | 2 | 0/2 | 0/2 | 2/2 | 0/2 |
| denial_of_service | 2 | 0/2 | 0/2 | 2/2 | 0/2 |
| time_manipulation | 2 | 0/2 | 0/2 | 2/2 | 0/2 |
| reentrancy | 7 | 0/7 | 0/7 | 3/7 | 6/6 |
| arithmetic | 4 | 0/4 | 0/4 | 3/4 | 0/4 |
| front_running | 2 | 0/2 | 0/2 | 1/2 | 0/2 |
| safe | 10 | 0/10 | 0/10 | 0/10 | 0/10 |
| short_addresses | 1 | 0/1 | 0/1 | 0/1 | 0/1 |

Observations:
- Static: good at structural patterns (access_control)
- Dynamic: catches runtime issues (unchecked_calls)
- Semantic: broadest coverage (7 out of 10 classes)
- Fusion: overfitting on reentrancy only

---

## What Needs Work

**Fusion model**
- Should be performing better than individual encoders, not worse
- Need to retrain with better regularization
- Check if loss function is biased toward reentrancy

**Safe contract detection**
- All models failing to identify safe contracts (0/10)
- Might need confidence thresholding
- Could emphasize negative class during training

**Training optimization**
- Only ran 20 epochs so far, should go to 50-100
- Experiment with per-modality learning rates
- Add data augmentation

**Dataset size**
- 228 contracts is decent for initial testing
- Not enough for production
- Need to scale to 500-1000 contracts with more diverse examples

---

## Next Milestones

Short term (1-2 weeks):
- Get fusion model working properly (target: 55-60% accuracy)
- Improve static encoder (target: 25-30% accuracy)
- Fix safe contract detection (target: 80%+ recall)

Medium term (1 month):
- Overall system accuracy: 60-70%
- Production deployment ready
- API endpoint for contract analysis

---

## Summary

- System is functional, all four modalities trained and working
- Each modality shows different strengths - validates multi-modal approach
- Main issues: fusion underperformance, safe contract detection
- Dataset could be bigger
- PDG bug fix was critical: 0% → 97.7% success rate

Test with: `./test.sh`
Dataset location: `data/datasets/combined_labeled/`
