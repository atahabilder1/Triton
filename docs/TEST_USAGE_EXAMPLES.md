# Unified Testing Script Usage

The `test_modality.py` script provides a single interface for testing all modalities.

## Usage

```bash
python3 test_modality.py --modality <MODALITY> [OPTIONS]
```

### Parameters

- `--modality`: **Required**. Choose one of:
  - `static` - Test static encoder (PDG-based)
  - `dynamic` - Test dynamic encoder (execution traces)
  - `semantic` - Test semantic encoder (CodeBERT)
  - `fusion` - Test fusion model (all encoders combined)

- `--test-dir`: Test dataset directory (default: `data/datasets/combined_labeled/test`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)

---

## Examples

### Test Static Encoder
```bash
python3 test_modality.py --modality static --test-dir data/datasets/combined_labeled/test
```

### Test Dynamic Encoder
```bash
python3 test_modality.py --modality dynamic --test-dir data/datasets/combined_labeled/test
```

### Test Semantic Encoder
```bash
python3 test_modality.py --modality semantic --test-dir data/datasets/combined_labeled/test
```

### Test Fusion Model
```bash
python3 test_modality.py --modality fusion --test-dir data/datasets/combined_labeled/test
```

### Test on CPU
```bash
python3 test_modality.py --modality semantic --device cpu
```

---

## Output Format

The script provides consistent output for all modalities:

```
================================================================================
RESULTS: SEMANTIC MODEL
================================================================================

Overall Performance:
  Success Rate: 44/44 (100.0%)
  Accuracy: 22/44 (50.00%)
  Macro Precision: 0.4523
  Macro Recall: 0.4812
  Macro F1: 0.5012

Per-Class Performance:
────────────────────────────────────────────────────────────────────────────────
Vulnerability Type              Precision     Recall         F1  Support
────────────────────────────────────────────────────────────────────────────────
access_control                      0.286      0.400      0.333        5
arithmetic                          0.667      0.500      0.571        4
bad_randomness                      1.000      1.000      1.000        2
denial_of_service                   1.000      1.000      1.000        2
front_running                       0.333      0.500      0.400        2
reentrancy                          0.500      0.571      0.533        7
short_addresses                     0.000      0.000      0.000        1
time_manipulation                   1.000      1.000      1.000        2
unchecked_low_level_calls           0.667      0.667      0.667        9
safe                                0.000      0.000      0.000       10
────────────────────────────────────────────────────────────────────────────────
MACRO AVERAGE                       0.545      0.564      0.550       44
────────────────────────────────────────────────────────────────────────────────
```

---

## Quick Test All Modalities

Run all four tests sequentially:

```bash
for modality in static dynamic semantic fusion; do
    echo "Testing $modality..."
    python3 test_modality.py --modality $modality
    echo ""
done
```

---

## What Each Modality Tests

### Static (`--modality static`)
- Uses Slither for PDG extraction
- Graph Attention Network (GAT) encoding
- Detects structural patterns and control flow
- Best for: access_control vulnerabilities

### Dynamic (`--modality dynamic`)
- Uses Mythril for execution trace analysis
- LSTM encoding of runtime behavior
- Detects execution-based vulnerabilities
- Best for: unchecked_low_level_calls

### Semantic (`--modality semantic`)
- Uses CodeBERT pre-trained model
- Understands code semantics and patterns
- Detects intent-based vulnerabilities
- Best overall performance (50% accuracy)

### Fusion (`--modality fusion`)
- Combines all three encoders
- Cross-modal attention fusion
- Expected: 55-65% accuracy
- Leverages strengths of all modalities

---

## Expected Performance (Nov 6, 2025)

| Modality | Success Rate | Accuracy | Avg F1 | Best For |
|----------|-------------|----------|--------|----------|
| Static   | 95.5%       | 11.90%   | 0.021  | access_control |
| Dynamic  | 100%        | 20.45%   | 0.034  | unchecked_calls |
| Semantic | 100%        | 50.00%   | 0.501  | Most vulnerabilities |
| Fusion   | TBD         | 55-65%*  | TBD    | All (expected) |

*Fusion performance not yet tested with this script
