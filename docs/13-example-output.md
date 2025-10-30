# Example Triton Test Output

This file shows you exactly what the output will look like when you run the tests.

---

## Console Output (What you'll see in terminal)

```
2025-10-30 15:30:00 - INFO - Testing on SmartBugs Curated dataset...
2025-10-30 15:30:05 - INFO - Testing access_control contracts...
2025-10-30 15:32:15 - INFO - Testing arithmetic contracts...
2025-10-30 15:34:20 - INFO - Testing bad_randomness contracts...
2025-10-30 15:35:30 - INFO - Testing denial_of_service contracts...
2025-10-30 15:36:40 - INFO - Testing front_running contracts...
2025-10-30 15:37:50 - INFO - Testing reentrancy contracts...
2025-10-30 15:45:10 - INFO - Testing time_manipulation contracts...
2025-10-30 15:46:20 - INFO - Testing unchecked_low_level_calls contracts...
2025-10-30 15:52:30 - INFO - Testing other contracts...
2025-10-30 15:53:40 - INFO - SmartBugs testing complete: 143/143 contracts analyzed
2025-10-30 15:53:45 - INFO - Generating test report...
2025-10-30 15:53:47 - INFO - Detailed results saved to results/smartbugs/triton_test_results_20251030_153347.json
2025-10-30 15:53:48 - INFO - Summary report saved to results/smartbugs/triton_test_summary_20251030_153347.txt
2025-10-30 15:53:49 - INFO - Markdown table saved to results/smartbugs/triton_results_table_20251030_153347.md

====================================================================================================
TRITON VULNERABILITY DETECTION - SUMMARY
====================================================================================================

Overall Metrics:
  Average Precision: 0.9250
  Average Recall: 0.9100
  Average F1: 0.9175
  Average Analysis Time: 2.3500
  Total Contracts Tested: 143

SMARTBUGS Dataset - Vulnerability Detection:
----------------------------------------------------------------------------------------------------
Vulnerability Type              | Total    | Detected   | Missed   | Detection %
----------------------------------------------------------------------------------------------------
Access Control                  | 18       | 17         | 1        |      94.44%
Arithmetic                      | 15       | 14         | 1        |      93.33%
Bad Randomness                  | 8        | 7          | 1        |      87.50%
Denial Of Service               | 6        | 6          | 0        |     100.00%
Front Running                   | 4        | 3          | 1        |      75.00%
Other                           | 3        | 2          | 1        |      66.67%
Reentrancy                      | 31       | 30         | 1        |      96.77%
Time Manipulation               | 5        | 5          | 0        |     100.00%
Unchecked Low Level Calls       | 52       | 50         | 2        |      96.15%
----------------------------------------------------------------------------------------------------
TOTAL                           | 143      | 134        | 9        |      93.71%

====================================================================================================

Results saved to:
  - Detailed JSON: results/smartbugs/triton_test_results_20251030_153347.json
  - Text Summary: results/smartbugs/triton_test_summary_20251030_153347.txt
  - Markdown Table: results/smartbugs/triton_results_table_20251030_153347.md
====================================================================================================
```

---

## File 1: triton_test_summary_TIMESTAMP.txt

```
====================================================================================================
TRITON VULNERABILITY DETECTION - COMPREHENSIVE SUMMARY
====================================================================================================

Timestamp: 2025-10-30T15:33:47.123456
System: Triton v2.0

OVERALL METRICS
----------------------------------------------------------------------------------------------------
Average Precision: 0.9250
Average Recall: 0.9100
Average F1: 0.9175
Average Analysis Time: 2.3500
Total Contracts Tested: 143

SMARTBUGS DATASET
====================================================================================================
Total Contracts: 143
Successful Analyses: 143
Total Time: 335.50s

Overall Dataset Metrics:
----------------------------------------------------------------------------------------------------
  Average Precision: 0.9250
  Average Recall: 0.9100
  Average F1: 0.9175
  Average Analysis Time: 2.3500
  Throughput: 0.4261

VULNERABILITY DETECTION BREAKDOWN
====================================================================================================

Vulnerability Type              | Total    | Detected   | Missed   | Detection %
----------------------------------------------------------------------------------------------------
Access Control                  | 18       | 17         | 1        |      94.44%
Arithmetic                      | 15       | 14         | 1        |      93.33%
Bad Randomness                  | 8        | 7          | 1        |      87.50%
Denial Of Service               | 6        | 6          | 0        |     100.00%
Front Running                   | 4        | 3          | 1        |      75.00%
Other                           | 3        | 2          | 1        |      66.67%
Reentrancy                      | 31       | 30         | 1        |      96.77%
Time Manipulation               | 5        | 5          | 0        |     100.00%
Unchecked Low Level Calls       | 52       | 50         | 2        |      96.15%
----------------------------------------------------------------------------------------------------
TOTAL                           | 143      | 134        | 9        |      93.71%
====================================================================================================
```

---

## File 2: triton_results_table_TIMESTAMP.md (Markdown for Reports)

```markdown
# Triton Vulnerability Detection Results

## SMARTBUGS Dataset

| Vulnerability Type | Total | Detected | Missed | Detection Rate |
|-------------------|-------|----------|--------|----------------|
| Access Control | 18 | 17 | 1 | 94.44% |
| Arithmetic | 15 | 14 | 1 | 93.33% |
| Bad Randomness | 8 | 7 | 1 | 87.50% |
| Denial Of Service | 6 | 6 | 0 | 100.00% |
| Front Running | 4 | 3 | 1 | 75.00% |
| Other | 3 | 2 | 1 | 66.67% |
| Reentrancy | 31 | 30 | 1 | 96.77% |
| Time Manipulation | 5 | 5 | 0 | 100.00% |
| Unchecked Low Level Calls | 52 | 50 | 2 | 96.15% |
| **TOTAL** | **143** | **134** | **9** | **93.71%** |
```

### How the markdown renders:

# Triton Vulnerability Detection Results

## SMARTBUGS Dataset

| Vulnerability Type | Total | Detected | Missed | Detection Rate |
|-------------------|-------|----------|--------|----------------|
| Access Control | 18 | 17 | 1 | 94.44% |
| Arithmetic | 15 | 14 | 1 | 93.33% |
| Bad Randomness | 8 | 7 | 1 | 87.50% |
| Denial Of Service | 6 | 6 | 0 | 100.00% |
| Front Running | 4 | 3 | 1 | 75.00% |
| Other | 3 | 2 | 1 | 66.67% |
| Reentrancy | 31 | 30 | 1 | 96.77% |
| Time Manipulation | 5 | 5 | 0 | 100.00% |
| Unchecked Low Level Calls | 52 | 50 | 2 | 96.15% |
| **TOTAL** | **143** | **134** | **9** | **93.71%** |

---

## File 3: triton_test_results_TIMESTAMP.json (Excerpt)

```json
{
  "metadata": {
    "timestamp": "2025-10-30T15:33:47.123456",
    "system": "Triton v2.0"
  },
  "datasets": {
    "smartbugs": {
      "dataset": "SmartBugs",
      "total_contracts": 143,
      "successful_analyses": 143,
      "total_time": 335.5,
      "by_vulnerability": {
        "access_control": {
          "total": 18,
          "detected": 17,
          "missed": 1,
          "false_positives": 0
        },
        "arithmetic": {
          "total": 15,
          "detected": 14,
          "missed": 1,
          "false_positives": 0
        },
        "reentrancy": {
          "total": 31,
          "detected": 30,
          "missed": 1,
          "false_positives": 0
        }
        // ... more categories
      },
      "contracts": [
        {
          "contract_path": "data/datasets/smartbugs-curated/dataset/reentrancy/DAO.sol",
          "analysis_time": 2.45,
          "vulnerabilities_found": ["reentrancy"],
          "confidence_scores": {
            "reentrancy": 0.94
          },
          "modality_contributions": {
            "static": 0.35,
            "dynamic": 0.28,
            "semantic": 0.37
          },
          "success": true,
          "ground_truth": {
            "vulnerabilities": ["reentrancy"],
            "has_vulnerability": true
          },
          "metrics": {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0
          }
        }
        // ... 142 more contracts
      ],
      "metrics": {
        "average_precision": 0.925,
        "average_recall": 0.91,
        "average_f1": 0.9175,
        "average_analysis_time": 2.35,
        "throughput": 0.4261
      }
    }
  },
  "overall_metrics": {
    "average_precision": 0.925,
    "average_recall": 0.91,
    "average_f1": 0.9175,
    "average_analysis_time": 2.35,
    "total_contracts_tested": 143
  }
}
```

---

## How to Use These Files

### For Your Professor

Show them the **markdown table** (`triton_results_table_TIMESTAMP.md`):
- Clean, professional format
- Easy to read and understand
- Shows exact numbers for each category
- Can be copy-pasted into reports

### For Your Paper/Thesis

Copy the markdown table directly into your document:
- LaTeX: Use a markdown-to-LaTeX converter or create table manually
- Word: Paste as table
- Google Docs: Use markdown preview

### For Analysis

Use the **JSON file** (`triton_test_results_TIMESTAMP.json`):
- Load into Python for further analysis
- Create custom visualizations
- Deep dive into individual contract results
- Extract confidence scores and modality contributions

### For Quick Review

Use the **text summary** (`triton_test_summary_TIMESTAMP.txt`):
- Quick overview of performance
- Easy to read in terminal: `cat results/smartbugs/triton_test_summary_*.txt`
- Can be emailed or shared as plain text

---

## Interpretation Guide

### High Performance Categories (>95%)
- **Reentrancy**: 96.77% - Excellent detection
- **Unchecked Low Level Calls**: 96.15% - Excellent detection
- **Denial of Service**: 100% - Perfect detection (but only 6 samples)
- **Time Manipulation**: 100% - Perfect detection (but only 5 samples)

**Interpretation**: Triton excels at detecting these common vulnerabilities. The multi-modal approach (static + dynamic + semantic) works well for these patterns.

### Good Performance Categories (90-95%)
- **Access Control**: 94.44% - Good detection
- **Arithmetic**: 93.33% - Good detection

**Interpretation**: Strong performance. Minor improvements could push to >95%.

### Moderate Performance Categories (85-90%)
- **Bad Randomness**: 87.50% - Moderate detection

**Interpretation**: Room for improvement. May need better pattern recognition or additional training data.

### Lower Performance Categories (<85%)
- **Front Running**: 75.00% - Needs improvement
- **Other**: 66.67% - Needs improvement

**Interpretation**: These are harder to detect or have less training data. Consider:
- Adding more training examples
- Improving feature extraction
- Using domain-specific patterns

---

## Key Takeaways

✅ **Overall Performance**: 93.71% detection rate (134/143 vulnerabilities)
✅ **Strengths**: Reentrancy (96.77%), Unchecked Calls (96.15%)
✅ **Opportunities**: Front Running (75%), Other (66.67%)
✅ **Consistency**: High performance across most categories
✅ **Reliability**: 100% successful analyses (143/143 contracts)

---

**Note**: The numbers shown here are example estimates based on your presentation's target of 92.5% F1-score. Your actual results will depend on your model's training state and implementation details.
