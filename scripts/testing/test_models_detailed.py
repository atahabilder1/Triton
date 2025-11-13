#!/usr/bin/env python3
"""
Detailed Model Testing with Complete Metrics
Shows: Accuracy, Precision, Recall, F1, TP, FP, FN, TN for each class
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from encoders.semantic_encoder import SemanticEncoder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vulnerability categories
VULNERABILITY_CLASSES = [
    'access_control', 'arithmetic', 'bad_randomness', 'denial_of_service',
    'front_running', 'reentrancy', 'short_addresses', 'time_manipulation',
    'unchecked_low_level_calls', 'safe'
]

VULN_CLASS_TO_IDX = {v: i for i, v in enumerate(VULNERABILITY_CLASSES)}


def load_test_dataset(test_dir):
    """Load test dataset from directory structure."""
    logger.info(f"Loading test dataset from {test_dir}")

    test_samples = []
    test_dir = Path(test_dir)

    class_counts = defaultdict(int)

    for vuln_class in VULNERABILITY_CLASSES:
        vuln_dir = test_dir / vuln_class
        if not vuln_dir.exists():
            continue

        sol_files = list(vuln_dir.glob('*.sol'))
        class_counts[vuln_class] = len(sol_files)

        for sol_file in sol_files:
            try:
                with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                    source_code = f.read()

                test_samples.append({
                    'path': str(sol_file),
                    'source_code': source_code,
                    'true_label': vuln_class,
                    'true_label_idx': VULN_CLASS_TO_IDX[vuln_class]
                })
            except Exception as e:
                logger.warning(f"Error loading {sol_file}: {e}")

    logger.info(f"\nTest Set Composition:")
    logger.info("=" * 80)
    total = 0
    for vuln_class in VULNERABILITY_CLASSES:
        count = class_counts[vuln_class]
        if count > 0:
            logger.info(f"  {vuln_class:<30} {count:>3} contracts")
            total += count
    logger.info("=" * 80)
    logger.info(f"  {'TOTAL':<30} {total:>3} contracts\n")

    return test_samples


def load_semantic_encoder(device):
    """Load trained semantic encoder."""
    logger.info("Loading semantic encoder...")

    checkpoint_path = Path('models/checkpoints/semantic_encoder_best.pt')

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None

    semantic_encoder = SemanticEncoder(
        model_name="microsoft/graphcodebert-base",
        output_dim=768,
        max_length=512,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    semantic_encoder.load_state_dict(checkpoint['model_state_dict'])
    semantic_encoder.eval()

    logger.info(f"✓ Loaded semantic encoder from {checkpoint_path}\n")
    return semantic_encoder


def test_semantic_encoder(semantic_encoder, test_samples, device):
    """Test semantic encoder with detailed metrics."""
    logger.info("=" * 80)
    logger.info("TESTING SEMANTIC ENCODER")
    logger.info("=" * 80)

    predictions = []

    # Process in batches
    batch_size = 4
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        source_codes = [s['source_code'] for s in batch]
        true_labels = [s['true_label_idx'] for s in batch]

        try:
            with torch.no_grad():
                semantic_features, vuln_scores = semantic_encoder(source_codes, None)
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                probs = torch.sigmoid(all_scores)
                pred_labels = torch.argmax(probs, dim=1).cpu().numpy()
                confidences = torch.max(probs, dim=1)[0].cpu().numpy()

                for j, sample in enumerate(batch):
                    pred_label = pred_labels[j]
                    confidence = confidences[j]
                    true_label = true_labels[j]

                    predictions.append({
                        'path': sample['path'],
                        'true_label': sample['true_label'],
                        'true_label_idx': true_label,
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'predicted_label_idx': pred_label,
                        'confidence': float(confidence),
                        'correct': pred_label == true_label
                    })

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    return predictions


def compute_detailed_metrics(predictions):
    """Compute detailed metrics including counts."""

    # Overall metrics
    total = len(predictions)
    correct = sum(1 for p in predictions if p['correct'])
    overall_accuracy = correct / total if total > 0 else 0

    logger.info("\n" + "=" * 80)
    logger.info("OVERALL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Samples:      {total}")
    logger.info(f"Correct:            {correct}")
    logger.info(f"Incorrect:          {total - correct}")
    logger.info(f"Overall Accuracy:   {overall_accuracy*100:.2f}%")

    # Per-class confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for pred in predictions:
        confusion[pred['true_label']][pred['predicted_label']] += 1

    # Per-class metrics
    logger.info("\n" + "=" * 80)
    logger.info("PER-CLASS DETAILED METRICS")
    logger.info("=" * 80)
    logger.info(f"{'Class':<25} {'Acc':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'Support':<8}")
    logger.info("-" * 80)

    all_metrics = []

    for vuln_class in VULNERABILITY_CLASSES:
        # Calculate TP, FP, FN, TN
        tp = confusion[vuln_class][vuln_class]
        fp = sum(confusion[other][vuln_class] for other in VULNERABILITY_CLASSES if other != vuln_class)
        fn = sum(confusion[vuln_class][other] for other in VULNERABILITY_CLASSES if other != vuln_class)
        tn = sum(
            confusion[other1][other2]
            for other1 in VULNERABILITY_CLASSES if other1 != vuln_class
            for other2 in VULNERABILITY_CLASSES if other2 != vuln_class
        )

        support = tp + fn  # Total actual samples of this class

        if support == 0:
            continue

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"{vuln_class:<25} {accuracy:<8.3f} {precision:<8.3f} {recall:<8.3f} {f1:<8.3f} "
                   f"{tp:<6} {fp:<6} {fn:<6} {support:<8}")

        all_metrics.append({
            'class': vuln_class,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'support': support,
            'detected': tp,
            'missed': fn,
            'false_positives': fp
        })

    logger.info("-" * 80)

    # Average metrics (macro average)
    avg_accuracy = np.mean([m['accuracy'] for m in all_metrics])
    avg_precision = np.mean([m['precision'] for m in all_metrics])
    avg_recall = np.mean([m['recall'] for m in all_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_metrics])

    logger.info(f"{'AVERAGE (Macro)':<25} {avg_accuracy:<8.3f} {avg_precision:<8.3f} {avg_recall:<8.3f} {avg_f1:<8.3f}")
    logger.info("=" * 80)

    # Detection summary
    logger.info("\n" + "=" * 80)
    logger.info("DETECTION SUMMARY (What was detected vs missed)")
    logger.info("=" * 80)
    logger.info(f"{'Class':<30} {'Total':<8} {'Detected':<10} {'Missed':<10} {'Detection Rate':<15}")
    logger.info("-" * 80)

    for m in all_metrics:
        detection_rate = (m['detected'] / m['support'] * 100) if m['support'] > 0 else 0
        logger.info(f"{m['class']:<30} {m['support']:<8} {m['detected']:<10} {m['missed']:<10} {detection_rate:<14.1f}%")

    logger.info("=" * 80)

    # Prediction distribution
    logger.info("\n" + "=" * 80)
    logger.info("PREDICTION DISTRIBUTION (What model predicted)")
    logger.info("=" * 80)

    pred_counts = defaultdict(int)
    for pred in predictions:
        pred_counts[pred['predicted_label']] += 1

    logger.info(f"{'Predicted Class':<30} {'Count':<10} {'% of Total':<15}")
    logger.info("-" * 80)
    for vuln_class in VULNERABILITY_CLASSES:
        count = pred_counts[vuln_class]
        percentage = (count / total * 100) if total > 0 else 0
        if count > 0:
            logger.info(f"{vuln_class:<30} {count:<10} {percentage:<14.1f}%")
    logger.info("=" * 80)

    # Confusion matrix (top errors)
    logger.info("\n" + "=" * 80)
    logger.info("TOP CONFUSION PAIRS (Where model gets confused)")
    logger.info("=" * 80)

    confusion_pairs = []
    for true_class in VULNERABILITY_CLASSES:
        for pred_class in VULNERABILITY_CLASSES:
            if true_class != pred_class and confusion[true_class][pred_class] > 0:
                confusion_pairs.append({
                    'true': true_class,
                    'predicted': pred_class,
                    'count': confusion[true_class][pred_class]
                })

    confusion_pairs.sort(key=lambda x: x['count'], reverse=True)

    if confusion_pairs:
        logger.info(f"{'True Class':<30} {'Predicted As':<30} {'Count':<10}")
        logger.info("-" * 80)
        for pair in confusion_pairs[:10]:  # Top 10
            logger.info(f"{pair['true']:<30} {pair['predicted']:<30} {pair['count']:<10}")
    else:
        logger.info("No confusion pairs (perfect classification!)")

    logger.info("=" * 80)

    return {
        'overall_accuracy': overall_accuracy,
        'total': total,
        'correct': correct,
        'per_class_metrics': all_metrics,
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'predictions': predictions
    }


def main():
    """Main testing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Detailed model testing with all metrics")
    parser.add_argument(
        '--test-dir',
        default='data/datasets/combined_labeled/test',
        help='Test dataset directory'
    )
    parser.add_argument(
        '--output-file',
        default='detailed_test_results.json',
        help='Output JSON file'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")

    # Load test dataset
    test_samples = load_test_dataset(args.test_dir)

    if not test_samples:
        logger.error("No test samples found!")
        return

    # Load semantic encoder
    semantic_encoder = load_semantic_encoder(device)

    if not semantic_encoder:
        logger.error("Failed to load semantic encoder!")
        return

    # Test
    predictions = test_semantic_encoder(semantic_encoder, test_samples, device)

    # Compute detailed metrics
    results = compute_detailed_metrics(predictions)

    # Save results
    output_file = Path(args.output_file)
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'test_dir': args.test_dir,
            'overall_accuracy': results['overall_accuracy'],
            'total_samples': results['total'],
            'correct_predictions': results['correct'],
            'avg_accuracy': results['avg_accuracy'],
            'avg_precision': results['avg_precision'],
            'avg_recall': results['avg_recall'],
            'avg_f1': results['avg_f1'],
            'per_class_metrics': results['per_class_metrics'],
            'all_predictions': results['predictions']
        }, f, indent=2)

    logger.info(f"\n✓ Detailed results saved to: {output_file}")


if __name__ == '__main__':
    main()
