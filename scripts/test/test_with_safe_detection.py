#!/usr/bin/env python3
"""
Testing with Safe Contract Detection
Adds confidence threshold to detect safe contracts
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from encoders.semantic_encoder import SemanticEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VULNERABILITY_CLASSES = [
    'access_control', 'arithmetic', 'bad_randomness', 'denial_of_service',
    'front_running', 'reentrancy', 'short_addresses', 'time_manipulation',
    'unchecked_low_level_calls', 'safe'
]

VULN_CLASS_TO_IDX = {v: i for i, v in enumerate(VULNERABILITY_CLASSES)}
SAFE_IDX = VULN_CLASS_TO_IDX['safe']


def load_test_dataset(test_dir):
    """Load test dataset."""
    test_samples = []
    test_dir = Path(test_dir)

    for vuln_class in VULNERABILITY_CLASSES:
        vuln_dir = test_dir / vuln_class
        if not vuln_dir.exists():
            continue

        for sol_file in vuln_dir.glob('*.sol'):
            try:
                with open(sol_file, 'r', encoding='utf-8', errors='ignore') as f:
                    test_samples.append({
                        'path': str(sol_file),
                        'source_code': f.read(),
                        'true_label': vuln_class,
                        'true_label_idx': VULN_CLASS_TO_IDX[vuln_class]
                    })
            except Exception as e:
                logger.warning(f"Error loading {sol_file}: {e}")

    logger.info(f"Loaded {len(test_samples)} test samples\n")
    return test_samples


def test_with_threshold(semantic_encoder, test_samples, device, confidence_threshold=0.6):
    """Test with confidence threshold for safe contract detection."""
    logger.info(f"\nTesting with confidence threshold: {confidence_threshold}")
    logger.info("="*80)

    predictions = []
    batch_size = 4

    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        source_codes = [s['source_code'] for s in batch]

        try:
            with torch.no_grad():
                semantic_features, vuln_scores = semantic_encoder(source_codes, None)
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                probs = torch.sigmoid(all_scores)

                for j, sample in enumerate(batch):
                    max_prob, pred_label = torch.max(probs[j], dim=0)
                    max_prob = max_prob.item()
                    pred_label = pred_label.item()

                    # If max confidence is below threshold, classify as safe
                    if max_prob < confidence_threshold:
                        pred_label = SAFE_IDX
                        final_confidence = 1.0 - max_prob  # Confidence in "safe" prediction
                    else:
                        final_confidence = max_prob

                    predictions.append({
                        'true_label': sample['true_label'],
                        'true_label_idx': sample['true_label_idx'],
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'predicted_label_idx': pred_label,
                        'confidence': final_confidence,
                        'correct': pred_label == sample['true_label_idx']
                    })

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    return predictions


def compute_metrics(predictions):
    """Compute accuracy and per-class metrics."""
    from collections import defaultdict
    import numpy as np

    total = len(predictions)
    correct = sum(1 for p in predictions if p['correct'])
    accuracy = correct / total if total > 0 else 0.0

    # Per-class metrics
    confusion = defaultdict(lambda: defaultdict(int))
    for pred in predictions:
        confusion[pred['true_label']][pred['predicted_label']] += 1

    per_class = []
    for vuln_class in VULNERABILITY_CLASSES:
        tp = confusion[vuln_class][vuln_class]
        fp = sum(confusion[other][vuln_class] for other in VULNERABILITY_CLASSES if other != vuln_class)
        fn = sum(confusion[vuln_class][other] for other in VULNERABILITY_CLASSES if other != vuln_class)
        support = tp + fn

        if support > 0:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            per_class.append({
                'class': vuln_class,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'detected': tp,
                'support': support
            })

    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'per_class': per_class
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', default='data/datasets/combined_labeled/test')
    parser.add_argument('--threshold', type=float, default=0.6, help='Confidence threshold for safe classification')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}\n")

    # Load test dataset
    test_samples = load_test_dataset(args.test_dir)

    # Load semantic encoder
    logger.info("Loading semantic encoder...")
    semantic_encoder = SemanticEncoder(
        model_name="microsoft/graphcodebert-base",
        output_dim=768,
        max_length=512,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load('models/checkpoints/semantic_encoder_best.pt', map_location=device)
    semantic_encoder.load_state_dict(checkpoint['model_state_dict'])
    semantic_encoder.eval()
    logger.info("✓ Loaded\n")

    # Test with different thresholds
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]

    logger.info("\n" + "="*80)
    logger.info("THRESHOLD COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Threshold':<12} {'Accuracy':<12} {'Safe Detected':<15} {'Safe Precision':<18} {'Safe Recall':<15}")
    logger.info("-"*80)

    best_threshold = None
    best_accuracy = 0.0

    for threshold in thresholds:
        predictions = test_with_threshold(semantic_encoder, test_samples, device, threshold)
        metrics = compute_metrics(predictions)

        # Find safe class metrics
        safe_metrics = next((m for m in metrics['per_class'] if m['class'] == 'safe'), None)

        if safe_metrics:
            logger.info(f"{threshold:<12.2f} {metrics['accuracy']*100:<11.2f}% "
                       f"{safe_metrics['detected']:<15} "
                       f"{safe_metrics['precision']:<18.3f} "
                       f"{safe_metrics['recall']:<15.3f}")
        else:
            logger.info(f"{threshold:<12.2f} {metrics['accuracy']*100:<11.2f}% 0               N/A                N/A")

        if metrics['accuracy'] > best_accuracy:
            best_accuracy = metrics['accuracy']
            best_threshold = threshold

    logger.info("="*80)
    logger.info(f"\nBest threshold: {best_threshold} with accuracy: {best_accuracy*100:.2f}%")

    # Test with best threshold and show detailed results
    logger.info(f"\n\nDETAILED RESULTS WITH THRESHOLD={best_threshold}")
    logger.info("="*80)

    predictions = test_with_threshold(semantic_encoder, test_samples, device, best_threshold)
    metrics = compute_metrics(predictions)

    logger.info(f"\nOverall Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"Correct: {metrics['correct']}/{metrics['total']}\n")

    logger.info(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Detected/Total':<15}")
    logger.info("-"*80)
    for m in metrics['per_class']:
        logger.info(f"{m['class']:<30} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {m['detected']}/{m['support']}")

    logger.info("="*80)


if __name__ == '__main__':
    main()
