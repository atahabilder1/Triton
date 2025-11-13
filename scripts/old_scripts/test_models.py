#!/usr/bin/env python3
"""
Test Each Modality Individually + Fusion Model
Tests: Static only, Dynamic only, Semantic only, and Full Fusion
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

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion
from tools.slither_wrapper import SlitherWrapper
from tools.mythril_wrapper import MythrilWrapper

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
    logger.info(f"Loading test dataset from {test_dir}\n")

    test_samples = []
    test_dir = Path(test_dir)

    for vuln_class in VULNERABILITY_CLASSES:
        vuln_dir = test_dir / vuln_class
        if not vuln_dir.exists():
            continue

        sol_files = list(vuln_dir.glob('*.sol'))

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

    logger.info(f"Loaded {len(test_samples)} test samples\n")
    return test_samples


def load_models(device):
    """Load all trained models."""
    logger.info("Loading trained models...\n")

    checkpoints_dir = Path('models/checkpoints')
    models = {}

    # Load Static Encoder
    static_path = checkpoints_dir / 'static_encoder_best.pt'
    if static_path.exists():
        try:
            static_encoder = StaticEncoder(
                node_feature_dim=128,
                hidden_dim=256,
                output_dim=768,
                dropout=0.2
            ).to(device)
            checkpoint = torch.load(static_path, map_location=device)
            static_encoder.load_state_dict(checkpoint['model_state_dict'])
            static_encoder.eval()
            models['static'] = static_encoder
            logger.info(f"✓ Loaded Static Encoder")
        except Exception as e:
            logger.warning(f"Failed to load static encoder: {e}")

    # Load Dynamic Encoder
    dynamic_path = checkpoints_dir / 'dynamic_encoder_best.pt'
    if dynamic_path.exists():
        try:
            dynamic_encoder = DynamicEncoder(
                vocab_size=50,
                embedding_dim=128,
                hidden_dim=256,
                output_dim=512,
                dropout=0.2
            ).to(device)
            checkpoint = torch.load(dynamic_path, map_location=device)
            dynamic_encoder.load_state_dict(checkpoint['model_state_dict'])
            dynamic_encoder.eval()
            models['dynamic'] = dynamic_encoder
            logger.info(f"✓ Loaded Dynamic Encoder")
        except Exception as e:
            logger.warning(f"Failed to load dynamic encoder: {e}")

    # Load Semantic Encoder
    semantic_path = checkpoints_dir / 'semantic_encoder_best.pt'
    if semantic_path.exists():
        try:
            semantic_encoder = SemanticEncoder(
                model_name="microsoft/graphcodebert-base",
                output_dim=768,
                max_length=512,
                dropout=0.1
            ).to(device)
            checkpoint = torch.load(semantic_path, map_location=device)
            semantic_encoder.load_state_dict(checkpoint['model_state_dict'])
            semantic_encoder.eval()
            models['semantic'] = semantic_encoder
            logger.info(f"✓ Loaded Semantic Encoder")
        except Exception as e:
            logger.warning(f"Failed to load semantic encoder: {e}")

    # Load Fusion Model
    fusion_path = checkpoints_dir / 'fusion_module_best.pt'
    static_fusion_path = checkpoints_dir / 'static_encoder_fusion_best.pt'
    dynamic_fusion_path = checkpoints_dir / 'dynamic_encoder_fusion_best.pt'
    semantic_fusion_path = checkpoints_dir / 'semantic_encoder_fusion_best.pt'

    if all([fusion_path.exists(), static_fusion_path.exists(),
            dynamic_fusion_path.exists(), semantic_fusion_path.exists()]):
        try:
            static_fusion = StaticEncoder(
                node_feature_dim=128,
                hidden_dim=256,
                output_dim=768,
                dropout=0.2
            ).to(device)
            checkpoint = torch.load(static_fusion_path, map_location=device)
            static_fusion.load_state_dict(checkpoint['model_state_dict'])
            static_fusion.eval()

            dynamic_fusion = DynamicEncoder(
                vocab_size=50,
                embedding_dim=128,
                hidden_dim=256,
                output_dim=512,
                dropout=0.2
            ).to(device)
            checkpoint = torch.load(dynamic_fusion_path, map_location=device)
            dynamic_fusion.load_state_dict(checkpoint['model_state_dict'])
            dynamic_fusion.eval()

            semantic_fusion = SemanticEncoder(
                model_name="microsoft/graphcodebert-base",
                output_dim=768,
                max_length=512,
                dropout=0.1
            ).to(device)
            checkpoint = torch.load(semantic_fusion_path, map_location=device)
            semantic_fusion.load_state_dict(checkpoint['model_state_dict'])
            semantic_fusion.eval()

            fusion_module = CrossModalFusion(
                static_dim=768,
                dynamic_dim=512,
                semantic_dim=768,
                hidden_dim=512,
                output_dim=768,
                dropout=0.1
            ).to(device)
            checkpoint = torch.load(fusion_path, map_location=device)
            fusion_module.load_state_dict(checkpoint['model_state_dict'])
            fusion_module.eval()

            models['fusion'] = {
                'static': static_fusion,
                'dynamic': dynamic_fusion,
                'semantic': semantic_fusion,
                'fusion': fusion_module
            }
            logger.info(f"✓ Loaded Fusion Model (all components)")
        except Exception as e:
            logger.warning(f"Failed to load fusion model: {e}")

    logger.info("")
    return models


def test_static_only(static_encoder, test_samples, device, slither):
    """Test ONLY static encoder."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: STATIC ENCODER ONLY")
    logger.info("="*80)

    predictions = []
    successful = 0
    failed = 0

    for i, sample in enumerate(test_samples):
        try:
            # Extract PDG
            pdg_result = slither.analyze_contract(sample['source_code'])

            if not pdg_result.get('success') or not pdg_result.get('pdg'):
                failed += 1
                continue

            pdg = pdg_result['pdg']

            with torch.no_grad():
                static_features, vuln_scores = static_encoder([pdg], None)
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                probs = torch.sigmoid(all_scores)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_label].item()

                predictions.append({
                    'true_label': sample['true_label'],
                    'true_label_idx': sample['true_label_idx'],
                    'predicted_label': VULNERABILITY_CLASSES[pred_label],
                    'predicted_label_idx': pred_label,
                    'correct': pred_label == sample['true_label_idx']
                })
                successful += 1

        except Exception as e:
            failed += 1
            continue

    return {
        'encoder': 'Static Only',
        'predictions': predictions,
        'successful': successful,
        'failed': failed,
        'total': len(test_samples)
    }


def test_dynamic_only(dynamic_encoder, test_samples, device, mythril):
    """Test ONLY dynamic encoder."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: DYNAMIC ENCODER ONLY")
    logger.info("="*80)

    predictions = []
    successful = 0
    failed = 0

    for i, sample in enumerate(test_samples):
        try:
            # Extract traces
            trace_result = mythril.analyze_contract(sample['source_code'])

            if not trace_result.get('success'):
                failed += 1
                continue

            traces = trace_result.get('execution_traces', [{'steps': []}])
            trace = traces[0] if traces else {'steps': []}

            with torch.no_grad():
                dynamic_features, vuln_scores = dynamic_encoder([trace], None)
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                probs = torch.sigmoid(all_scores)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_label].item()

                predictions.append({
                    'true_label': sample['true_label'],
                    'true_label_idx': sample['true_label_idx'],
                    'predicted_label': VULNERABILITY_CLASSES[pred_label],
                    'predicted_label_idx': pred_label,
                    'correct': pred_label == sample['true_label_idx']
                })
                successful += 1

        except Exception as e:
            failed += 1
            continue

    return {
        'encoder': 'Dynamic Only',
        'predictions': predictions,
        'successful': successful,
        'failed': failed,
        'total': len(test_samples)
    }


def test_semantic_only(semantic_encoder, test_samples, device):
    """Test ONLY semantic encoder."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: SEMANTIC ENCODER ONLY")
    logger.info("="*80)

    predictions = []
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

                for j, sample in enumerate(batch):
                    pred_label = pred_labels[j]
                    predictions.append({
                        'true_label': sample['true_label'],
                        'true_label_idx': sample['true_label_idx'],
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'predicted_label_idx': int(pred_label),
                        'correct': int(pred_label) == sample['true_label_idx']
                    })

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    return {
        'encoder': 'Semantic Only',
        'predictions': predictions,
        'successful': len(predictions),
        'failed': len(test_samples) - len(predictions),
        'total': len(test_samples)
    }


def test_fusion_model(fusion_models, test_samples, device, slither, mythril):
    """Test fusion model combining all modalities."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: FUSION MODEL (ALL MODALITIES COMBINED)")
    logger.info("="*80)

    static_encoder = fusion_models['static']
    dynamic_encoder = fusion_models['dynamic']
    semantic_encoder = fusion_models['semantic']
    fusion_module = fusion_models['fusion']

    predictions = []
    successful = 0
    failed = 0

    for i, sample in enumerate(test_samples):
        try:
            # Extract static features (PDG)
            pdg_result = slither.analyze_contract(sample['source_code'])
            if pdg_result.get('success') and pdg_result.get('pdg'):
                pdg = pdg_result['pdg']
                with torch.no_grad():
                    static_features, _ = static_encoder([pdg], None)
            else:
                static_features = None

            # Extract dynamic features (traces)
            trace_result = mythril.analyze_contract(sample['source_code'])
            if trace_result.get('success'):
                traces = trace_result.get('execution_traces', [{'steps': []}])
                trace = traces[0] if traces else {'steps': []}
                with torch.no_grad():
                    dynamic_features, _ = dynamic_encoder([trace], None)
            else:
                dynamic_features = None

            # Extract semantic features
            with torch.no_grad():
                semantic_features, _ = semantic_encoder([sample['source_code']], None)

            # If we have at least semantic features, proceed
            if semantic_features is not None:
                with torch.no_grad():
                    # Fusion module returns a dictionary
                    fusion_output = fusion_module(
                        static_features,
                        dynamic_features,
                        semantic_features,
                        None
                    )

                    # Get vulnerability logits from the output dictionary
                    vuln_logits = fusion_output['vulnerability_logits']
                    probs = torch.softmax(vuln_logits, dim=1)
                    pred_label = torch.argmax(probs, dim=1).item()

                    predictions.append({
                        'true_label': sample['true_label'],
                        'true_label_idx': sample['true_label_idx'],
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'predicted_label_idx': pred_label,
                        'correct': pred_label == sample['true_label_idx']
                    })
                    successful += 1
            else:
                failed += 1

        except Exception as e:
            logger.debug(f"Fusion error on sample {i}: {e}")
            failed += 1
            continue

    return {
        'encoder': 'Fusion (All Modalities)',
        'predictions': predictions,
        'successful': successful,
        'failed': failed,
        'total': len(test_samples)
    }


def compute_metrics(results):
    """Compute metrics for a single encoder."""
    predictions = results['predictions']

    if not predictions:
        return {
            'encoder': results['encoder'],
            'accuracy': 0.0,
            'total': results['total'],
            'successful': results['successful'],
            'failed': results['failed'],
            'correct': 0,
            'avg_f1': 0.0
        }

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
                'detected': int(tp),
                'support': int(support)
            })

    avg_f1 = np.mean([c['f1'] for c in per_class]) if per_class else 0.0

    return {
        'encoder': results['encoder'],
        'accuracy': accuracy,
        'total': results['total'],
        'successful': results['successful'],
        'failed': results['failed'],
        'correct': correct,
        'avg_f1': avg_f1,
        'per_class': per_class
    }


def print_per_class_table(encoder_name, metrics, predictions):
    """Print detailed per-class performance table for one modality."""
    logger.info("\n" + "="*100)
    logger.info(f"{encoder_name.upper()} - DETAILED PER-CLASS PERFORMANCE")
    logger.info("="*100)
    logger.info(f"Overall Accuracy: {metrics['accuracy']*100:.2f}% ({metrics['correct']}/{metrics['successful']} correct)")
    logger.info(f"Successfully Tested: {metrics['successful']}/{metrics['total']} contracts (Failed: {metrics['failed']})")

    # Handle case where no predictions were made
    if not predictions or 'per_class' not in metrics:
        logger.info("-"*100)
        logger.info("No predictions were made - all contracts failed processing")
        logger.info("="*100)
        return

    logger.info("-"*100)
    logger.info(f"{'Vulnerability Type':<30} {'Total':<10} {'Detected':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    logger.info("-"*100)

    # Count actual totals from predictions
    class_totals = defaultdict(int)
    class_detected = defaultdict(int)

    for pred in predictions:
        true_class = pred['true_label']
        class_totals[true_class] += 1
        if pred['correct']:
            class_detected[true_class] += 1

    total_support = 0
    weighted_f1 = 0.0

    for cls_info in metrics['per_class']:
        vuln_class = cls_info['class']
        total = class_totals.get(vuln_class, 0)
        detected = class_detected.get(vuln_class, 0)
        accuracy_pct = (detected / total * 100) if total > 0 else 0.0

        logger.info(f"{vuln_class:<30} "
                   f"{total:<10} "
                   f"{detected:<12} "
                   f"{accuracy_pct:<11.2f}% "
                   f"{cls_info['precision']*100:<11.2f}% "
                   f"{cls_info['recall']*100:<11.2f}% "
                   f"{cls_info['f1']:<12.3f}")

        total_support += total
        weighted_f1 += cls_info['f1'] * total

    # Classes with no predictions
    for vuln_class in VULNERABILITY_CLASSES:
        if vuln_class not in class_totals and vuln_class in [c['class'] for c in metrics['per_class']]:
            logger.info(f"{vuln_class:<30} {'0':<10} {'0':<12} {'0.00':<11}% {'0.00':<11}% {'0.00':<11}% {'0.000':<12}")

    logger.info("-"*100)
    avg_f1 = weighted_f1 / total_support if total_support > 0 else 0.0
    logger.info(f"{'AVERAGE (weighted)':<30} {total_support:<10} {sum(class_detected.values()):<12} "
               f"{metrics['accuracy']*100:<11.2f}% {'---':<11}  {'---':<11}  {avg_f1:<12.3f}")
    logger.info("="*100)


def print_combined_table(all_results):
    """Print a single combined table showing all modalities side-by-side."""
    logger.info("\n" + "="*130)
    logger.info("COMBINED PERFORMANCE TABLE - ALL MODALITIES")
    logger.info("="*130)

    # Filter out results with no predictions
    valid_results = [r for r in all_results if r['raw']['predictions']]

    if not valid_results:
        logger.info("No valid predictions from any modality")
        logger.info("="*130)
        return

    # Header line 1
    header1 = f"{'Vulnerability Type':<25} {'Total':<8} "
    for result in valid_results:
        header1 += f"{'| ' + result['raw']['encoder']:<30} "
    logger.info(header1)

    # Header line 2
    header2 = f"{'':25} {'':8} "
    for _ in valid_results:
        header2 += f"{'| Detected / Accuracy':<30} "
    logger.info(header2)
    logger.info("-"*130)

    # Count totals for each class
    all_class_totals = defaultdict(int)
    for result in valid_results:
        for pred in result['raw']['predictions']:
            all_class_totals[pred['true_label']] += 1

    # Average across all predictions
    max_total = {cls: 0 for cls in VULNERABILITY_CLASSES}
    for result in valid_results:
        class_counts = defaultdict(int)
        for pred in result['raw']['predictions']:
            class_counts[pred['true_label']] += 1
        for cls in VULNERABILITY_CLASSES:
            if class_counts[cls] > max_total[cls]:
                max_total[cls] = class_counts[cls]

    # Print each vulnerability class
    for vuln_class in VULNERABILITY_CLASSES:
        total = max_total.get(vuln_class, 0)
        if total == 0:
            continue

        row = f"{vuln_class:<25} {total:<8} "

        for result in valid_results:
            # Count for this class in this result
            class_total = 0
            class_detected = 0
            for pred in result['raw']['predictions']:
                if pred['true_label'] == vuln_class:
                    class_total += 1
                    if pred['correct']:
                        class_detected += 1

            accuracy_pct = (class_detected / class_total * 100) if class_total > 0 else 0.0
            row += f"| {class_detected:>3}/{class_total:<3} ({accuracy_pct:>6.2f}%){'':<12} "

        logger.info(row)

    logger.info("-"*130)

    # Overall accuracy row
    overall_row = f"{'OVERALL ACCURACY':<25} {'':<8} "
    for result in valid_results:
        metrics = result['metrics']
        overall_row += f"| {metrics['correct']:>3}/{metrics['successful']:<3} ({metrics['accuracy']*100:>6.2f}%){'':<12} "
    logger.info(overall_row)

    logger.info("="*130)


def print_comparison(all_results):
    """Print detailed per-class tables for each modality."""

    # Print individual tables for each modality
    for result in all_results:
        encoder_name = result['raw']['encoder']
        metrics = result['metrics']
        predictions = result['raw']['predictions']
        print_per_class_table(encoder_name, metrics, predictions)

    # Print combined table
    print_combined_table(all_results)

    # Print summary comparison
    logger.info("\n" + "="*100)
    logger.info("SUMMARY COMPARISON - ALL MODALITIES")
    logger.info("="*100)
    logger.info(f"{'Model':<25} {'Overall Accuracy':<20} {'Avg F1-Score':<15} {'Tested/Total':<15}")
    logger.info("-"*100)

    for result in all_results:
        metrics = result['metrics']
        logger.info(f"{metrics['encoder']:<25} "
                   f"{metrics['accuracy']*100:>18.2f}% "
                   f"{metrics['avg_f1']:>14.3f} "
                   f"{metrics['successful']:>7}/{metrics['total']:<6}")

    logger.info("="*100)


def main():
    """Main testing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test each modality individually")
    parser.add_argument(
        '--test-dir',
        default='data/datasets/combined_labeled/test',
        help='Test dataset directory'
    )
    parser.add_argument(
        '--skip-static',
        action='store_true',
        help='Skip static encoder (if Slither broken)'
    )
    parser.add_argument(
        '--skip-dynamic',
        action='store_true',
        help='Skip dynamic encoder (if Mythril broken)'
    )
    parser.add_argument(
        '--skip-fusion',
        action='store_true',
        help='Skip fusion model'
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

    # Load models
    models = load_models(device)

    # Initialize tools (for static/dynamic)
    slither = SlitherWrapper(timeout=30)
    mythril = MythrilWrapper(timeout=30, max_depth=12)

    # Test each modality
    all_results = []

    # 1. Test Static Only
    if 'static' in models and not args.skip_static:
        result = test_static_only(models['static'], test_samples, device, slither)
        metrics = compute_metrics(result)
        all_results.append({'type': 'static', 'raw': result, 'metrics': metrics})
        logger.info(f"\nStatic Only: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # 2. Test Dynamic Only
    if 'dynamic' in models and not args.skip_dynamic:
        result = test_dynamic_only(models['dynamic'], test_samples, device, mythril)
        metrics = compute_metrics(result)
        all_results.append({'type': 'dynamic', 'raw': result, 'metrics': metrics})
        logger.info(f"\nDynamic Only: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # 3. Test Semantic Only
    if 'semantic' in models:
        result = test_semantic_only(models['semantic'], test_samples, device)
        metrics = compute_metrics(result)
        all_results.append({'type': 'semantic', 'raw': result, 'metrics': metrics})
        logger.info(f"\nSemantic Only: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # 4. Test Fusion Model
    if 'fusion' in models and not args.skip_fusion:
        result = test_fusion_model(models['fusion'], test_samples, device, slither, mythril)
        metrics = compute_metrics(result)
        all_results.append({'type': 'fusion', 'raw': result, 'metrics': metrics})
        logger.info(f"\nFusion Model: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # Print comparison
    if all_results:
        print_comparison(all_results)

    logger.info("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
