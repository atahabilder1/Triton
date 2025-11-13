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


def compute_metrics(results):
    """Compute metrics for a single encoder."""
    predictions = results['predictions']

    if not predictions:
        return {
            'accuracy': 0.0,
            'total': results['total'],
            'successful': results['successful'],
            'failed': results['failed'],
            'correct': 0
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


def print_comparison(all_results):
    """Print comparison table."""
    logger.info("\n" + "="*100)
    logger.info("MODEL COMPARISON - INDIVIDUAL MODALITIES + FUSION")
    logger.info("="*100)
    logger.info(f"{'Model':<25} {'Accuracy':<12} {'Avg F1':<12} {'Correct/Tested':<18} {'Failed':<10}")
    logger.info("-"*100)

    for result in all_results:
        metrics = result['metrics']
        logger.info(f"{metrics['encoder']:<25} "
                   f"{metrics['accuracy']*100:>10.2f}% "
                   f"{metrics['avg_f1']:>10.3f} "
                   f"{metrics['correct']:>8}/{metrics['successful']:<8} "
                   f"{metrics['failed']:>8}")

    logger.info("="*100)

    # Per-class comparison
    logger.info("\nPER-CLASS F1 SCORE COMPARISON")
    logger.info("="*100)

    # Get all classes that appear in any result
    all_classes = set()
    for result in all_results:
        for cls in result['metrics']['per_class']:
            all_classes.add(cls['class'])

    # Header
    header = f"{'Vulnerability':<30}"
    for result in all_results:
        header += f"{result['metrics']['encoder']:<20}"
    logger.info(header)
    logger.info("-"*100)

    # Rows
    for vuln_class in VULNERABILITY_CLASSES:
        if vuln_class not in all_classes:
            continue

        row = f"{vuln_class:<30}"
        for result in all_results:
            f1_val = None
            for cls in result['metrics']['per_class']:
                if cls['class'] == vuln_class:
                    f1_val = cls['f1']
                    break

            if f1_val is not None:
                row += f"{f1_val:<20.3f}"
            else:
                row += f"{'N/A':<20}"

        logger.info(row)

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

    # 4. Test Fusion (would need to implement full fusion testing)
    # Skipping for now as it requires all modalities working

    # Print comparison
    if all_results:
        print_comparison(all_results)

    logger.info("\n✓ Testing complete!")


if __name__ == '__main__':
    main()
