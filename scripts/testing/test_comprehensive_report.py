#!/usr/bin/env python3
"""
Comprehensive Test Suite - All Modalities + Fusion
Tests all contracts in test directory and generates detailed report
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
                    'filename': sol_file.name,
                    'source_code': source_code,
                    'true_label': vuln_class,
                    'true_label_idx': VULN_CLASS_TO_IDX[vuln_class]
                })
            except Exception as e:
                logger.warning(f"Error loading {sol_file}: {e}")

    logger.info(f"Loaded {len(test_samples)} test samples")

    # Log distribution
    dist = defaultdict(int)
    for sample in test_samples:
        dist[sample['true_label']] += 1

    logger.info("Test set distribution:")
    for vuln_class in VULNERABILITY_CLASSES:
        if dist[vuln_class] > 0:
            logger.info(f"  {vuln_class}: {dist[vuln_class]} contracts")
    logger.info("")

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

    # Load Fusion Model (all components)
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
            logger.info(f"✓ Loaded Fusion Model (all 4 components)")
        except Exception as e:
            logger.warning(f"Failed to load fusion model: {e}")

    logger.info("")
    return models


def test_single_encoder(encoder_name, encoder, test_samples, device, slither=None, mythril=None):
    """Test a single encoder (static, dynamic, or semantic)."""
    logger.info("\n" + "="*80)
    logger.info(f"TESTING: {encoder_name.upper()} ENCODER")
    logger.info("="*80)

    predictions = []
    successful = 0
    failed = 0

    if encoder_name == 'static':
        # Static encoder needs Slither
        for i, sample in enumerate(test_samples):
            try:
                pdg_result = slither.analyze_contract(sample['source_code'])
                if not pdg_result.get('success') or not pdg_result.get('pdg'):
                    failed += 1
                    continue

                pdg = pdg_result['pdg']
                with torch.no_grad():
                    static_features, vuln_scores = encoder([pdg], None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                    probs = torch.sigmoid(all_scores)
                    pred_label = torch.argmax(probs, dim=1).item()

                    predictions.append({
                        'filename': sample['filename'],
                        'true_label': sample['true_label'],
                        'true_label_idx': sample['true_label_idx'],
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'predicted_label_idx': pred_label,
                        'confidence': probs[0, pred_label].item(),
                        'correct': pred_label == sample['true_label_idx']
                    })
                    successful += 1

            except Exception as e:
                failed += 1
                continue

    elif encoder_name == 'dynamic':
        # Dynamic encoder needs Mythril
        for i, sample in enumerate(test_samples):
            try:
                trace_result = mythril.analyze_contract(sample['source_code'])
                if not trace_result.get('success'):
                    failed += 1
                    continue

                traces = trace_result.get('execution_traces', [{'steps': []}])
                trace = traces[0] if traces else {'steps': []}

                with torch.no_grad():
                    dynamic_features, vuln_scores = encoder([trace], None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                    probs = torch.sigmoid(all_scores)
                    pred_label = torch.argmax(probs, dim=1).item()

                    predictions.append({
                        'filename': sample['filename'],
                        'true_label': sample['true_label'],
                        'true_label_idx': sample['true_label_idx'],
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'predicted_label_idx': pred_label,
                        'confidence': probs[0, pred_label].item(),
                        'correct': pred_label == sample['true_label_idx']
                    })
                    successful += 1

            except Exception as e:
                failed += 1
                continue

    elif encoder_name == 'semantic':
        # Semantic encoder - batch processing
        batch_size = 4
        for i in range(0, len(test_samples), batch_size):
            batch = test_samples[i:i+batch_size]
            source_codes = [s['source_code'] for s in batch]

            try:
                with torch.no_grad():
                    semantic_features, vuln_scores = encoder(source_codes, None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                    probs = torch.sigmoid(all_scores)
                    pred_labels = torch.argmax(probs, dim=1).cpu().numpy()

                    for j, sample in enumerate(batch):
                        pred_label = pred_labels[j]
                        predictions.append({
                            'filename': sample['filename'],
                            'true_label': sample['true_label'],
                            'true_label_idx': sample['true_label_idx'],
                            'predicted_label': VULNERABILITY_CLASSES[pred_label],
                            'predicted_label_idx': int(pred_label),
                            'confidence': probs[j, pred_label].item(),
                            'correct': int(pred_label) == sample['true_label_idx']
                        })
                        successful += 1

            except Exception as e:
                logger.error(f"Error processing batch {i}: {e}")
                failed += len(batch)
                continue

    return {
        'encoder': f'{encoder_name.capitalize()} Only',
        'predictions': predictions,
        'successful': successful,
        'failed': failed,
        'total': len(test_samples)
    }


def test_fusion_model(fusion_models, test_samples, device, slither, mythril):
    """Test fusion model (all modalities combined)."""
    logger.info("\n" + "="*80)
    logger.info("TESTING: FUSION MODEL (All Modalities)")
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
            # Extract features from all modalities
            # 1. Static (PDG)
            pdg_result = slither.analyze_contract(sample['source_code'])
            if not pdg_result.get('success') or not pdg_result.get('pdg'):
                failed += 1
                continue
            pdg = pdg_result['pdg']

            # 2. Dynamic (Traces)
            trace_result = mythril.analyze_contract(sample['source_code'])
            if not trace_result.get('success'):
                failed += 1
                continue
            traces = trace_result.get('execution_traces', [{'steps': []}])
            trace = traces[0] if traces else {'steps': []}

            # 3. Semantic (Source code)
            source_codes = [sample['source_code']]

            with torch.no_grad():
                # Get features from each encoder
                static_features, _ = static_encoder([pdg], None)
                dynamic_features, _ = dynamic_encoder([trace], None)
                semantic_features, _ = semantic_encoder(source_codes, None)

                # Fuse features
                fused_features, vuln_scores = fusion_module(
                    static_features,
                    dynamic_features,
                    semantic_features,
                    None
                )

                # Get prediction
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                probs = torch.sigmoid(all_scores)
                pred_label = torch.argmax(probs, dim=1).item()

                predictions.append({
                    'filename': sample['filename'],
                    'true_label': sample['true_label'],
                    'true_label_idx': sample['true_label_idx'],
                    'predicted_label': VULNERABILITY_CLASSES[pred_label],
                    'predicted_label_idx': pred_label,
                    'confidence': probs[0, pred_label].item(),
                    'correct': pred_label == sample['true_label_idx']
                })
                successful += 1

        except Exception as e:
            logger.error(f"Fusion error on {sample['filename']}: {e}")
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
    """Compute comprehensive metrics."""
    predictions = results['predictions']

    if not predictions:
        return {
            'encoder': results['encoder'],
            'accuracy': 0.0,
            'total': results['total'],
            'successful': results['successful'],
            'failed': results['failed'],
            'correct': 0,
            'avg_f1': 0.0,
            'avg_precision': 0.0,
            'avg_recall': 0.0,
            'per_class': []
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
        tn = sum(confusion[other1][other2] for other1 in VULNERABILITY_CLASSES
                 for other2 in VULNERABILITY_CLASSES
                 if other1 != vuln_class and other2 != vuln_class)
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
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn),
                'support': int(support)
            })

    avg_f1 = np.mean([c['f1'] for c in per_class]) if per_class else 0.0
    avg_precision = np.mean([c['precision'] for c in per_class]) if per_class else 0.0
    avg_recall = np.mean([c['recall'] for c in per_class]) if per_class else 0.0

    return {
        'encoder': results['encoder'],
        'accuracy': accuracy,
        'total': results['total'],
        'successful': results['successful'],
        'failed': results['failed'],
        'correct': correct,
        'avg_f1': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'per_class': per_class
    }


def generate_report(all_results, output_file):
    """Generate comprehensive Markdown report."""

    report = []
    report.append("# Triton Comprehensive Test Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Test Dataset:** {all_results[0]['metrics']['total']} contracts\n")
    report.append("---\n\n")

    # Overall comparison
    report.append("## Overall Performance Comparison\n\n")
    report.append("| Model | Success Rate | Accuracy | Avg F1 | Avg Precision | Avg Recall | Correct/Total |\n")
    report.append("|-------|-------------|----------|--------|---------------|------------|---------------|\n")

    for result in all_results:
        m = result['metrics']
        success_rate = (m['successful'] / m['total'] * 100) if m['total'] > 0 else 0
        report.append(f"| **{m['encoder']}** | {success_rate:.1f}% | {m['accuracy']*100:.2f}% | "
                     f"{m['avg_f1']:.3f} | {m['avg_precision']:.3f} | {m['avg_recall']:.3f} | "
                     f"{m['correct']}/{m['successful']} |\n")

    report.append("\n")

    # Per-class F1 comparison
    report.append("## Per-Class F1 Score Comparison\n\n")
    report.append("| Vulnerability Type | ")
    for result in all_results:
        report.append(f"{result['metrics']['encoder']} | ")
    report.append("Best |\n")

    report.append("|" + "----|" * (len(all_results) + 2) + "\n")

    for vuln_class in VULNERABILITY_CLASSES:
        f1_scores = []
        row = f"| **{vuln_class}** | "

        for result in all_results:
            f1_val = None
            for cls in result['metrics']['per_class']:
                if cls['class'] == vuln_class:
                    f1_val = cls['f1']
                    break

            if f1_val is not None:
                row += f"{f1_val:.3f} | "
                f1_scores.append((f1_val, result['metrics']['encoder']))
            else:
                row += "N/A | "

        # Find best
        if f1_scores:
            best = max(f1_scores, key=lambda x: x[0])
            row += f"{best[1][:10]} |"
        else:
            row += "N/A |"

        report.append(row + "\n")

    report.append("\n")

    # Detailed per-model results
    for result in all_results:
        m = result['metrics']
        report.append(f"## {m['encoder']} - Detailed Results\n\n")
        report.append(f"**Success Rate:** {m['successful']}/{m['total']} ({m['successful']/m['total']*100:.1f}%)\n")
        report.append(f"**Accuracy:** {m['accuracy']*100:.2f}%\n")
        report.append(f"**Correct Predictions:** {m['correct']}/{m['successful']}\n\n")

        report.append("### Per-Class Metrics:\n\n")
        report.append("| Class | Precision | Recall | F1 | TP | FP | FN | Support |\n")
        report.append("|-------|-----------|--------|----|----|----|----|----------|\n")

        for cls in m['per_class']:
            report.append(f"| {cls['class']} | {cls['precision']:.3f} | {cls['recall']:.3f} | "
                         f"{cls['f1']:.3f} | {cls['tp']} | {cls['fp']} | {cls['fn']} | "
                         f"{cls['support']} |\n")

        report.append("\n")

    # Write to file
    with open(output_file, 'w') as f:
        f.writelines(report)

    logger.info(f"\n✓ Report saved to: {output_file}")


def main():
    """Main testing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive testing of all modalities")
    parser.add_argument(
        '--test-dir',
        default='data/datasets/combined_labeled/test',
        help='Test dataset directory'
    )
    parser.add_argument(
        '--output',
        default='COMPREHENSIVE_TEST_REPORT.md',
        help='Output report filename'
    )
    parser.add_argument(
        '--skip-static',
        action='store_true',
        help='Skip static encoder'
    )
    parser.add_argument(
        '--skip-dynamic',
        action='store_true',
        help='Skip dynamic encoder'
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

    # Initialize tools
    slither = SlitherWrapper(timeout=30)
    mythril = MythrilWrapper(timeout=30, max_depth=12)

    # Test each modality
    all_results = []

    # 1. Test Static Only
    if 'static' in models and not args.skip_static:
        result = test_single_encoder('static', models['static'], test_samples, device, slither=slither)
        metrics = compute_metrics(result)
        all_results.append({'type': 'static', 'raw': result, 'metrics': metrics})
        logger.info(f"\n✓ Static: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # 2. Test Dynamic Only
    if 'dynamic' in models and not args.skip_dynamic:
        result = test_single_encoder('dynamic', models['dynamic'], test_samples, device, mythril=mythril)
        metrics = compute_metrics(result)
        all_results.append({'type': 'dynamic', 'raw': result, 'metrics': metrics})
        logger.info(f"\n✓ Dynamic: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # 3. Test Semantic Only
    if 'semantic' in models:
        result = test_single_encoder('semantic', models['semantic'], test_samples, device)
        metrics = compute_metrics(result)
        all_results.append({'type': 'semantic', 'raw': result, 'metrics': metrics})
        logger.info(f"\n✓ Semantic: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # 4. Test Fusion Model
    if 'fusion' in models and not args.skip_fusion:
        result = test_fusion_model(models['fusion'], test_samples, device, slither, mythril)
        metrics = compute_metrics(result)
        all_results.append({'type': 'fusion', 'raw': result, 'metrics': metrics})
        logger.info(f"\n✓ Fusion: {metrics['successful']}/{metrics['total']} successful, "
                   f"Accuracy: {metrics['accuracy']*100:.2f}%")

    # Generate comprehensive report
    if all_results:
        generate_report(all_results, args.output)
        logger.info(f"\n{'='*80}")
        logger.info("FINAL SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"{'Model':<30} {'Accuracy':<12} {'Avg F1':<10}")
        logger.info("-"*80)
        for result in all_results:
            m = result['metrics']
            logger.info(f"{m['encoder']:<30} {m['accuracy']*100:>10.2f}% {m['avg_f1']:>8.3f}")
        logger.info("="*80)

    logger.info("\n✓ Comprehensive testing complete!")
    logger.info(f"✓ Report saved to: {args.output}")


if __name__ == '__main__':
    main()
