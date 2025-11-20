#!/usr/bin/env python3
"""
Test All Triton Models
Tests Static, Dynamic, Semantic encoders individually and the Full Fusion model
on the test dataset.
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
from torch.utils.data import DataLoader

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

    for vuln_class in VULNERABILITY_CLASSES:
        vuln_dir = test_dir / vuln_class
        if not vuln_dir.exists():
            continue

        sol_files = list(vuln_dir.glob('*.sol'))
        logger.info(f"Found {len(sol_files)} {vuln_class} contracts")

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

    logger.info(f"Loaded {len(test_samples)} test samples total")
    return test_samples


def load_models(device):
    """Load all trained models."""
    logger.info("Loading trained models...")

    checkpoints_dir = Path('models/checkpoints')

    models = {}

    # Load Static Encoder
    static_path = checkpoints_dir / 'static_encoder_best.pt'
    if static_path.exists():
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
    else:
        logger.warning(f"Static encoder checkpoint not found: {static_path}")

    # Load Dynamic Encoder
    dynamic_path = checkpoints_dir / 'dynamic_encoder_best.pt'
    if dynamic_path.exists():
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
    else:
        logger.warning(f"Dynamic encoder checkpoint not found: {dynamic_path}")

    # Load Semantic Encoder
    semantic_path = checkpoints_dir / 'semantic_encoder_best.pt'
    if semantic_path.exists():
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
    else:
        logger.warning(f"Semantic encoder checkpoint not found: {semantic_path}")

    # Load Fusion Model (all components)
    fusion_path = checkpoints_dir / 'fusion_module_best.pt'
    static_fusion_path = checkpoints_dir / 'static_encoder_fusion_best.pt'
    dynamic_fusion_path = checkpoints_dir / 'dynamic_encoder_fusion_best.pt'
    semantic_fusion_path = checkpoints_dir / 'semantic_encoder_fusion_best.pt'

    if all([fusion_path.exists(), static_fusion_path.exists(),
            dynamic_fusion_path.exists(), semantic_fusion_path.exists()]):

        # Load fusion components
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
    else:
        logger.warning(f"Fusion model checkpoints incomplete")

    return models


def test_semantic_encoder(semantic_encoder, test_samples, device):
    """Test semantic encoder on test samples."""
    logger.info("\n" + "="*80)
    logger.info("TESTING SEMANTIC ENCODER")
    logger.info("="*80)

    predictions = []
    correct = 0
    total = 0

    # Process in small batches
    batch_size = 4
    for i in range(0, len(test_samples), batch_size):
        batch = test_samples[i:i+batch_size]
        source_codes = [s['source_code'] for s in batch]
        true_labels = [s['true_label_idx'] for s in batch]

        try:
            with torch.no_grad():
                # Get semantic features and vulnerability scores
                semantic_features, vuln_scores = semantic_encoder(source_codes, None)

                # Aggregate scores across all vulnerability heads
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                # Get predictions
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
                        'predicted_label': VULNERABILITY_CLASSES[pred_label],
                        'confidence': float(confidence),
                        'correct': pred_label == true_label
                    })

                    if pred_label == true_label:
                        correct += 1
                    total += 1

        except Exception as e:
            logger.error(f"Error processing batch {i}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0

    logger.info(f"\nSemantic Encoder Results:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")

    # Per-class metrics
    compute_per_class_metrics(predictions, "Semantic")

    return {
        'encoder': 'semantic',
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'predictions': predictions
    }


def compute_per_class_metrics(predictions, encoder_name):
    """Compute and display per-class metrics."""

    # Initialize confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))

    for pred in predictions:
        confusion[pred['true_label']][pred['predicted_label']] += 1

    logger.info(f"\nPer-class Performance ({encoder_name}):")
    logger.info("-" * 80)
    logger.info(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    logger.info("-" * 80)

    all_f1_scores = []

    for vuln_class in VULNERABILITY_CLASSES:
        # Calculate metrics
        tp = confusion[vuln_class][vuln_class]
        fp = sum(confusion[other][vuln_class] for other in VULNERABILITY_CLASSES if other != vuln_class)
        fn = sum(confusion[vuln_class][other] for other in VULNERABILITY_CLASSES if other != vuln_class)

        support = tp + fn

        if support == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        logger.info(f"{vuln_class:<30} {precision:<12.3f} {recall:<12.3f} {f1:<12.3f} {support:<10}")

        all_f1_scores.append(f1)

    avg_f1 = np.mean(all_f1_scores) if all_f1_scores else 0.0
    logger.info("-" * 80)
    logger.info(f"Average F1 Score: {avg_f1:.3f}")


def test_fusion_model(fusion_models, test_samples, device):
    """Test full fusion model."""
    logger.info("\n" + "="*80)
    logger.info("TESTING FUSION MODEL (Full Multi-Modal)")
    logger.info("="*80)

    static_encoder = fusion_models['static']
    dynamic_encoder = fusion_models['dynamic']
    semantic_encoder = fusion_models['semantic']
    fusion_module = fusion_models['fusion']

    predictions = []
    correct = 0
    total = 0

    # Import tools for feature extraction
    from tools.slither_wrapper import SlitherWrapper
    from tools.mythril_wrapper import MythrilWrapper

    slither = SlitherWrapper(timeout=30)
    mythril = MythrilWrapper(timeout=30, max_depth=12)

    for sample in test_samples:
        try:
            # Extract features
            source_code = sample['source_code']
            true_label = sample['true_label_idx']

            # Static features (PDG)
            pdg_result = slither.analyze_contract(source_code)
            pdgs = [pdg_result.get('pdg')] if pdg_result.get('success') else [None]

            # Dynamic features (traces)
            trace_result = mythril.analyze_contract(source_code)
            traces = [trace_result.get('execution_traces', [{'steps': []}])] if trace_result.get('success') else [[{'steps': []}]]

            with torch.no_grad():
                # Get features from all encoders
                static_features, _ = static_encoder(pdgs, None)

                all_traces = []
                for trace_list in traces:
                    if trace_list:
                        all_traces.append(trace_list[0] if trace_list else {'steps': []})
                    else:
                        all_traces.append({'steps': []})

                dynamic_features, _ = dynamic_encoder(all_traces, None)
                semantic_features, _ = semantic_encoder([source_code], None)

                # Fusion
                fusion_output = fusion_module(
                    static_features,
                    dynamic_features,
                    semantic_features,
                    None
                )

                vulnerability_logits = fusion_output['vulnerability_logits']
                probs = torch.softmax(vulnerability_logits, dim=1)
                pred_label = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_label].item()

                predictions.append({
                    'path': sample['path'],
                    'true_label': sample['true_label'],
                    'predicted_label': VULNERABILITY_CLASSES[pred_label],
                    'confidence': float(confidence),
                    'correct': pred_label == true_label
                })

                if pred_label == true_label:
                    correct += 1
                total += 1

                logger.info(f"[{total}/{len(test_samples)}] {Path(sample['path']).name}: "
                          f"True={sample['true_label']}, Pred={VULNERABILITY_CLASSES[pred_label]}, "
                          f"Conf={confidence:.2f}, {'✓' if pred_label == true_label else '✗'}")

        except Exception as e:
            logger.error(f"Error processing {sample['path']}: {e}")
            continue

    accuracy = correct / total if total > 0 else 0

    logger.info(f"\nFusion Model Results:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Correct predictions: {correct}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")

    # Per-class metrics
    compute_per_class_metrics(predictions, "Fusion")

    return {
        'encoder': 'fusion',
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'predictions': predictions
    }


def main():
    """Main testing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test all Triton models")
    parser.add_argument(
        '--test-dir',
        default='data/datasets/combined_labeled/test',
        help='Test dataset directory'
    )
    parser.add_argument(
        '--output-dir',
        default='results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['semantic', 'static', 'dynamic', 'fusion', 'all'],
        default=['all'],
        help='Which models to test'
    )

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load test dataset
    test_samples = load_test_dataset(args.test_dir)

    if not test_samples:
        logger.error("No test samples found!")
        return

    # Load models
    models = load_models(device)

    if not models:
        logger.error("No models loaded!")
        return

    # Determine which models to test
    models_to_test = args.models
    if 'all' in models_to_test:
        models_to_test = ['semantic', 'fusion']  # Default to these since static/dynamic need working tools

    # Test each model
    all_results = {}

    if 'semantic' in models_to_test and 'semantic' in models:
        results = test_semantic_encoder(models['semantic'], test_samples, device)
        all_results['semantic'] = results

    if 'fusion' in models_to_test and 'fusion' in models:
        results = test_fusion_model(models['fusion'], test_samples, device)
        all_results['fusion'] = results

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f'test_results_{timestamp}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_file}")

    # Print comparison
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPARISON")
    logger.info("="*80)
    logger.info(f"{'Model':<20} {'Accuracy':<15} {'Correct/Total':<20}")
    logger.info("-"*80)

    for model_name, results in all_results.items():
        logger.info(f"{model_name.capitalize():<20} {results['accuracy']*100:>12.2f}% {results['correct']:>8}/{results['total']:<10}")

    logger.info("="*80)


if __name__ == '__main__':
    main()
