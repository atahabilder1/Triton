#!/usr/bin/env python3
"""
Test Individual Encoder Performance
Tests each encoder (Static, Dynamic, Semantic) separately on the test set.
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
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


def load_encoder_model(encoder_type, checkpoint_path, device):
    """Load a specific encoder model."""
    logger.info(f"Loading {encoder_type} encoder from {checkpoint_path}")

    if encoder_type == 'static':
        encoder = StaticEncoder(
            node_feature_dim=256,
            hidden_dim=128,
            output_dim=128,
            num_classes=len(VULNERABILITY_CLASSES)
        ).to(device)
    elif encoder_type == 'dynamic':
        encoder = DynamicEncoder(
            input_dim=256,
            hidden_dim=128,
            output_dim=128,
            num_classes=len(VULNERABILITY_CLASSES)
        ).to(device)
    elif encoder_type == 'semantic':
        encoder = SemanticEncoder(
            model_name='microsoft/codebert-base',
            num_classes=len(VULNERABILITY_CLASSES)
        ).to(device)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        encoder.load_state_dict(checkpoint)
        logger.info(f"✓ Loaded {encoder_type} encoder successfully")
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None

    encoder.eval()
    return encoder


def predict_with_encoder(encoder, encoder_type, contract_path, device):
    """Make prediction using a specific encoder."""

    try:
        if encoder_type == 'static':
            # Get PDG from Slither
            slither = SlitherWrapper()
            pdg = slither.analyze(contract_path)

            if not pdg or 'nodes' not in pdg or len(pdg['nodes']) == 0:
                logger.warning(f"Empty PDG for {contract_path}")
                return None, None

            # Create graph data
            from torch_geometric.data import Data
            node_features = torch.randn(len(pdg['nodes']), 256).to(device)
            edge_index = torch.tensor(pdg.get('edges', []), dtype=torch.long).t().contiguous().to(device)

            if edge_index.numel() == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

            graph_data = Data(x=node_features, edge_index=edge_index)

            with torch.no_grad():
                logits = encoder(graph_data)
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_class].item()

            return pred_class, confidence

        elif encoder_type == 'dynamic':
            # Get execution trace from Mythril
            mythril = MythrilWrapper()
            trace = mythril.analyze(contract_path)

            if not trace or len(trace) == 0:
                logger.warning(f"Empty trace for {contract_path}")
                return None, None

            # Create trace tensor
            trace_tensor = torch.randn(1, min(len(trace), 100), 256).to(device)

            with torch.no_grad():
                logits = encoder(trace_tensor)
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_class].item()

            return pred_class, confidence

        elif encoder_type == 'semantic':
            # Read source code
            with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            # Tokenize and encode
            from transformers import RobertaTokenizer
            tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

            inputs = tokenizer(
                source_code,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding='max_length'
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = encoder(**inputs)
                probs = torch.softmax(logits, dim=-1)
                pred_class = torch.argmax(probs, dim=-1).item()
                confidence = probs[0, pred_class].item()

            return pred_class, confidence

    except Exception as e:
        logger.error(f"Error in {encoder_type} prediction for {contract_path}: {e}")
        return None, None


def get_ground_truth(contract_path):
    """Extract ground truth label from directory structure."""
    parts = Path(contract_path).parts

    # Find the vulnerability category directory
    for part in reversed(parts):
        if part in VULNERABILITY_CLASSES:
            return VULNERABILITY_CLASSES.index(part)

    return None


def test_encoder(encoder_type, checkpoint_path, test_dir, device):
    """Test a single encoder on the test set."""

    logger.info(f"\n{'='*80}")
    logger.info(f"Testing {encoder_type.upper()} Encoder")
    logger.info(f"{'='*80}")

    # Load encoder
    encoder = load_encoder_model(encoder_type, checkpoint_path, device)
    if encoder is None:
        logger.error(f"Failed to load {encoder_type} encoder")
        return None

    # Collect all test contracts
    test_contracts = []
    for vuln_type in VULNERABILITY_CLASSES:
        vuln_dir = Path(test_dir) / vuln_type
        if vuln_dir.exists():
            test_contracts.extend(list(vuln_dir.glob('*.sol')))

    logger.info(f"Found {len(test_contracts)} test contracts")

    # Test each contract
    results = {
        'encoder_type': encoder_type,
        'checkpoint': checkpoint_path,
        'total_contracts': len(test_contracts),
        'successful_predictions': 0,
        'failed_predictions': 0,
        'predictions': [],
        'confusion_matrix': defaultdict(lambda: defaultdict(int)),
        'per_class_metrics': {}
    }

    for i, contract_path in enumerate(test_contracts, 1):
        logger.info(f"[{i}/{len(test_contracts)}] Testing {contract_path.name}")

        # Get ground truth
        true_label = get_ground_truth(str(contract_path))
        if true_label is None:
            logger.warning(f"Could not determine ground truth for {contract_path}")
            continue

        # Predict
        pred_label, confidence = predict_with_encoder(encoder, encoder_type, str(contract_path), device)

        if pred_label is not None:
            results['successful_predictions'] += 1
            results['predictions'].append({
                'contract': str(contract_path),
                'true_label': VULNERABILITY_CLASSES[true_label],
                'predicted_label': VULNERABILITY_CLASSES[pred_label],
                'confidence': confidence
            })
            results['confusion_matrix'][VULNERABILITY_CLASSES[true_label]][VULNERABILITY_CLASSES[pred_label]] += 1
        else:
            results['failed_predictions'] += 1

    # Calculate metrics
    calculate_metrics(results)

    return results


def calculate_metrics(results):
    """Calculate accuracy, precision, recall, F1 for the encoder."""

    # Overall accuracy
    correct = sum(1 for p in results['predictions'] if p['true_label'] == p['predicted_label'])
    total = len(results['predictions'])
    results['accuracy'] = correct / total if total > 0 else 0.0

    logger.info(f"\n{'='*80}")
    logger.info(f"RESULTS: {results['encoder_type'].upper()} Encoder")
    logger.info(f"{'='*80}")
    logger.info(f"Total Contracts: {results['total_contracts']}")
    logger.info(f"Successful Predictions: {results['successful_predictions']}")
    logger.info(f"Failed Predictions: {results['failed_predictions']}")
    logger.info(f"Overall Accuracy: {results['accuracy']*100:.2f}%")

    # Per-class metrics
    logger.info(f"\n{'='*80}")
    logger.info(f"Per-Class Performance")
    logger.info(f"{'='*80}")

    for vuln_class in VULNERABILITY_CLASSES:
        tp = results['confusion_matrix'][vuln_class][vuln_class]
        fp = sum(results['confusion_matrix'][other][vuln_class] for other in VULNERABILITY_CLASSES if other != vuln_class)
        fn = sum(results['confusion_matrix'][vuln_class][other] for other in VULNERABILITY_CLASSES if other != vuln_class)
        tn = sum(
            results['confusion_matrix'][other1][other2]
            for other1 in VULNERABILITY_CLASSES if other1 != vuln_class
            for other2 in VULNERABILITY_CLASSES if other2 != vuln_class
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results['per_class_metrics'][vuln_class] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': tp + fn
        }

        if tp + fn > 0:  # Only show classes that appear in test set
            logger.info(f"{vuln_class:30s} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f} | Support: {tp+fn}")

    # Average F1
    f1_scores = [m['f1'] for m in results['per_class_metrics'].values() if m['support'] > 0]
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    results['avg_f1'] = avg_f1
    logger.info(f"\nAverage F1 Score: {avg_f1:.3f}")


def main():
    """Main testing function."""

    # Configuration
    test_dir = project_root / 'data' / 'datasets' / 'combined_labeled' / 'test'
    checkpoints_dir = project_root / 'models' / 'checkpoints'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Test each encoder
    encoders_to_test = [
        ('semantic', checkpoints_dir / 'semantic_encoder_best.pt'),
        ('static', checkpoints_dir / 'static_encoder_best.pt'),
        ('dynamic', checkpoints_dir / 'dynamic_encoder_best.pt'),
    ]

    all_results = {}

    for encoder_type, checkpoint_path in encoders_to_test:
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            continue

        results = test_encoder(encoder_type, str(checkpoint_path), str(test_dir), device)
        if results:
            all_results[encoder_type] = results

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = project_root / 'results' / f'individual_encoder_results_{timestamp}.json'
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        # Convert defaultdict to dict for JSON serialization
        serializable_results = {}
        for encoder_type, results in all_results.items():
            serializable_results[encoder_type] = {
                **results,
                'confusion_matrix': {k: dict(v) for k, v in results['confusion_matrix'].items()}
            }
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\n✓ Results saved to: {output_file}")

    # Print comparison
    print("\n" + "="*80)
    print("ENCODER COMPARISON")
    print("="*80)
    print(f"{'Encoder':<15} {'Accuracy':<12} {'Avg F1':<12} {'Successful':<15}")
    print("-"*80)
    for encoder_type, results in all_results.items():
        print(f"{encoder_type.capitalize():<15} {results['accuracy']*100:>10.2f}% {results['avg_f1']:>10.3f} {results['successful_predictions']:>14}")
    print("="*80)


if __name__ == '__main__':
    main()
