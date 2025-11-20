#!/usr/bin/env python3
"""
Static-Only Vulnerability Detection Training
Trains only the static encoder with detailed per-vulnerability metrics.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score
)
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from encoders.static_encoder import StaticEncoder
from tools.slither_wrapper import SlitherWrapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StaticDataset(Dataset):
    """Dataset for static analysis using PDG only"""

    def __init__(
        self,
        contracts_dir: str,
        max_samples: Optional[int] = None,
        use_cache: bool = True,
        cache_dir: str = "data/cache"
    ):
        self.contracts = []
        self.labels = []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        contracts_path = Path(contracts_dir)

        # Vulnerability type mapping
        self.vuln_types = {
            'access_control': 0,
            'arithmetic': 1,
            'bad_randomness': 2,
            'denial_of_service': 3,
            'front_running': 4,
            'reentrancy': 5,
            'short_addresses': 6,
            'time_manipulation': 7,
            'unchecked_low_level_calls': 8,
            'other': 9,
            'safe': 10
        }

        logger.info(f"Loading contracts from {contracts_dir}...")

        # Load from organized dataset
        if (contracts_path / 'reentrancy').exists():
            for vuln_type, label in self.vuln_types.items():
                vuln_dir = contracts_path / vuln_type
                if not vuln_dir.exists():
                    continue

                sol_files = list(vuln_dir.glob("*.sol"))
                logger.info(f"Found {len(sol_files)} {vuln_type} contracts")

                for contract_file in sol_files:
                    try:
                        with open(contract_file, 'r', encoding='utf-8', errors='ignore') as f:
                            source_code = f.read()

                        self.contracts.append({
                            'source_code': source_code,
                            'path': str(contract_file),
                            'vulnerability_type': vuln_type
                        })
                        self.labels.append(label)

                        if max_samples and len(self.contracts) >= max_samples:
                            break
                    except Exception as e:
                        logger.warning(f"Error loading {contract_file}: {e}")
                        continue

                if max_samples and len(self.contracts) >= max_samples:
                    break

        logger.info(f"Loaded {len(self.contracts)} contracts total")

        # Print distribution
        label_counts = defaultdict(int)
        for label in self.labels:
            label_counts[label] += 1

        logger.info("Label distribution:")
        for vuln_type, label in sorted(self.vuln_types.items(), key=lambda x: x[1]):
            count = label_counts[label]
            if count > 0:
                logger.info(f"  {vuln_type}: {count} contracts")

        # Initialize Slither
        self.slither = SlitherWrapper(timeout=60)

    def __len__(self):
        return len(self.contracts)

    def _get_cache_path(self, idx: int) -> Path:
        """Get cache file path for PDG"""
        contract_hash = hash(self.contracts[idx]['path'])
        return self.cache_dir / f"{contract_hash}_pdg.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[nx.DiGraph]:
        """Load PDG from cache"""
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)

                pdg = nx.DiGraph()
                if 'nodes' in data and 'edges' in data:
                    for node, attrs in data['nodes']:
                        pdg.add_node(node, **attrs)
                    for src, tgt, attrs in data['edges']:
                        pdg.add_edge(src, tgt, **attrs)
                return pdg
            except:
                return None
        return None

    def _save_to_cache(self, cache_path: Path, pdg: nx.DiGraph):
        """Save PDG to cache"""
        try:
            data = {
                'nodes': list(pdg.nodes(data=True)),
                'edges': list(pdg.edges(data=True))
            }
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache PDG: {e}")

    def _extract_pdg(self, source_code: str, idx: int) -> nx.DiGraph:
        """Extract PDG using Slither with caching"""
        cache_path = self._get_cache_path(idx)

        if self.use_cache:
            cached = self._load_from_cache(cache_path)
            if cached is not None:
                return cached

        # Extract using Slither
        result = self.slither.analyze_contract(source_code)

        if result.get('success') and result.get('pdg'):
            pdg = result['pdg']
            if self.use_cache:
                self._save_to_cache(cache_path, pdg)
            return pdg

        # Return empty graph if extraction failed
        return nx.DiGraph()

    def __getitem__(self, idx):
        contract = self.contracts[idx]
        label = self.labels[idx]

        # Extract PDG for static analysis
        pdg = self._extract_pdg(contract['source_code'], idx)

        return {
            'pdg': pdg,
            'path': contract['path'],
            'vulnerability_type': contract['vulnerability_type'],
            'label': label
        }


def collate_fn(batch):
    """Custom collate function for PDGs"""
    return {
        'pdg': [item['pdg'] for item in batch],
        'path': [item['path'] for item in batch],
        'vulnerability_type': [item['vulnerability_type'] for item in batch],
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long)
    }


class StaticVulnerabilityDetector:
    """Static-only vulnerability detector with detailed metrics"""

    def __init__(
        self,
        output_dir: str = "models/checkpoints",
        device: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 20,
        class_weights: Optional[torch.Tensor] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Vulnerability type mapping
        self.vuln_types = {
            0: 'access_control',
            1: 'arithmetic',
            2: 'bad_randomness',
            3: 'denial_of_service',
            4: 'front_running',
            5: 'reentrancy',
            6: 'short_addresses',
            7: 'time_manipulation',
            8: 'unchecked_low_level_calls',
            9: 'other',
            10: 'safe'
        }

        # Initialize model
        logger.info("Initializing Static Encoder...")
        self.model = StaticEncoder(
            node_feature_dim=128,
            hidden_dim=256,
            output_dim=768,
            dropout=0.2
        ).to(self.device)

        # Loss function with class weighting
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"Using class-weighted loss function")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"runs/static_only_{timestamp}")
        logger.info(f"TensorBoard logging: runs/static_only_{timestamp}")

    def print_detailed_metrics(self, all_preds: List[int], all_labels: List[int], phase_name: str):
        """Print comprehensive per-vulnerability metrics"""

        logger.info("\n" + "="*100)
        logger.info(f"{phase_name} - DETAILED VULNERABILITY DETECTION METRICS")
        logger.info("="*100)

        # Overall metrics
        accuracy = accuracy_score(all_labels, all_preds)

        # Calculate overall precision, recall, F1
        overall_precision, overall_recall, overall_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted', zero_division=0
        )

        # Macro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        logger.info(f"\n{'='*100}")
        logger.info(f"OVERALL PERFORMANCE METRICS")
        logger.info(f"{'='*100}")
        logger.info(f"{'Metric':<30} {'Score':>15} {'Percentage':>15}")
        logger.info(f"{'-'*100}")
        logger.info(f"{'Overall Accuracy':<30} {accuracy:>15.4f} {accuracy*100:>14.2f}%")
        logger.info(f"{'Overall Precision (Weighted)':<30} {overall_precision:>15.4f} {overall_precision*100:>14.2f}%")
        logger.info(f"{'Overall Recall (Weighted)':<30} {overall_recall:>15.4f} {overall_recall*100:>14.2f}%")
        logger.info(f"{'Overall F1-Score (Weighted)':<30} {overall_f1:>15.4f} {overall_f1*100:>14.2f}%")
        logger.info(f"{'-'*100}")
        logger.info(f"{'Macro Precision':<30} {macro_precision:>15.4f} {macro_precision*100:>14.2f}%")
        logger.info(f"{'Macro Recall':<30} {macro_recall:>15.4f} {macro_recall*100:>14.2f}%")
        logger.info(f"{'Macro F1-Score':<30} {macro_f1:>15.4f} {macro_f1*100:>14.2f}%")
        logger.info(f"{'='*100}\n")

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(11)), zero_division=0
        )

        logger.info(f"{'='*100}")
        logger.info(f"PER-VULNERABILITY TYPE DETECTION RATES")
        logger.info(f"{'='*100}")
        logger.info(f"{'Vulnerability Type':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Detected/Total':>15}")
        logger.info("-" * 100)

        total_detected = 0
        total_samples = 0

        for i in range(11):
            if support[i] > 0:
                vuln_name = self.vuln_types[i]
                detected = sum(1 for pred, label in zip(all_preds, all_labels) if label == i and pred == i)
                total = support[i]

                logger.info(
                    f"{vuln_name:<30} "
                    f"{precision[i]:>10.4f} {recall[i]:>10.4f} "
                    f"{f1[i]:>10.4f} {detected:>6}/{total:<7} ({recall[i]*100:.1f}%)"
                )

                total_detected += detected
                total_samples += total

        logger.info("-" * 100)
        logger.info(f"{'TOTAL DETECTION RATE':<30} {'':>10} {'':>10} {'':>10} {total_detected:>6}/{total_samples:<7} ({total_detected/total_samples*100:.2f}%)")
        logger.info("="*100 + "\n")

        return {
            'accuracy': accuracy,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_f1': overall_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'support': support
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train static encoder"""
        logger.info("="*100)
        logger.info("TRAINING STATIC VULNERABILITY DETECTOR")
        logger.info("="*100)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        best_val_f1 = 0.0

        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"EPOCH {epoch+1}/{self.num_epochs}")
            logger.info(f"{'='*50}")

            # Training phase
            self.model.train()
            train_loss = 0
            train_preds = []
            train_labels = []

            pbar = tqdm(train_loader, desc=f"Training")
            for batch in pbar:
                pdgs = batch['pdg']
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                try:
                    static_features, vuln_scores = self.model(pdgs, None)

                    # Aggregate vulnerability scores for classification
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(all_scores, 1)

                    train_preds.extend(predicted.cpu().numpy().tolist())
                    train_labels.extend(labels.cpu().numpy().tolist())

                    pbar.set_postfix({'loss': loss.item()})
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
            train_acc = accuracy_score(train_labels, train_preds)

            # Validation phase
            val_loss, val_preds, val_labels = self.validate(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

            logger.info(f"\nTrain Loss: {avg_train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%, Val F1: {val_f1:.4f}")

            # TensorBoard logging
            self.writer.add_scalar('Train/Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Val/F1', val_f1, epoch)

            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.save_checkpoint(epoch, val_loss, val_acc, val_f1)
                logger.info(f"✓ Saved best model (F1: {val_f1:.4f})")

            # Print detailed metrics on last epoch
            if epoch == self.num_epochs - 1:
                self.print_detailed_metrics(val_preds, val_labels, "FINAL VALIDATION")

        logger.info(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")

    def validate(self, val_loader: DataLoader) -> Tuple[float, List[int], List[int]]:
        """Validate model"""
        self.model.eval()

        val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                pdgs = batch['pdg']
                labels = batch['label'].to(self.device)

                try:
                    static_features, vuln_scores = self.model(pdgs, None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(all_scores, 1)
                    all_preds.extend(predicted.cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                except:
                    continue

        avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else 0
        return avg_val_loss, all_preds, all_labels

    def test(self, test_loader: DataLoader):
        """Test model and print detailed metrics"""
        logger.info("\n" + "="*100)
        logger.info("TESTING STATIC VULNERABILITY DETECTOR")
        logger.info("="*100)

        test_loss, test_preds, test_labels = self.validate(test_loader)

        # Print comprehensive metrics
        metrics = self.print_detailed_metrics(test_preds, test_labels, "TEST SET")

        # Save results to file
        results_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("STATIC VULNERABILITY DETECTION - TEST RESULTS\n")
            f.write("="*100 + "\n\n")

            # Overall metrics
            f.write("="*100 + "\n")
            f.write("OVERALL PERFORMANCE METRICS\n")
            f.write("="*100 + "\n")
            f.write(f"{'Metric':<30} {'Score':>15} {'Percentage':>15}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Overall Accuracy':<30} {metrics['accuracy']:>15.4f} {metrics['accuracy']*100:>14.2f}%\n")
            f.write(f"{'Overall Precision (Weighted)':<30} {metrics['overall_precision']:>15.4f} {metrics['overall_precision']*100:>14.2f}%\n")
            f.write(f"{'Overall Recall (Weighted)':<30} {metrics['overall_recall']:>15.4f} {metrics['overall_recall']*100:>14.2f}%\n")
            f.write(f"{'Overall F1-Score (Weighted)':<30} {metrics['overall_f1']:>15.4f} {metrics['overall_f1']*100:>14.2f}%\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Macro Precision':<30} {metrics['macro_precision']:>15.4f} {metrics['macro_precision']*100:>14.2f}%\n")
            f.write(f"{'Macro Recall':<30} {metrics['macro_recall']:>15.4f} {metrics['macro_recall']*100:>14.2f}%\n")
            f.write(f"{'Macro F1-Score':<30} {metrics['macro_f1']:>15.4f} {metrics['macro_f1']*100:>14.2f}%\n")
            f.write("="*100 + "\n\n")

            # Per-vulnerability metrics
            f.write("="*100 + "\n")
            f.write("PER-VULNERABILITY TYPE DETECTION RATES\n")
            f.write("="*100 + "\n")
            f.write(f"{'Vulnerability Type':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
            f.write("-"*100 + "\n")

            for i in range(11):
                if metrics['support'][i] > 0:
                    vuln_name = self.vuln_types[i]
                    f.write(
                        f"{vuln_name:<30} "
                        f"{metrics['per_class_precision'][i]:>10.4f} {metrics['per_class_recall'][i]:>10.4f} "
                        f"{metrics['per_class_f1'][i]:>10.4f} {int(metrics['support'][i]):>10}\n"
                    )
            f.write("="*100 + "\n")

        logger.info(f"\nResults saved to: {results_file}")
        return metrics

    def save_checkpoint(self, epoch, val_loss, val_acc, val_f1):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        }
        path = self.output_dir / "static_encoder_best.pt"
        torch.save(checkpoint, path)


def calculate_class_weights(dataset: StaticDataset, num_classes: int = 11) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    class_counts = torch.zeros(num_classes)
    for label in dataset.labels:
        class_counts[label] += 1

    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    logger.info("\nClass weights:")
    vuln_types_inv = {v: k for k, v in dataset.vuln_types.items()}
    for i in range(num_classes):
        if class_counts[i] > 0:
            vuln_name = vuln_types_inv.get(i, f"class_{i}")
            logger.info(f"  {vuln_name}: count={int(class_counts[i])}, weight={class_weights[i]:.4f}")

    return class_weights


def main():
    parser = argparse.ArgumentParser(description="Static-Only Vulnerability Detection Training")
    parser.add_argument("--train-dir", required=True, help="Training data directory")
    parser.add_argument("--val-dir", required=True, help="Validation data directory")
    parser.add_argument("--test-dir", required=True, help="Test data directory")
    parser.add_argument("--output-dir", default="models/checkpoints", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to use")
    parser.add_argument("--no-cache", action='store_true', help="Disable caching")

    args = parser.parse_args()

    logger.info("="*100)
    logger.info("STATIC VULNERABILITY DETECTION - TRAINING PIPELINE")
    logger.info("="*100)
    logger.info(f"Train: {args.train_dir}")
    logger.info(f"Val: {args.val_dir}")
    logger.info(f"Test: {args.test_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")

    # Load datasets
    logger.info("\nLoading datasets...")
    train_dataset = StaticDataset(args.train_dir, max_samples=args.max_samples, use_cache=not args.no_cache)
    val_dataset = StaticDataset(args.val_dir, use_cache=not args.no_cache)
    test_dataset = StaticDataset(args.test_dir, use_cache=not args.no_cache)

    logger.info(f"\nTrain: {len(train_dataset)} contracts")
    logger.info(f"Val: {len(val_dataset)} contracts")
    logger.info(f"Test: {len(test_dataset)} contracts")

    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate_fn)

    # Initialize detector
    detector = StaticVulnerabilityDetector(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        class_weights=class_weights
    )

    # Train
    detector.train(train_loader, val_loader)

    # Test
    detector.test(test_loader)

    logger.info("\n" + "="*100)
    logger.info("TRAINING AND TESTING COMPLETE!")
    logger.info("="*100)
    logger.info(f"Model saved to: {args.output_dir}/static_encoder_best.pt")
    logger.info(f"TensorBoard: tensorboard --logdir runs/")


if __name__ == "__main__":
    main()
