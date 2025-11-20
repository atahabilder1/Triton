#!/usr/bin/env python3
"""
Optimized Static Vulnerability Detection Training
- Uses full GPU power with optimal batch size
- Multi-worker data loading for CPU parallelism
- Real-time training progress monitoring
- Early stopping if model isn't improving
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
import time

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
    accuracy_score
)
from torch.utils.tensorboard import SummaryWriter

# Add project root directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from encoders.static_encoder import StaticEncoder
from tools.slither_wrapper import SlitherWrapper
from utils.config import get_config

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
        cache_dir: Optional[str] = None
    ):
        self.contracts = []
        self.labels = []

        # Use config for cache_dir if not provided
        if cache_dir is None:
            config = get_config()
            cache_dir = str(config.cache_dir)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        contracts_path = Path(contracts_dir)

        # Vulnerability type mapping (DYNAMIC - detect from dataset)
        # Full mapping for reference
        self.all_vuln_types = {
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

        # Detect which classes actually exist in dataset
        logger.info(f"Loading contracts from {contracts_dir}...")
        logger.info("Detecting vulnerability classes in dataset...")

        contracts_path = Path(contracts_dir)
        available_classes = []
        for vuln_type in self.all_vuln_types.keys():
            if (contracts_path / vuln_type).exists():
                available_classes.append(vuln_type)

        # Create dynamic mapping based on available classes
        self.vuln_types = {vuln: idx for idx, vuln in enumerate(sorted(available_classes))}
        self.num_classes = len(self.vuln_types)

        logger.info(f"\n{'='*80}")
        logger.info(f"DETECTED {self.num_classes} VULNERABILITY CLASSES:")
        logger.info(f"{'='*80}")
        for vuln_type, label in sorted(self.vuln_types.items(), key=lambda x: x[1]):
            logger.info(f"  [{label}] {vuln_type}")
        logger.info(f"{'='*80}\n")

        # Load from organized dataset
        if len(available_classes) > 0:
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

        logger.info(f"✓ Loaded {len(self.contracts)} contracts total")

        # Print distribution
        label_counts = defaultdict(int)
        for label in self.labels:
            label_counts[label] += 1

        logger.info("\n" + "="*80)
        logger.info("DATASET DISTRIBUTION")
        logger.info("="*80)
        for vuln_type, label in sorted(self.vuln_types.items(), key=lambda x: x[1]):
            count = label_counts[label]
            if count > 0:
                percentage = (count / len(self.contracts)) * 100
                logger.info(f"  {vuln_type:<30} {count:>5} contracts ({percentage:>5.2f}%)")
        logger.info("="*80 + "\n")

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

        # Extract using Slither - pass contract path for failure logging
        contract_path = self.contracts[idx]['path']
        result = self.slither.analyze_contract(source_code, contract_path=contract_path)

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


class TrainingMonitor:
    """Monitor training progress and detect if model is improving"""

    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.best_f1 = 0.0
        self.counter = 0
        self.should_stop = False

    def update(self, val_loss: float, val_f1: float) -> bool:
        """Returns True if training should stop"""
        improved = False

        # Check if validation loss improved
        if val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.counter = 0
            improved = True

        # Check if F1 score improved
        if val_f1 > (self.best_f1 + self.min_delta):
            self.best_f1 = val_f1
            self.counter = 0
            improved = True

        if not improved:
            self.counter += 1
            logger.warning(f"⚠️  No improvement for {self.counter}/{self.patience} epochs")

            if self.counter >= self.patience:
                logger.warning(f"🛑 EARLY STOPPING: No improvement for {self.patience} epochs")
                return True

        return False


class StaticVulnerabilityDetector:
    """Optimized static vulnerability detector"""

    def __init__(
        self,
        num_classes: int,
        vuln_types_map: Dict[int, str],
        output_dir: Optional[str] = None,
        device: str = None,
        learning_rate: Optional[float] = None,
        batch_size: Optional[int] = None,
        num_epochs: Optional[int] = None,
        class_weights: Optional[torch.Tensor] = None,
        early_stopping_patience: Optional[int] = None
    ):
        # Load config for defaults
        config = get_config()

        # Use config values if not provided
        if output_dir is None:
            output_dir = str(config.checkpoints_dir)
        if learning_rate is None:
            learning_rate = config.learning_rate
        if batch_size is None:
            batch_size = config.batch_size
        if num_epochs is None:
            num_epochs = config.num_epochs
        if early_stopping_patience is None:
            early_stopping_patience = config.get('training.early_stopping_patience', 5)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Use GPU if available
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
                # Print GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🚀 GPU DETECTED: {gpu_name} ({gpu_memory:.1f} GB)")
            else:
                self.device = 'cpu'
                logger.warning("⚠️  No GPU detected, using CPU (training will be slower)")
        else:
            self.device = device

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_classes = num_classes
        self.vuln_types = vuln_types_map

        # Initialize model with dynamic vulnerability heads
        logger.info("🔧 Initializing Static Encoder (GAT)...")
        vuln_type_names = [vuln_types_map[i] for i in range(num_classes)]
        self.model = StaticEncoder(
            node_feature_dim=128,
            hidden_dim=256,
            output_dim=768,
            dropout=0.2,
            vulnerability_types=vuln_type_names
        ).to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"📊 Model parameters: {trainable_params:,} trainable / {total_params:,} total")

        # Loss function with class weighting and label smoothing
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"⚖️  Using class-weighted loss function with label smoothing")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"runs/static_optimized_{timestamp}")
        logger.info(f"📈 TensorBoard: runs/static_optimized_{timestamp}")

        # Early stopping monitor
        self.monitor = TrainingMonitor(patience=early_stopping_patience)

    def print_batch_stats(self, batch_idx: int, total_batches: int, loss: float,
                         correct: int, total: int, epoch_start_time: float):
        """Print detailed batch statistics"""
        if batch_idx % 10 == 0:  # Print every 10 batches
            accuracy = 100 * correct / total if total > 0 else 0
            elapsed = time.time() - epoch_start_time
            batches_per_sec = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta_seconds = (total_batches - batch_idx - 1) / batches_per_sec if batches_per_sec > 0 else 0

            logger.info(
                f"  Batch [{batch_idx+1:>4}/{total_batches}] | "
                f"Loss: {loss:.4f} | "
                f"Acc: {accuracy:>5.2f}% | "
                f"Speed: {batches_per_sec:.2f} batch/s | "
                f"ETA: {int(eta_seconds//60)}m {int(eta_seconds%60)}s"
            )

    def print_epoch_summary(self, epoch: int, train_loss: float, train_acc: float,
                           val_loss: float, val_acc: float, val_f1: float,
                           epoch_time: float, is_best: bool):
        """Print detailed epoch summary"""
        logger.info("\n" + "="*100)
        logger.info(f"EPOCH {epoch+1}/{self.num_epochs} SUMMARY")
        logger.info("="*100)
        logger.info(f"⏱️  Time: {int(epoch_time//60)}m {int(epoch_time%60)}s")
        logger.info(f"📉 Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        logger.info(f"📊 Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}% | Val F1: {val_f1:.4f}")

        if is_best:
            logger.info("✅ NEW BEST MODEL SAVED!")

        logger.info("="*100 + "\n")

    def print_detailed_metrics(self, all_preds: List[int], all_labels: List[int], phase_name: str):
        """Print comprehensive per-vulnerability metrics"""

        logger.info("\n" + "="*100)
        logger.info(f"{phase_name} - DETAILED METRICS")
        logger.info("="*100)

        # Overall metrics
        accuracy = accuracy_score(all_labels, all_preds)
        logger.info(f"\n🎯 OVERALL ACCURACY: {accuracy*100:.2f}%\n")

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(self.num_classes)), zero_division=0
        )

        logger.info(f"{'Vulnerability Type':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Detected':>10}")
        logger.info("-" * 100)

        total_detected = 0
        total_samples = 0

        for i in range(self.num_classes):
            if support[i] > 0:
                vuln_name = self.vuln_types[i]
                detected = sum(1 for pred, label in zip(all_preds, all_labels) if label == i and pred == i)
                total = support[i]
                detection_rate = (detected / total) * 100 if total > 0 else 0

                # Color coding based on performance
                if recall[i] >= 0.7:
                    status = "✅"
                elif recall[i] >= 0.5:
                    status = "⚠️ "
                else:
                    status = "❌"

                logger.info(
                    f"{status} {vuln_name:<27} "
                    f"{precision[i]:>10.4f} {recall[i]:>10.4f} "
                    f"{f1[i]:>10.4f} {int(support[i]):>10} "
                    f"{detected:>4}/{total:<4} ({detection_rate:.1f}%)"
                )

                total_detected += detected
                total_samples += total

        # Macro and weighted averages
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        logger.info("-" * 100)
        logger.info(f"{'📊 MACRO F1':<30} {macro_f1:>10.4f}")
        logger.info(f"{'📊 WEIGHTED F1':<30} {weighted_f1:>10.4f}")
        logger.info(f"{'📊 TOTAL DETECTED':<30} {total_detected:>4}/{total_samples:<4} ({total_detected/total_samples*100:.2f}%)")
        logger.info("="*100 + "\n")

        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Train static encoder with detailed monitoring"""
        logger.info("\n" + "="*100)
        logger.info("🚀 STARTING TRAINING")
        logger.info("="*100)
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Max epochs: {self.num_epochs}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info("="*100 + "\n")

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )

        best_val_f1 = 0.0
        training_start = time.time()

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            logger.info(f"\n{'='*100}")
            logger.info(f"🔄 EPOCH {epoch+1}/{self.num_epochs} - TRAINING")
            logger.info(f"{'='*100}\n")

            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_labels = []

            for batch_idx, batch in enumerate(train_loader):
                pdgs = batch['pdg']
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                try:
                    static_features, vuln_scores = self.model(pdgs, None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(all_scores, 1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)

                    train_preds.extend(predicted.cpu().numpy().tolist())
                    train_labels.extend(labels.cpu().numpy().tolist())

                    # Print batch stats
                    self.print_batch_stats(
                        batch_idx, len(train_loader), loss.item(),
                        train_correct, train_total, epoch_start
                    )

                except Exception as e:
                    logger.error(f"❌ Training error at batch {batch_idx}: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
            train_acc = train_correct / train_total if train_total > 0 else 0

            # Validation phase
            logger.info(f"\n{'='*50}")
            logger.info("📊 VALIDATING...")
            logger.info(f"{'='*50}\n")

            val_loss, val_preds, val_labels = self.validate(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted', zero_division=0)

            epoch_time = time.time() - epoch_start

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Check if best model
            is_best = val_f1 > best_val_f1
            if is_best:
                best_val_f1 = val_f1
                self.save_checkpoint(epoch, val_loss, val_acc, val_f1, optimizer)

            # Print epoch summary
            self.print_epoch_summary(
                epoch, avg_train_loss, train_acc,
                val_loss, val_acc, val_f1, epoch_time, is_best
            )

            # TensorBoard logging
            self.writer.add_scalar('Train/Loss', avg_train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Val/F1', val_f1, epoch)
            self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            # Print detailed metrics every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == self.num_epochs - 1:
                self.print_detailed_metrics(val_preds, val_labels, f"EPOCH {epoch+1} VALIDATION")

            # Early stopping check
            if self.monitor.update(val_loss, val_f1):
                logger.info("\n🛑 Training stopped early - model not improving")
                break

        total_time = time.time() - training_start
        logger.info(f"\n✅ Training complete! Total time: {int(total_time//60)}m {int(total_time%60)}s")
        logger.info(f"🏆 Best validation F1: {best_val_f1:.4f}")

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
        logger.info("🧪 TESTING MODEL")
        logger.info("="*100)

        test_loss, test_preds, test_labels = self.validate(test_loader)

        # Print comprehensive metrics
        metrics = self.print_detailed_metrics(test_preds, test_labels, "FINAL TEST SET")

        # Save results to file
        results_file = self.output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(results_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write("STATIC VULNERABILITY DETECTION - TEST RESULTS\n")
            f.write("="*100 + "\n\n")
            f.write(f"Overall Accuracy: {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Macro F1: {metrics['macro_f1']:.4f}\n")
            f.write(f"Weighted F1: {metrics['weighted_f1']:.4f}\n\n")

            f.write(f"{'Vulnerability Type':<30} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}\n")
            f.write("-"*100 + "\n")

            for i in range(self.num_classes):
                if metrics['support'][i] > 0:
                    vuln_name = self.vuln_types[i]
                    f.write(
                        f"{vuln_name:<30} "
                        f"{metrics['precision'][i]:>10.4f} {metrics['recall'][i]:>10.4f} "
                        f"{metrics['f1'][i]:>10.4f} {int(metrics['support'][i]):>10}\n"
                    )

        logger.info(f"\n💾 Results saved to: {results_file}")
        return metrics

    def save_checkpoint(self, epoch, val_loss, val_acc, val_f1, optimizer):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1
        }
        path = self.output_dir / "static_encoder_best.pt"
        torch.save(checkpoint, path)


def calculate_class_weights(dataset: StaticDataset) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset"""
    num_classes = dataset.num_classes
    class_counts = torch.zeros(num_classes)
    for label in dataset.labels:
        class_counts[label] += 1

    # Inverse frequency weighting
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes

    logger.info("\n" + "="*80)
    logger.info("CLASS WEIGHTS (for handling imbalanced data)")
    logger.info("="*80)
    vuln_types_inv = {v: k for k, v in dataset.vuln_types.items()}
    for i in range(num_classes):
        if class_counts[i] > 0:
            vuln_name = vuln_types_inv.get(i, f"class_{i}")
            logger.info(f"  {vuln_name:<30} Count: {int(class_counts[i]):>5} | Weight: {class_weights[i]:.4f}")
    logger.info("="*80 + "\n")

    return class_weights


def main():
    # Load configuration
    config = get_config()

    # Get static-specific training config
    static_config = config.get_training_config('static')

    parser = argparse.ArgumentParser(description="Static Vulnerability Detection Training")
    parser.add_argument("--train-dir", default=None, help="Training data directory (default: from config)")
    parser.add_argument("--val-dir", default=None, help="Validation data directory (default: from config)")
    parser.add_argument("--test-dir", default=None, help="Test data directory (default: from config)")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: from config)")
    parser.add_argument("--batch-size", type=int, default=None, help=f"Batch size (default: {static_config.get('batch_size', 16)})")
    parser.add_argument("--num-epochs", type=int, default=None, help=f"Maximum number of epochs (default: {static_config.get('num_epochs', 50)})")
    parser.add_argument("--learning-rate", type=float, default=None, help=f"Learning rate (default: {static_config.get('learning_rate', 0.001)})")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for debugging)")
    parser.add_argument("--no-cache", action='store_true', help="Disable caching")
    parser.add_argument("--num-workers", type=int, default=None, help=f"Data loader workers (default: {static_config.get('num_workers', 4)})")
    parser.add_argument("--early-stopping", type=int, default=None, help=f"Early stopping patience (default: {static_config.get('early_stopping_patience', 5)})")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (default: config.yaml)")

    args = parser.parse_args()

    # Load custom config if provided
    if args.config:
        from utils.config import load_config
        config = load_config(args.config)
        static_config = config.get_training_config('static')

    # Use config values as defaults, allow command-line overrides
    train_dir = args.train_dir or str(config.train_dir)
    val_dir = args.val_dir or str(config.val_dir)
    test_dir = args.test_dir or str(config.test_dir)
    output_dir = args.output_dir or str(config.checkpoints_dir)
    batch_size = args.batch_size or static_config.get('batch_size', 16)
    num_epochs = args.num_epochs or static_config.get('num_epochs', 50)
    learning_rate = args.learning_rate or static_config.get('learning_rate', 0.001)
    num_workers = args.num_workers or static_config.get('num_workers', 4)
    early_stopping = args.early_stopping or static_config.get('early_stopping_patience', 5)

    logger.info("\n" + "="*100)
    logger.info("🚀 OPTIMIZED STATIC VULNERABILITY DETECTION")
    logger.info("="*100)
    logger.info(f"📋 Configuration: {'Custom' if args.config else 'Default (config.yaml)'}")
    logger.info(f"Train: {train_dir}")
    logger.info(f"Val:   {val_dir}")
    logger.info(f"Test:  {test_dir}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Max epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Early stopping patience: {early_stopping}")
    logger.info(f"Data loader workers: {num_workers}")
    logger.info("="*100 + "\n")

    # Load datasets
    logger.info("📁 Loading datasets...\n")
    train_dataset = StaticDataset(train_dir, max_samples=args.max_samples, use_cache=not args.no_cache)
    val_dataset = StaticDataset(val_dir, use_cache=not args.no_cache)
    test_dataset = StaticDataset(test_dir, use_cache=not args.no_cache)

    logger.info(f"✓ Train: {len(train_dataset)} contracts")
    logger.info(f"✓ Val:   {len(val_dataset)} contracts")
    logger.info(f"✓ Test:  {len(test_dataset)} contracts\n")

    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset)

    # Create data loaders with multi-worker support
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Create vulnerability types mapping (label -> name)
    vuln_types_map = {v: k for k, v in train_dataset.vuln_types.items()}

    # Initialize detector
    detector = StaticVulnerabilityDetector(
        num_classes=train_dataset.num_classes,
        vuln_types_map=vuln_types_map,
        output_dir=output_dir,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_epochs=num_epochs,
        class_weights=class_weights,
        early_stopping_patience=early_stopping
    )

    # Train
    detector.train(train_loader, val_loader)

    # Test
    detector.test(test_loader)

    logger.info("\n" + "="*100)
    logger.info("✅ TRAINING AND TESTING COMPLETE!")
    logger.info("="*100)
    logger.info(f"💾 Model: {output_dir}/static_encoder_best.pt")
    logger.info(f"📊 Results: {output_dir}/test_results_*.txt")
    logger.info(f"📈 TensorBoard: tensorboard --logdir runs/")
    logger.info("="*100 + "\n")


if __name__ == "__main__":
    main()
