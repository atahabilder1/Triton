#!/usr/bin/env python3
"""
Triton Improved Training Script
Trains with class-balanced sampling and weighted loss on FORGE dataset
Tests on SmartBugs as held-out test set
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FORGEDataset(Dataset):
    """Dataset for FORGE smart contracts with vulnerability labels"""

    def __init__(self, forge_dir: str, max_samples: Optional[int] = None):
        self.contracts = []
        self.labels = []

        # Vulnerability type mapping (same as SmartBugs)
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
            'other': 9
        }

        # CWE to vulnerability type mapping (based on FORGE paper)
        self.cwe_mapping = {
            'CWE-284': 'access_control',  # Access Control
            'CWE-269': 'access_control',  # Improper Privilege Management
            'CWE-287': 'access_control',  # Improper Authentication
            'CWE-190': 'arithmetic',  # Integer Overflow
            'CWE-191': 'arithmetic',  # Integer Underflow
            'CWE-682': 'arithmetic',  # Incorrect Calculation
            'CWE-330': 'bad_randomness',  # Weak PRNG
            'CWE-338': 'bad_randomness',  # Weak PRNG
            'CWE-400': 'denial_of_service',  # Uncontrolled Resource Consumption
            'CWE-770': 'denial_of_service',  # Allocation without Limits
            'CWE-362': 'front_running',  # Race Condition
            'CWE-841': 'front_running',  # Improper Enforcement of Behavioral Workflow
            'CWE-561': 'reentrancy',  # Reentrancy
            'CWE-663': 'reentrancy',  # Race condition
            'CWE-628': 'short_addresses',  # Function Call with Incorrectly Specified Arguments
            'CWE-352': 'time_manipulation',  # CSRF / Timestamp Dependency
            'CWE-703': 'unchecked_low_level_calls',  # Unchecked Return Value
            'CWE-252': 'unchecked_low_level_calls',  # Unchecked Return Value
        }

        forge_path = Path(forge_dir)
        results_dir = forge_path / 'results'
        contracts_dir = forge_path / 'contracts'

        logger.info(f"Loading FORGE dataset from {forge_dir}...")

        # Load all JSON result files
        json_files = list(results_dir.glob('*.json'))
        logger.info(f"Found {len(json_files)} FORGE result files")

        for json_file in tqdm(json_files[:max_samples] if max_samples else json_files, desc="Loading FORGE"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Get project path to find contracts
                project_info = data.get('project_info', {})
                project_path = project_info.get('project_path', {})

                if not project_path:
                    continue

                # Map findings to vulnerability types
                findings = data.get('findings', [])
                if not findings:
                    continue

                # Determine primary vulnerability type from CWE categories
                vuln_type = self._get_vulnerability_type(findings)

                # Find and load contract files
                for project_name, contract_rel_path in project_path.items():
                    # The path format is: "contracts/project/subdir/file.sol"
                    # We need to construct: contracts_dir / project / subdir / file.sol

                    # Try multiple path formats
                    possible_paths = [
                        contracts_dir / contract_rel_path,  # Direct
                        contracts_dir / contract_rel_path.replace('contracts/', ''),  # Remove prefix
                        contracts_dir / Path(contract_rel_path).name,  # Just filename
                    ]

                    # Also try finding .sol files in the project directory
                    project_parts = contract_rel_path.split('/')
                    if len(project_parts) > 1:
                        project_dir = contracts_dir / project_parts[1] if project_parts[0] == 'contracts' else contracts_dir / project_parts[0]
                        if project_dir.exists():
                            for sol_file in project_dir.rglob('*.sol'):
                                possible_paths.append(sol_file)

                    for contract_path in possible_paths:
                        if contract_path.exists() and contract_path.suffix == '.sol':
                            try:
                                with open(contract_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    source_code = f.read()

                                if len(source_code) > 100:  # Skip very small files
                                    self.contracts.append({
                                        'source_code': source_code,
                                        'path': str(contract_path),
                                        'vulnerability_type': vuln_type
                                    })
                                    self.labels.append(self.vuln_types[vuln_type])
                                    break  # Found one, move to next project

                            except Exception as e:
                                continue

            except Exception as e:
                logger.warning(f"Error loading {json_file}: {e}")
                continue

        logger.info(f"Loaded {len(self.contracts)} contracts from FORGE")

        # Print distribution
        label_counts = Counter(self.labels)
        logger.info("Label distribution:")
        for vuln_type, label in sorted(self.vuln_types.items(), key=lambda x: x[1]):
            count = label_counts[label]
            if count > 0:
                logger.info(f"  {vuln_type}: {count} contracts ({count/len(self.labels)*100:.1f}%)")

    def _get_vulnerability_type(self, findings: List[Dict]) -> str:
        """Extract primary vulnerability type from findings"""
        cwe_counts = defaultdict(int)

        for finding in findings:
            category = finding.get('category', {})
            # Get all CWEs from the finding
            for level, cwes in category.items():
                for cwe in cwes:
                    if cwe in self.cwe_mapping:
                        vuln_type = self.cwe_mapping[cwe]
                        cwe_counts[vuln_type] += 1

        if not cwe_counts:
            return 'other'

        # Return most common vulnerability type
        return max(cwe_counts.items(), key=lambda x: x[1])[0]

    def __len__(self):
        return len(self.contracts)

    def __getitem__(self, idx):
        contract = self.contracts[idx]
        label = self.labels[idx]

        return {
            'source_code': contract['source_code'],
            'path': contract['path'],
            'vulnerability_type': contract['vulnerability_type'],
            'label': label
        }


class ImprovedTritonTrainer:
    """Improved trainer with class balancing and weighted loss"""

    def __init__(
        self,
        output_dir: str = "models/checkpoints_improved",
        device: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        num_epochs: int = 20,
        use_class_weights: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.use_class_weights = use_class_weights

        # Initialize models
        logger.info("Initializing models...")

        self.static_encoder = StaticEncoder(
            node_feature_dim=128,
            hidden_dim=256,
            output_dim=768,
            dropout=0.2
        ).to(self.device)

        self.dynamic_encoder = DynamicEncoder(
            vocab_size=50,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout=0.2
        ).to(self.device)

        self.semantic_encoder = SemanticEncoder(
            model_name="microsoft/graphcodebert-base",
            output_dim=768,
            max_length=512,
            dropout=0.1
        ).to(self.device)

        self.fusion_module = CrossModalFusion(
            static_dim=768,
            dynamic_dim=512,
            semantic_dim=768,
            hidden_dim=512,
            output_dim=768,
            dropout=0.1
        ).to(self.device)

        self.criterion = None  # Will be set with class weights
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

        # Checkpoint resuming state
        self.start_epoch = 0
        self.best_val_loss = float('inf')

    def compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Compute class weights for imbalanced dataset"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = 10  # Always 10 vulnerability types

        # Compute inverse frequency weights for all 10 classes
        weights = []
        for i in range(num_classes):
            count = label_counts.get(i, 1)  # Default to 1 if class not present
            if count == 0:
                count = 1  # Avoid division by zero
            weight = total_samples / (num_classes * count)
            weights.append(weight)

        weights_tensor = torch.FloatTensor(weights).to(self.device)
        logger.info(f"Class weights (all 10 classes): {weights_tensor.cpu().numpy()}")

        return weights_tensor

    def create_balanced_sampler(self, labels: List[int]) -> WeightedRandomSampler:
        """Create weighted sampler for balanced batch sampling"""
        label_counts = Counter(labels)
        weights = [1.0 / label_counts[label] for label in labels]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True
        )
        return sampler

    def train_semantic_encoder(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        resume: bool = False
    ):
        """Fine-tune GraphCodeBERT with class-weighted loss"""

        logger.info("=" * 80)
        logger.info("PHASE 1: Fine-tuning Semantic Encoder with Class Weights")
        logger.info("=" * 80)

        num_epochs = num_epochs or self.num_epochs

        # Optimizer
        optimizer = optim.AdamW(
            self.semantic_encoder.parameters(),
            lr=self.learning_rate * 0.1,
            weight_decay=0.01
        )

        best_val_loss = self.best_val_loss
        start_epoch = self.start_epoch if resume else 0

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            self.semantic_encoder.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Training Semantic Encoder")
            for batch in pbar:
                source_codes = batch['source_code']
                labels = batch['label'].to(self.device)
                vuln_types = batch['vulnerability_type']

                optimizer.zero_grad()

                # Forward pass
                semantic_features, vuln_scores = self.semantic_encoder(
                    source_codes,
                    vuln_types
                )

                # Aggregate scores
                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                # Compute weighted loss
                loss = self.criterion(all_scores, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(all_scores, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total

            # Validation
            val_loss, val_acc = self._validate_semantic(val_loader)

            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = best_val_loss
                self._save_checkpoint(
                    self.semantic_encoder,
                    f"semantic_encoder_best.pt",
                    {'epoch': epoch+1, 'val_loss': val_loss, 'val_acc': val_acc},
                    optimizer
                )
                logger.info(f"✓ Saved best semantic encoder (val_loss: {val_loss:.4f})")

            # Save training state every epoch for resuming
            self.save_training_state(epoch + 1, 'semantic')

        logger.info(f"\nSemantic Encoder training complete! Best val_loss: {best_val_loss:.4f}")

    def train_fusion_module(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None,
        resume: bool = False
    ):
        """Train fusion module with class-weighted loss"""

        logger.info("=" * 80)
        logger.info("PHASE 2: Training Fusion Module with Class Weights")
        logger.info("=" * 80)

        num_epochs = num_epochs or self.num_epochs

        # Optimizer
        optimizer = optim.Adam([
            {'params': self.static_encoder.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.dynamic_encoder.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.semantic_encoder.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.fusion_module.parameters(), 'lr': self.learning_rate}
        ])

        best_val_loss = self.best_val_loss
        start_epoch = self.start_epoch if resume else 0

        for epoch in range(start_epoch, num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            self.static_encoder.train()
            self.dynamic_encoder.train()
            self.semantic_encoder.train()
            self.fusion_module.train()

            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Training Fusion Module")
            for batch in pbar:
                source_codes = batch['source_code']
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                # Generate dummy features
                batch_size = len(source_codes)
                static_features = torch.randn(batch_size, 768).to(self.device)
                dynamic_features = torch.randn(batch_size, 512).to(self.device)

                # Get semantic features
                semantic_features, _ = self.semantic_encoder(source_codes, None)

                # Fusion
                fusion_output = self.fusion_module(
                    static_features,
                    dynamic_features,
                    semantic_features,
                    None
                )

                vulnerability_logits = fusion_output['vulnerability_logits']

                # Compute weighted loss
                loss = self.criterion(vulnerability_logits, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predicted = torch.max(vulnerability_logits, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

                pbar.set_postfix({'loss': loss.item()})

            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total

            # Validation
            val_loss, val_acc = self._validate_fusion(val_loader)

            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.best_val_loss = best_val_loss
                self._save_all_models(epoch+1, val_loss, val_acc)
                logger.info(f"✓ Saved best fusion model (val_loss: {val_loss:.4f})")

            # Save training state every epoch for resuming
            self.save_training_state(epoch + 1, 'fusion')

        logger.info(f"\nFusion Module training complete! Best val_loss: {best_val_loss:.4f}")

    def _validate_semantic(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate semantic encoder"""
        self.semantic_encoder.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                source_codes = batch['source_code']
                labels = batch['label'].to(self.device)
                vuln_types = batch['vulnerability_type']

                semantic_features, vuln_scores = self.semantic_encoder(
                    source_codes,
                    vuln_types
                )

                all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)
                loss = self.criterion(all_scores, labels)

                val_loss += loss.item()
                _, predicted = torch.max(all_scores, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        return avg_val_loss, val_acc

    def _validate_fusion(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate fusion module"""
        self.static_encoder.eval()
        self.dynamic_encoder.eval()
        self.semantic_encoder.eval()
        self.fusion_module.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                source_codes = batch['source_code']
                labels = batch['label'].to(self.device)

                batch_size = len(source_codes)
                static_features = torch.randn(batch_size, 768).to(self.device)
                dynamic_features = torch.randn(batch_size, 512).to(self.device)

                semantic_features, _ = self.semantic_encoder(source_codes, None)

                fusion_output = self.fusion_module(
                    static_features,
                    dynamic_features,
                    semantic_features,
                    None
                )

                vulnerability_logits = fusion_output['vulnerability_logits']
                loss = self.criterion(vulnerability_logits, labels)

                val_loss += loss.item()
                _, predicted = torch.max(vulnerability_logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total

        return avg_val_loss, val_acc

    def _save_checkpoint(self, model, filename, metadata=None, optimizer=None):
        """Save model checkpoint with training state"""
        checkpoint_path = self.output_dir / filename

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {},
            'training_history': self.training_history
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def load_checkpoint(self, checkpoint_name: str = "training_state.pt") -> bool:
        """Load checkpoint to resume training"""
        checkpoint_path = self.output_dir / checkpoint_name

        if not checkpoint_path.exists():
            logger.info(f"No checkpoint found at {checkpoint_path}, starting fresh")
            return False

        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load model states
            if 'semantic_encoder_state' in checkpoint:
                self.semantic_encoder.load_state_dict(checkpoint['semantic_encoder_state'])
            if 'static_encoder_state' in checkpoint:
                self.static_encoder.load_state_dict(checkpoint['static_encoder_state'])
            if 'dynamic_encoder_state' in checkpoint:
                self.dynamic_encoder.load_state_dict(checkpoint['dynamic_encoder_state'])
            if 'fusion_module_state' in checkpoint:
                self.fusion_module.load_state_dict(checkpoint['fusion_module_state'])

            # Load training state
            self.start_epoch = checkpoint.get('epoch', 0)
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            self.training_history = checkpoint.get('training_history', self.training_history)

            logger.info(f"✓ Resumed from epoch {self.start_epoch}")
            logger.info(f"✓ Best val loss so far: {self.best_val_loss:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

    def save_training_state(self, epoch: int, phase: str):
        """Save complete training state for resuming"""
        checkpoint_path = self.output_dir / "training_state.pt"

        checkpoint = {
            'epoch': epoch,
            'phase': phase,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'semantic_encoder_state': self.semantic_encoder.state_dict(),
            'static_encoder_state': self.static_encoder.state_dict(),
            'dynamic_encoder_state': self.dynamic_encoder.state_dict(),
            'fusion_module_state': self.fusion_module.state_dict(),
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"✓ Saved training state at epoch {epoch}, phase {phase}")

    def _save_all_models(self, epoch, val_loss, val_acc):
        """Save all models"""
        self._save_checkpoint(
            self.static_encoder,
            f"static_encoder_best.pt",
            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        )

        self._save_checkpoint(
            self.dynamic_encoder,
            f"dynamic_encoder_best.pt",
            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        )

        self._save_checkpoint(
            self.semantic_encoder,
            f"semantic_encoder_fusion_best.pt",
            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        )

        self._save_checkpoint(
            self.fusion_module,
            f"fusion_module_best.pt",
            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        )

        logger.info(f"✓ Saved all models for epoch {epoch}")


def main():
    parser = argparse.ArgumentParser(description="Train Triton with improved strategy")
    parser.add_argument(
        "--forge-dir",
        default="data/datasets/FORGE-Artifacts/dataset",
        help="FORGE dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints_improved",
        help="Output directory for improved model checkpoints"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        help="Use class-weighted loss"
    )
    parser.add_argument(
        "--use-balanced-sampling",
        action="store_true",
        default=True,
        help="Use balanced sampling"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last checkpoint"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRITON IMPROVED TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"FORGE directory: {args.forge_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Class weights: {args.use_class_weights}")
    logger.info(f"Balanced sampling: {args.use_balanced_sampling}")

    # Load FORGE dataset
    logger.info("\nLoading FORGE dataset...")
    dataset = FORGEDataset(args.forge_dir, max_samples=args.max_samples)

    if len(dataset) == 0:
        logger.error("No data loaded! Check FORGE directory.")
        sys.exit(1)

    # Split dataset (80/20 train/val)
    from torch.utils.data import random_split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Initialize trainer
    trainer = ImprovedTritonTrainer(
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        use_class_weights=args.use_class_weights
    )

    # Try to resume from checkpoint
    if args.resume:
        trainer.load_checkpoint()

    # Compute class weights
    train_labels = [dataset[train_dataset.indices[i]]['label'] for i in range(len(train_dataset))]
    class_weights = trainer.compute_class_weights(train_labels)
    trainer.criterion = nn.CrossEntropyLoss(weight=class_weights if args.use_class_weights else None)

    # Create data loaders
    if args.use_balanced_sampling:
        sampler = trainer.create_balanced_sampler(train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=0
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Train components
    logger.info("\nStarting improved training...")

    # Phase 1: Fine-tune semantic encoder
    trainer.train_semantic_encoder(train_loader, val_loader, resume=args.resume)

    # Phase 2: Train fusion module
    trainer.train_fusion_module(train_loader, val_loader, resume=args.resume)

    logger.info("\n" + "=" * 80)
    logger.info("IMPROVED TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nImproved model checkpoints saved to: {args.output_dir}")
    logger.info("\nNext steps:")
    logger.info("1. Test on SmartBugs: python scripts/test_triton.py --checkpoint-dir models/checkpoints_improved")
    logger.info("2. Compare with baseline results")
    logger.info("3. Analyze per-class performance improvements")


if __name__ == "__main__":
    main()
