#!/usr/bin/env python3
"""
Complete Triton Training Pipeline
Tests and trains Static, Dynamic, and Semantic encoders individually,
then trains the fusion module end-to-end.
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
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion
from tools.slither_wrapper import SlitherWrapper, extract_static_features
from tools.mythril_wrapper import MythrilWrapper, extract_dynamic_features
from utils.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModalDataset(Dataset):
    """Dataset that extracts static, dynamic, and semantic features"""

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

        # Load from organized dataset (SmartBugs structure)
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
        else:
            # Load from flat directory
            sol_files = list(contracts_path.rglob("*.sol"))
            if max_samples:
                sol_files = sol_files[:max_samples]

            logger.info(f"Found {len(sol_files)} contracts (flat structure)")

            for contract_file in sol_files:
                try:
                    with open(contract_file, 'r', encoding='utf-8', errors='ignore') as f:
                        source_code = f.read()

                    # Try to infer vulnerability type from path or filename
                    vuln_type = 'other'
                    for vtype in self.vuln_types.keys():
                        if vtype in str(contract_file).lower():
                            vuln_type = vtype
                            break

                    self.contracts.append({
                        'source_code': source_code,
                        'path': str(contract_file),
                        'vulnerability_type': vuln_type
                    })
                    self.labels.append(self.vuln_types[vuln_type])

                except Exception as e:
                    logger.warning(f"Error loading {contract_file}: {e}")
                    continue

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

        # Initialize analysis tools
        self.slither = SlitherWrapper(timeout=60)
        self.mythril = MythrilWrapper(timeout=60, max_depth=12)

    def __len__(self):
        return len(self.contracts)

    def _get_cache_path(self, idx: int, feature_type: str) -> Path:
        """Get cache file path for a specific feature"""
        contract_hash = hash(self.contracts[idx]['path'])
        return self.cache_dir / f"{contract_hash}_{feature_type}.json"

    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load features from cache"""
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save features to cache"""
        try:
            # Convert NetworkX graph to serializable format
            if 'pdg' in data and isinstance(data['pdg'], nx.DiGraph):
                data['pdg'] = {
                    'nodes': list(data['pdg'].nodes(data=True)),
                    'edges': list(data['pdg'].edges(data=True))
                }
            with open(cache_path, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache data: {e}")

    def _extract_pdg(self, source_code: str, idx: int) -> nx.DiGraph:
        """Extract PDG using Slither with caching"""
        cache_path = self._get_cache_path(idx, 'pdg')

        if self.use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                # Reconstruct NetworkX graph from cached data
                pdg = nx.DiGraph()
                # Handle both old and new cache formats
                if 'nodes' in cached and 'edges' in cached:
                    for node, attrs in cached['nodes']:
                        pdg.add_node(node, **attrs)
                    for src, tgt, attrs in cached['edges']:
                        pdg.add_edge(src, tgt, **attrs)
                elif 'pdg' in cached and isinstance(cached['pdg'], dict):
                    # Old format: pdg stored as dict with nodes/edges
                    if 'nodes' in cached['pdg'] and 'edges' in cached['pdg']:
                        for node, attrs in cached['pdg']['nodes']:
                            pdg.add_node(node, **attrs)
                        for src, tgt, attrs in cached['pdg']['edges']:
                            pdg.add_edge(src, tgt, **attrs)
                return pdg

        # Extract using Slither
        result = self.slither.analyze_contract(source_code)

        if result.get('success') and result.get('pdg'):
            pdg = result['pdg']
            if self.use_cache:
                self._save_to_cache(cache_path, result)
            return pdg

        # Return empty graph if extraction failed
        return nx.DiGraph()

    def _extract_execution_traces(self, source_code: str, idx: int) -> List[Dict]:
        """Extract execution traces using Mythril with caching"""
        cache_path = self._get_cache_path(idx, 'traces')

        if self.use_cache:
            cached = self._load_from_cache(cache_path)
            if cached:
                return cached.get('execution_traces', [])

        # Extract using Mythril
        result = self.mythril.analyze_contract(source_code)

        if result.get('success'):
            traces = result.get('execution_traces', [])
            if self.use_cache:
                self._save_to_cache(cache_path, result)
            return traces

        # Return empty trace if extraction failed
        return [{'steps': []}]

    def __getitem__(self, idx):
        contract = self.contracts[idx]
        label = self.labels[idx]

        # Extract PDG for static analysis
        pdg = self._extract_pdg(contract['source_code'], idx)

        # Extract execution traces for dynamic analysis
        traces = self._extract_execution_traces(contract['source_code'], idx)

        return {
            'source_code': contract['source_code'],
            'pdg': pdg,
            'execution_traces': traces,
            'path': contract['path'],
            'vulnerability_type': contract['vulnerability_type'],
            'label': label
        }


def collate_fn(batch):
    """Custom collate function to handle variable-sized PDGs and traces"""
    return {
        'source_code': [item['source_code'] for item in batch],
        'pdg': [item['pdg'] for item in batch],
        'execution_traces': [item['execution_traces'] for item in batch],
        'path': [item['path'] for item in batch],
        'vulnerability_type': [item['vulnerability_type'] for item in batch],
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long)
    }


class CompleteTritonTrainer:
    """Complete trainer for all Triton components"""

    def __init__(
        self,
        output_dir: str = "models/checkpoints",
        device: str = None,
        learning_rate: float = 0.001,
        batch_size: int = 4,
        num_epochs: int = 10,
        class_weights: Optional[torch.Tensor] = None,
        use_tensorboard: bool = True
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs

        # Vulnerability type mapping for reporting
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

        # TensorBoard writer
        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(f"runs/triton_{timestamp}")
            logger.info(f"TensorBoard logging enabled: runs/triton_{timestamp}")
        else:
            self.writer = None

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

        # Loss and metrics with class weighting
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            logger.info(f"Using class-weighted loss function")
            logger.info(f"Class weights: {class_weights.cpu().numpy()}")
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.training_history = defaultdict(list)

    def _compute_per_class_metrics(self, all_preds: List[int], all_labels: List[int], phase_name: str):
        """Compute and log per-class precision, recall, F1"""
        # Get classification report
        target_names = [self.vuln_types[i] for i in range(11)]

        # Compute metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, labels=list(range(11)), zero_division=0
        )

        logger.info(f"\n{phase_name} - Per-Class Metrics:")
        logger.info("-" * 80)
        logger.info(f"{'Class':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
        logger.info("-" * 80)

        for i in range(11):
            if support[i] > 0:  # Only show classes with samples
                logger.info(
                    f"{target_names[i]:<25} "
                    f"{precision[i]:>10.4f} {recall[i]:>10.4f} "
                    f"{f1[i]:>10.4f} {support[i]:>10.0f}"
                )

        # Compute macro and weighted averages
        macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

        logger.info("-" * 80)
        logger.info(f"{'Macro Average F1':<25} {macro_f1:>10.4f}")
        logger.info(f"{'Weighted Average F1':<25} {weighted_f1:>10.4f}")
        logger.info("-" * 80)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1
        }

    def test_static_encoder(self, test_loader: DataLoader):
        """Test static encoder individually"""
        logger.info("=" * 80)
        logger.info("TESTING STATIC ENCODER")
        logger.info("=" * 80)

        self.static_encoder.eval()

        total_samples = 0
        successful = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing Static Encoder"):
                pdgs = batch['pdg']

                try:
                    static_features, vuln_scores = self.static_encoder(pdgs)
                    successful += len(pdgs)
                except Exception as e:
                    logger.error(f"Static encoder error: {e}")

                total_samples += len(pdgs)

        logger.info(f"Static Encoder Test: {successful}/{total_samples} successful")
        logger.info(f"Success rate: {100*successful/total_samples:.2f}%")

    def test_dynamic_encoder(self, test_loader: DataLoader):
        """Test dynamic encoder individually"""
        logger.info("=" * 80)
        logger.info("TESTING DYNAMIC ENCODER")
        logger.info("=" * 80)

        self.dynamic_encoder.eval()

        total_samples = 0
        successful = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing Dynamic Encoder"):
                traces = batch['execution_traces']

                try:
                    # Flatten traces
                    all_traces = []
                    for trace_list in traces:
                        if trace_list:
                            all_traces.append(trace_list[0] if trace_list else {'steps': []})
                        else:
                            all_traces.append({'steps': []})

                    dynamic_features, vuln_scores = self.dynamic_encoder(all_traces)
                    successful += len(all_traces)
                except Exception as e:
                    logger.error(f"Dynamic encoder error: {e}")

                total_samples += len(traces)

        logger.info(f"Dynamic Encoder Test: {successful}/{total_samples} successful")
        logger.info(f"Success rate: {100*successful/total_samples:.2f}%")

    def test_semantic_encoder(self, test_loader: DataLoader):
        """Test semantic encoder individually"""
        logger.info("=" * 80)
        logger.info("TESTING SEMANTIC ENCODER")
        logger.info("=" * 80)

        self.semantic_encoder.eval()

        total_samples = 0
        successful = 0

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing Semantic Encoder"):
                source_codes = batch['source_code']
                vuln_types = batch['vulnerability_type']

                try:
                    semantic_features, vuln_scores = self.semantic_encoder(
                        source_codes, vuln_types
                    )
                    successful += len(source_codes)
                except Exception as e:
                    logger.error(f"Semantic encoder error: {e}")

                total_samples += len(source_codes)

        logger.info(f"Semantic Encoder Test: {successful}/{total_samples} successful")
        logger.info(f"Success rate: {100*successful/total_samples:.2f}%")

    def train_static_encoder(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ):
        """Train static encoder individually"""
        logger.info("=" * 80)
        logger.info("PHASE 1: Training Static Encoder")
        logger.info("=" * 80)

        num_epochs = num_epochs or self.num_epochs

        optimizer = optim.Adam(
            self.static_encoder.parameters(),
            lr=self.learning_rate
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            self.static_encoder.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Training Static Encoder")
            for batch in pbar:
                pdgs = batch['pdg']
                labels = batch['label'].to(self.device)
                vuln_types = batch['vulnerability_type']

                optimizer.zero_grad()

                try:
                    static_features, vuln_scores = self.static_encoder(pdgs, None)

                    # Aggregate vulnerability scores for classification
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(all_scores, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    pbar.set_postfix({'loss': loss.item()})
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0

            # Validation (compute metrics on last epoch)
            compute_metrics = (epoch == num_epochs - 1)
            val_loss, val_acc = self._validate_static(val_loader, compute_metrics=compute_metrics)

            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Static/Train/Loss', avg_train_loss, epoch)
                self.writer.add_scalar('Static/Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Static/Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Static/Val/Accuracy', val_acc, epoch)

            # Save best model with optimizer state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    self.static_encoder,
                    f"static_encoder_best.pt",
                    {'epoch': epoch+1, 'val_loss': val_loss, 'val_acc': val_acc},
                    optimizer=optimizer
                )
                logger.info(f"✓ Saved best static encoder (val_loss: {val_loss:.4f})")

        logger.info(f"\nStatic Encoder training complete! Best val_loss: {best_val_loss:.4f}")

    def train_dynamic_encoder(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ):
        """Train dynamic encoder individually"""
        logger.info("=" * 80)
        logger.info("PHASE 2: Training Dynamic Encoder")
        logger.info("=" * 80)

        num_epochs = num_epochs or self.num_epochs

        optimizer = optim.Adam(
            self.dynamic_encoder.parameters(),
            lr=self.learning_rate
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch+1}/{num_epochs}")

            # Training
            self.dynamic_encoder.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            pbar = tqdm(train_loader, desc=f"Training Dynamic Encoder")
            for batch in pbar:
                traces = batch['execution_traces']
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                try:
                    # Flatten traces
                    all_traces = []
                    for trace_list in traces:
                        if trace_list:
                            all_traces.append(trace_list[0] if trace_list else {'steps': []})
                        else:
                            all_traces.append({'steps': []})

                    dynamic_features, vuln_scores = self.dynamic_encoder(all_traces, None)

                    # Aggregate vulnerability scores for classification
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(all_scores, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    pbar.set_postfix({'loss': loss.item()})
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0

            # Validation (compute metrics on last epoch)
            compute_metrics = (epoch == num_epochs - 1)
            val_loss, val_acc = self._validate_dynamic(val_loader, compute_metrics=compute_metrics)

            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Dynamic/Train/Loss', avg_train_loss, epoch)
                self.writer.add_scalar('Dynamic/Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Dynamic/Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Dynamic/Val/Accuracy', val_acc, epoch)

            # Save best model with optimizer state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    self.dynamic_encoder,
                    f"dynamic_encoder_best.pt",
                    {'epoch': epoch+1, 'val_loss': val_loss, 'val_acc': val_acc},
                    optimizer=optimizer
                )
                logger.info(f"✓ Saved best dynamic encoder (val_loss: {val_loss:.4f})")

        logger.info(f"\nDynamic Encoder training complete! Best val_loss: {best_val_loss:.4f}")

    def train_semantic_encoder(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ):
        """Train semantic encoder individually"""
        logger.info("=" * 80)
        logger.info("PHASE 3: Training Semantic Encoder")
        logger.info("=" * 80)

        num_epochs = num_epochs or self.num_epochs

        optimizer = optim.AdamW(
            self.semantic_encoder.parameters(),
            lr=self.learning_rate * 0.1,  # Lower LR for fine-tuning
            weight_decay=0.01
        )

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
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

                try:
                    semantic_features, vuln_scores = self.semantic_encoder(
                        source_codes, vuln_types
                    )

                    # Aggregate vulnerability scores
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(all_scores, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()

                    pbar.set_postfix({'loss': loss.item()})
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0

            # Validation (compute metrics on last epoch)
            compute_metrics = (epoch == num_epochs - 1)
            val_loss, val_acc = self._validate_semantic(val_loader, compute_metrics=compute_metrics)

            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Semantic/Train/Loss', avg_train_loss, epoch)
                self.writer.add_scalar('Semantic/Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Semantic/Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Semantic/Val/Accuracy', val_acc, epoch)

            # Save best model with optimizer state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    self.semantic_encoder,
                    f"semantic_encoder_best.pt",
                    {'epoch': epoch+1, 'val_loss': val_loss, 'val_acc': val_acc},
                    optimizer=optimizer
                )
                logger.info(f"✓ Saved best semantic encoder (val_loss: {val_loss:.4f})")

        logger.info(f"\nSemantic Encoder training complete! Best val_loss: {best_val_loss:.4f}")

    def train_fusion_module(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = None
    ):
        """Train fusion module end-to-end"""
        logger.info("=" * 80)
        logger.info("PHASE 4: Training Fusion Module End-to-End")
        logger.info("=" * 80)

        num_epochs = num_epochs or self.num_epochs

        # Optimizer for all components with different learning rates
        optimizer = optim.Adam([
            {'params': self.static_encoder.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.dynamic_encoder.parameters(), 'lr': self.learning_rate * 0.5},
            {'params': self.semantic_encoder.parameters(), 'lr': self.learning_rate * 0.1},
            {'params': self.fusion_module.parameters(), 'lr': self.learning_rate}
        ])

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
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
                pdgs = batch['pdg']
                traces = batch['execution_traces']
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                try:
                    # Get features from all encoders
                    static_features, _ = self.static_encoder(pdgs, None)

                    # Process traces
                    all_traces = []
                    for trace_list in traces:
                        if trace_list:
                            all_traces.append(trace_list[0] if trace_list else {'steps': []})
                        else:
                            all_traces.append({'steps': []})

                    dynamic_features, _ = self.dynamic_encoder(all_traces, None)
                    semantic_features, _ = self.semantic_encoder(source_codes, None)

                    # Fusion
                    fusion_output = self.fusion_module(
                        static_features,
                        dynamic_features,
                        semantic_features,
                        None
                    )

                    vulnerability_logits = fusion_output['vulnerability_logits']

                    # Compute loss
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
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    continue

            avg_train_loss = train_loss / len(train_loader) if train_loss > 0 else 0
            train_acc = 100 * train_correct / train_total if train_total > 0 else 0

            # Validation (compute metrics on last epoch)
            compute_metrics = (epoch == num_epochs - 1)
            val_loss, val_acc = self._validate_fusion(val_loader, compute_metrics=compute_metrics)

            logger.info(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # TensorBoard logging
            if self.writer is not None:
                self.writer.add_scalar('Fusion/Train/Loss', avg_train_loss, epoch)
                self.writer.add_scalar('Fusion/Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Fusion/Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Fusion/Val/Accuracy', val_acc, epoch)

            # Save best model with optimizer state
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_all_models(epoch+1, val_loss, val_acc, optimizer=optimizer)
                logger.info(f"✓ Saved best fusion model (val_loss: {val_loss:.4f})")

        logger.info(f"\nFusion Module training complete! Best val_loss: {best_val_loss:.4f}")

    def _validate_static(self, val_loader: DataLoader, compute_metrics: bool = False) -> Tuple[float, float]:
        """Validate static encoder"""
        self.static_encoder.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                pdgs = batch['pdg']
                labels = batch['label'].to(self.device)

                try:
                    static_features, vuln_scores = self.static_encoder(pdgs, None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(all_scores, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    if compute_metrics:
                        all_preds.extend(predicted.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                except:
                    continue

        avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if compute_metrics and len(all_preds) > 0:
            self._compute_per_class_metrics(all_preds, all_labels, "Static Encoder Validation")

        return avg_val_loss, val_acc

    def _validate_dynamic(self, val_loader: DataLoader, compute_metrics: bool = False) -> Tuple[float, float]:
        """Validate dynamic encoder"""
        self.dynamic_encoder.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                traces = batch['execution_traces']
                labels = batch['label'].to(self.device)

                try:
                    all_traces = []
                    for trace_list in traces:
                        if trace_list:
                            all_traces.append(trace_list[0] if trace_list else {'steps': []})
                        else:
                            all_traces.append({'steps': []})

                    dynamic_features, vuln_scores = self.dynamic_encoder(all_traces, None)
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(all_scores, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    if compute_metrics:
                        all_preds.extend(predicted.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                except:
                    continue

        avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if compute_metrics and len(all_preds) > 0:
            self._compute_per_class_metrics(all_preds, all_labels, "Dynamic Encoder Validation")

        return avg_val_loss, val_acc

    def _validate_semantic(self, val_loader: DataLoader, compute_metrics: bool = False) -> Tuple[float, float]:
        """Validate semantic encoder"""
        self.semantic_encoder.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                source_codes = batch['source_code']
                labels = batch['label'].to(self.device)
                vuln_types = batch['vulnerability_type']

                try:
                    semantic_features, vuln_scores = self.semantic_encoder(
                        source_codes, vuln_types
                    )
                    all_scores = torch.cat([v for v in vuln_scores.values()], dim=1)

                    loss = self.criterion(all_scores, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(all_scores, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    if compute_metrics:
                        all_preds.extend(predicted.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                except:
                    continue

        avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if compute_metrics and len(all_preds) > 0:
            self._compute_per_class_metrics(all_preds, all_labels, "Semantic Encoder Validation")

        return avg_val_loss, val_acc

    def _validate_fusion(self, val_loader: DataLoader, compute_metrics: bool = False) -> Tuple[float, float]:
        """Validate fusion module"""
        self.static_encoder.eval()
        self.dynamic_encoder.eval()
        self.semantic_encoder.eval()
        self.fusion_module.eval()

        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                source_codes = batch['source_code']
                pdgs = batch['pdg']
                traces = batch['execution_traces']
                labels = batch['label'].to(self.device)

                try:
                    static_features, _ = self.static_encoder(pdgs, None)

                    all_traces = []
                    for trace_list in traces:
                        if trace_list:
                            all_traces.append(trace_list[0] if trace_list else {'steps': []})
                        else:
                            all_traces.append({'steps': []})

                    dynamic_features, _ = self.dynamic_encoder(all_traces, None)
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

                    if compute_metrics:
                        all_preds.extend(predicted.cpu().numpy().tolist())
                        all_labels.extend(labels.cpu().numpy().tolist())
                except:
                    continue

        avg_val_loss = val_loss / len(val_loader) if val_loss > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0

        if compute_metrics and len(all_preds) > 0:
            self._compute_per_class_metrics(all_preds, all_labels, "Fusion Module Validation")

        return avg_val_loss, val_acc

    def _save_checkpoint(self, model, filename, metadata=None, optimizer=None):
        """Save model checkpoint with optimizer state for resuming"""
        checkpoint_path = self.output_dir / filename

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metadata': metadata or {}
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self, model, filename, optimizer=None):
        """Load model checkpoint and optionally restore optimizer state"""
        checkpoint_path = self.output_dir / filename

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        metadata = checkpoint.get('metadata', {})
        logger.info(f"Loaded checkpoint from {filename}: {metadata}")

        return metadata

    def _save_all_models(self, epoch, val_loss, val_acc, optimizer=None):
        """Save all models with optional optimizer state"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self._save_checkpoint(
            self.static_encoder,
            f"static_encoder_fusion_best.pt",
            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc}
        )

        self._save_checkpoint(
            self.dynamic_encoder,
            f"dynamic_encoder_fusion_best.pt",
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
            {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc},
            optimizer=optimizer
        )

        logger.info(f"✓ Saved all models for epoch {epoch}")


def calculate_class_weights(dataset: MultiModalDataset, num_classes: int = 11) -> torch.Tensor:
    """Calculate class weights to handle imbalanced dataset"""
    # Count samples per class
    class_counts = torch.zeros(num_classes)
    for label in dataset.labels:
        class_counts[label] += 1

    # Calculate weights (inverse frequency)
    # Add small epsilon to avoid division by zero
    class_weights = 1.0 / (class_counts + 1e-6)

    # Normalize weights to sum to num_classes
    class_weights = class_weights / class_weights.sum() * num_classes

    logger.info("\nClass weights calculated:")
    vuln_types_inv = {v: k for k, v in dataset.vuln_types.items()}
    for i in range(num_classes):
        if class_counts[i] > 0:
            vuln_name = vuln_types_inv.get(i, f"class_{i}")
            logger.info(f"  {vuln_name}: count={int(class_counts[i])}, weight={class_weights[i]:.4f}")

    return class_weights


def main():
    parser = argparse.ArgumentParser(description="Complete Triton Training Pipeline")
    parser.add_argument(
        "--train-dir",
        required=True,
        help="Training data directory (e.g., data/datasets/forge_balanced_accurate/train)"
    )
    parser.add_argument(
        "--val-dir",
        default=None,
        help="Validation data directory (e.g., data/datasets/forge_balanced_accurate/val). If not provided, will split train_dir 80/20"
    )
    parser.add_argument(
        "--test-dir",
        default=None,
        help="Test data directory (e.g., data/datasets/forge_balanced_accurate/test). Used for final evaluation after training"
    )
    parser.add_argument(
        "--output-dir",
        default="models/checkpoints",
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
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
        default=10000,
        help="Maximum number of samples to use (default: 10000, set to None for all)"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--skip-tests",
        action='store_true',
        help="Skip encoder tests"
    )
    parser.add_argument(
        "--train-mode",
        choices=['all', 'static', 'dynamic', 'semantic', 'fusion'],
        default='all',
        help="Which components to train"
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COMPLETE TRITON TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Training directory: {args.train_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Training mode: {args.train_mode}")

    # Load dataset
    logger.info("\nLoading dataset...")

    if args.val_dir:
        # Load separate train and val datasets
        logger.info(f"Loading training data from: {args.train_dir}")
        train_dataset = MultiModalDataset(args.train_dir, max_samples=args.max_samples, use_cache=True)

        logger.info(f"Loading validation data from: {args.val_dir}")
        val_dataset = MultiModalDataset(args.val_dir, max_samples=None, use_cache=True)

        # Calculate class weights from training set only
        class_weights = calculate_class_weights(train_dataset)
    else:
        # Load and split single dataset
        logger.info(f"Loading data from: {args.train_dir}")
        dataset = MultiModalDataset(args.train_dir, max_samples=args.max_samples, use_cache=True)

        # Calculate class weights for imbalanced dataset
        class_weights = calculate_class_weights(dataset)

        # Split dataset 80/20
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    logger.info(f"\nTraining samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Initialize trainer with class weights
    trainer = CompleteTritonTrainer(
        output_dir=args.output_dir,
        device=args.device,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        class_weights=class_weights
    )

    # Test encoders
    if not args.skip_tests:
        logger.info("\n" + "=" * 80)
        logger.info("TESTING ALL ENCODERS")
        logger.info("=" * 80)

        trainer.test_static_encoder(val_loader)
        trainer.test_dynamic_encoder(val_loader)
        trainer.test_semantic_encoder(val_loader)

    # Train components
    logger.info("\n" + "=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)

    if args.train_mode in ['all', 'static']:
        trainer.train_static_encoder(train_loader, val_loader)

    if args.train_mode in ['all', 'dynamic']:
        trainer.train_dynamic_encoder(train_loader, val_loader)

    if args.train_mode in ['all', 'semantic']:
        trainer.train_semantic_encoder(train_loader, val_loader)

    if args.train_mode in ['all', 'fusion']:
        trainer.train_fusion_module(train_loader, val_loader)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nModel checkpoints saved to: {args.output_dir}")

    # Final test evaluation if test_dir provided
    if args.test_dir:
        logger.info("\n" + "=" * 80)
        logger.info("FINAL TEST SET EVALUATION")
        logger.info("=" * 80)
        logger.info(f"Loading test data from: {args.test_dir}")

        test_dataset = MultiModalDataset(args.test_dir, max_samples=None, use_cache=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        logger.info(f"Test samples: {len(test_dataset)}")

        # Evaluate each component on test set
        logger.info("\nEvaluating trained models on test set...")

        test_loss_static, test_acc_static = trainer._validate_static(test_loader)
        logger.info(f"Static Encoder  - Test Loss: {test_loss_static:.4f}, Test Acc: {test_acc_static:.2f}%")

        test_loss_dynamic, test_acc_dynamic = trainer._validate_dynamic(test_loader)
        logger.info(f"Dynamic Encoder - Test Loss: {test_loss_dynamic:.4f}, Test Acc: {test_acc_dynamic:.2f}%")

        test_loss_semantic, test_acc_semantic = trainer._validate_semantic(test_loader)
        logger.info(f"Semantic Encoder - Test Loss: {test_loss_semantic:.4f}, Test Acc: {test_acc_semantic:.2f}%")

        test_loss_fusion, test_acc_fusion = trainer._validate_fusion(test_loader)
        logger.info(f"Fusion Module   - Test Loss: {test_loss_fusion:.4f}, Test Acc: {test_acc_fusion:.2f}%")

        logger.info("\n" + "=" * 80)
        logger.info("FINAL TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Static:  {test_acc_static:.2f}%")
        logger.info(f"Dynamic: {test_acc_dynamic:.2f}%")
        logger.info(f"Semantic: {test_acc_semantic:.2f}%")
        logger.info(f"Fusion:  {test_acc_fusion:.2f}%")
    else:
        logger.info("\nNext steps:")
        logger.info(f"1. Test with: python scripts/train_complete_pipeline.py --train-dir {args.train_dir} --val-dir {args.val_dir} --test-dir <test_dir>")
        logger.info("2. Compare performance across encoders")
        logger.info("3. Analyze fusion benefits")


if __name__ == "__main__":
    main()
