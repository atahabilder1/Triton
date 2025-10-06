import os
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SmartContractDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        max_length: int = 512,
        include_labels: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.max_length = max_length
        self.include_labels = include_labels

        self.contracts = []
        self.labels = []
        self.vulnerabilities = []

        self._load_data()

    def _load_data(self):
        split_file = os.path.join(self.data_dir, f"{self.split}.json")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Data file not found: {split_file}")

        with open(split_file, 'r') as f:
            data = json.load(f)

        for item in data:
            self.contracts.append({
                'source_code': item.get('source_code', ''),
                'contract_address': item.get('address', ''),
                'bytecode': item.get('bytecode', ''),
                'abi': item.get('abi', [])
            })

            if self.include_labels:
                self.labels.append(item.get('has_vulnerability', 0))
                self.vulnerabilities.append(item.get('vulnerability_types', []))

    def __len__(self):
        return len(self.contracts)

    def __getitem__(self, idx):
        contract = self.contracts[idx]

        sample = {
            'source_code': contract['source_code'],
            'contract_address': contract['contract_address'],
            'bytecode': contract['bytecode'],
            'abi': contract['abi']
        }

        if self.include_labels:
            sample['label'] = self.labels[idx]
            sample['vulnerability_types'] = self.vulnerabilities[idx]

        return sample


class MultiModalBatch:
    def __init__(
        self,
        static_features: Optional[torch.Tensor] = None,
        dynamic_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        vulnerability_masks: Optional[torch.Tensor] = None
    ):
        self.static_features = static_features
        self.dynamic_features = dynamic_features
        self.semantic_features = semantic_features
        self.labels = labels
        self.vulnerability_masks = vulnerability_masks


def collate_multimodal(batch: List[Dict]) -> MultiModalBatch:
    source_codes = [item['source_code'] for item in batch]

    labels = None
    vuln_masks = None

    if 'label' in batch[0]:
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)

    if 'vulnerability_types' in batch[0]:
        vuln_types = ['reentrancy', 'overflow', 'underflow', 'access_control',
                      'unchecked_call', 'timestamp_dependency', 'tx_origin',
                      'delegatecall', 'self_destruct', 'gas_limit']

        vuln_masks = torch.zeros(len(batch), len(vuln_types))
        for i, item in enumerate(batch):
            for vuln in item['vulnerability_types']:
                if vuln in vuln_types:
                    vuln_masks[i, vuln_types.index(vuln)] = 1

    return MultiModalBatch(
        static_features=torch.zeros(len(batch), 768),
        dynamic_features=torch.zeros(len(batch), 512),
        semantic_features=torch.zeros(len(batch), 768),
        labels=labels,
        vulnerability_masks=vuln_masks
    )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: str = "train",
    val_split: str = "val",
    test_split: str = "test"
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    train_dataset = SmartContractDataset(data_dir, split=train_split)
    val_dataset = SmartContractDataset(data_dir, split=val_split)
    test_dataset = SmartContractDataset(data_dir, split=test_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_multimodal
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_multimodal
    )

    return train_loader, val_loader, test_loader