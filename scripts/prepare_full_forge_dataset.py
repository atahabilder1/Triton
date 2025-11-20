#!/usr/bin/env python3
"""
Prepare FULL FORGE Dataset - Use ALL 6,449 contracts
Strategy: Include interfaces and contracts without audit files
"""
import os
import sys
import json
import shutil
import logging
import re
from pathlib import Path
from collections import defaultdict, Counter
import random
import hashlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_contract_type(filepath: Path) -> str:
    """
    Detect if contract is interface, abstract, or concrete
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().lower()
            
        # Check for interface
        if re.search(r'\binterface\s+\w+', content):
            return 'interface'
        
        # Check for abstract
        if re.search(r'\babstract\s+contract\s+\w+', content):
            return 'abstract'
        
        # Check for library
        if re.search(r'\blibrary\s+\w+', content):
            return 'library'
            
        return 'contract'
    except:
        return 'unknown'

def infer_vulnerability_from_filename(filename: str) -> str:
    """
    Infer vulnerability type from filename patterns
    Based on audit report naming conventions
    """
    filename_lower = filename.lower()
    
    # Reentrancy patterns
    if any(term in filename_lower for term in ['reentr', 'reentran', 'callback', 'race', 'toctou']):
        return 'reentrancy'
    
    # Arithmetic patterns  
    if any(term in filename_lower for term in ['overflow', 'underflow', 'arithmetic', 'safemath', 'math', 'integer']):
        return 'arithmetic'
    
    # Access control patterns
    if any(term in filename_lower for term in ['access', 'auth', 'owner', 'admin', 'privilege', 'permission', 'role']):
        return 'access_control'
    
    # Unchecked calls patterns
    if any(term in filename_lower for term in ['unchecked', 'call', 'delegatecall', 'send', 'transfer', 'return']):
        return 'unchecked_low_level_calls'
    
    # DOS patterns
    if any(term in filename_lower for term in ['dos', 'denial', 'gas', 'loop', 'unbounded', 'block']):
        return 'denial_of_service'
    
    # Time manipulation patterns
    if any(term in filename_lower for term in ['timestamp', 'time', 'block.number', 'now', 'temporal']):
        return 'time_manipulation'
    
    # Randomness patterns
    if any(term in filename_lower for term in ['random', 'rng', 'seed', 'nonce', 'predictable']):
        return 'bad_randomness'
    
    # Front-running patterns  
    if any(term in filename_lower for term in ['frontrun', 'front-run', 'mev', 'order']):
        return 'front_running'
    
    # Safe contracts (tokens, standard implementations)
    if any(term in filename_lower for term in ['erc20', 'erc721', 'erc1155', 'standard', 'openzeppelin']):
        return 'safe'
    
    return 'other'

def extract_vulnerability_from_existing_labels(forge_reconstructed_dir: Path) -> dict:
    """
    Extract known labels from forge_reconstructed directory
    Returns: {contract_filename: vulnerability_class}
    """
    known_labels = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = forge_reconstructed_dir / split
        if not split_dir.exists():
            continue
            
        for vuln_dir in split_dir.iterdir():
            if not vuln_dir.is_dir():
                continue
                
            vuln_class = vuln_dir.name
            for contract_file in vuln_dir.glob('*.sol'):
                # Extract original filename (remove prefix if added)
                filename = contract_file.name
                known_labels[filename] = vuln_class
    
    return known_labels

def prepare_full_forge_dataset(
    forge_flattened_dir: str,
    forge_reconstructed_dir: str,
    output_dir: str,
    include_interfaces: bool = True,
    split_ratio: tuple = (0.7, 0.15, 0.15),
    seed: int = 42
):
    """
    Prepare dataset using ALL 6,449 contracts
    
    Labeling strategy:
    1. Use known labels from forge_reconstructed
    2. Infer from filename patterns
    3. Label interfaces/libraries as 'safe' by default
    4. Everything else as 'other'
    """
    random.seed(seed)
    
    forge_flat_path = Path(forge_flattened_dir)
    forge_recon_path = Path(forge_reconstructed_dir)
    output_path = Path(output_dir)
    
    logger.info("="*80)
    logger.info("FULL FORGE DATASET PREPARATION")
    logger.info("="*80)
    logger.info(f"Source (flattened): {forge_flattened_dir}")
    logger.info(f"Labels (reconstructed): {forge_reconstructed_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Include interfaces: {include_interfaces}")
    logger.info("")
    
    # Step 1: Get known labels
    logger.info("Step 1: Extracting known labels from forge_reconstructed...")
    known_labels = extract_vulnerability_from_existing_labels(forge_recon_path)
    logger.info(f"Found {len(known_labels)} contracts with known labels")
    logger.info("")
    
    # Step 2: Process all contracts
    logger.info("Step 2: Processing all 6,449 contracts...")
    
    all_contracts = list(forge_flat_path.glob("*.sol"))
    logger.info(f"Found {len(all_contracts)} total contracts")
    
    contract_info = []
    type_counts = Counter()
    label_counts = Counter()
    label_sources = Counter()
    
    for contract_file in all_contracts:
        # Skip empty or tiny files
        if contract_file.stat().st_size < 50:
            continue
            
        filename = contract_file.name
        contract_type = detect_contract_type(contract_file)
        type_counts[contract_type] += 1
        
        # Determine label
        if filename in known_labels:
            label = known_labels[filename]
            label_source = 'known'
        elif contract_type == 'interface' and include_interfaces:
            label = 'safe'  # Most interfaces are just definitions
            label_source = 'interface_heuristic'
        elif contract_type == 'library':
            label = 'safe'  # Libraries are typically utilities
            label_source = 'library_heuristic'
        else:
            label = infer_vulnerability_from_filename(filename)
            label_source = 'filename_inference'
        
        contract_info.append({
            'file': contract_file,
            'filename': filename,
            'type': contract_type,
            'label': label,
            'label_source': label_source
        })
        
        label_counts[label] += 1
        label_sources[label_source] += 1
    
    logger.info(f"\nProcessed {len(contract_info)} valid contracts")
    logger.info(f"\nContract types:")
    for ctype, count in type_counts.most_common():
        logger.info(f"  {ctype:15s}: {count:5d}")
    
    logger.info(f"\nLabel distribution:")
    for label, count in label_counts.most_common():
        pct = 100 * count / len(contract_info)
        logger.info(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")
    
    logger.info(f"\nLabel sources:")
    for source, count in label_sources.most_common():
        pct = 100 * count / len(contract_info)
        logger.info(f"  {source:30s}: {count:5d} ({pct:5.1f}%)")
    
    # Step 3: Create balanced splits
    logger.info("\n" + "="*80)
    logger.info("Step 3: Creating train/val/test splits...")
    logger.info("="*80)
    
    # Group by label
    contracts_by_label = defaultdict(list)
    for info in contract_info:
        contracts_by_label[info['label']].append(info)
    
    # Split each class
    splits = {'train': [], 'val': [], 'test': []}
    train_ratio, val_ratio, test_ratio = split_ratio
    
    for label, contracts in contracts_by_label.items():
        n_total = len(contracts)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val
        
        random.shuffle(contracts)
        
        splits['train'].extend(contracts[:n_train])
        splits['val'].extend(contracts[n_train:n_train+n_val])
        splits['test'].extend(contracts[n_train+n_val:])
        
        logger.info(f"  {label:30s}: {n_train:4d} train, {n_val:4d} val, {n_test:4d} test")
    
    # Step 4: Copy files
    logger.info("\n" + "="*80)
    logger.info("Step 4: Copying files to output directory...")
    logger.info("="*80)
    
    total_copied = 0
    
    for split_name, contracts in splits.items():
        logger.info(f"\nCopying {split_name} split ({len(contracts)} contracts)...")
        
        # Group by label for organized output
        by_label = defaultdict(list)
        for contract in contracts:
            by_label[contract['label']].append(contract)
        
        for label, label_contracts in by_label.items():
            output_class_dir = output_path / split_name / label
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            for contract in label_contracts:
                try:
                    dest_path = output_class_dir / contract['filename']
                    shutil.copy2(contract['file'], dest_path)
                    total_copied += 1
                except Exception as e:
                    logger.warning(f"Error copying {contract['filename']}: {e}")
                    
            logger.info(f"  {label:30s}: {len(label_contracts):4d} contracts")
    
    # Step 5: Save summary
    logger.info("\n" + "="*80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total contracts: {total_copied}")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    summary = {
        'total_contracts': total_copied,
        'source_contracts': len(all_contracts),
        'known_labels': len(known_labels),
        'label_distribution': dict(label_counts),
        'label_sources': dict(label_sources),
        'type_distribution': dict(type_counts),
        'splits': {
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        },
        'include_interfaces': include_interfaces,
        'seed': seed
    }
    
    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("\nNext step: Train with PDG extraction on this dataset!")
    logger.info(f"  python scripts/train_static_optimized.py --train-dir {output_dir}/train")
    

if __name__ == "__main__":
    prepare_full_forge_dataset(
        forge_flattened_dir="data/datasets/forge_cleaned",  # Use CLEANED contracts!
        forge_reconstructed_dir="data/datasets/forge_reconstructed",
        output_dir="data/datasets/forge_full_cleaned",
        include_interfaces=True,  # Include everything!
        split_ratio=(0.7, 0.15, 0.15),
        seed=42
    )
