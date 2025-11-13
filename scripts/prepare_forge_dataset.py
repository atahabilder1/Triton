#!/usr/bin/env python3
"""
Prepare FORGE Dataset for Training
Maps 303 CWE codes → 10 vulnerability classes
Creates balanced train/val/test splits
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set
import argparse
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CWE → 10 CLASS MAPPING (This is the key!)
# ============================================================================

CWE_TO_CLASS = {
    # ========================
    # 1. ACCESS CONTROL
    # ========================
    'CWE-284': 'access_control',  # Improper Access Control
    'CWE-269': 'access_control',  # Improper Privilege Management
    'CWE-285': 'access_control',  # Improper Authorization
    'CWE-862': 'access_control',  # Missing Authorization
    'CWE-732': 'access_control',  # Incorrect Permission Assignment
    'CWE-266': 'access_control',  # Incorrect Privilege Assignment
    'CWE-287': 'access_control',  # Improper Authentication
    'CWE-306': 'access_control',  # Missing Authentication
    'CWE-639': 'access_control',  # Authorization Bypass

    # ========================
    # 2. ARITHMETIC
    # ========================
    'CWE-682': 'arithmetic',      # Incorrect Calculation
    'CWE-190': 'arithmetic',      # Integer Overflow
    'CWE-191': 'arithmetic',      # Integer Underflow
    'CWE-369': 'arithmetic',      # Divide by Zero
    'CWE-128': 'arithmetic',      # Wrap-around Error
    'CWE-1339': 'arithmetic',     # Insufficient Precision

    # ========================
    # 3. UNCHECKED LOW LEVEL CALLS
    # ========================
    'CWE-703': 'unchecked_low_level_calls',  # Improper Exception Handling
    'CWE-252': 'unchecked_low_level_calls',  # Unchecked Return Value
    'CWE-476': 'unchecked_low_level_calls',  # NULL Pointer Dereference
    'CWE-754': 'unchecked_low_level_calls',  # Improper Check for Unusual Conditions
    'CWE-755': 'unchecked_low_level_calls',  # Improper Exception Handling
    'CWE-758': 'unchecked_low_level_calls',  # Undefined Behavior

    # ========================
    # 4. REENTRANCY
    # ========================
    'CWE-841': 'reentrancy',      # Improper Enforcement of Behavioral Workflow
    'CWE-362': 'reentrancy',      # Race Condition (can also be front_running)
    'CWE-667': 'reentrancy',      # Improper Locking
    'CWE-691': 'reentrancy',      # Insufficient Control Flow Management
    'CWE-1265': 'reentrancy',     # Unintended Reentrant Invocation

    # ========================
    # 5. BAD RANDOMNESS
    # ========================
    'CWE-330': 'bad_randomness',  # Use of Insufficiently Random Values
    'CWE-338': 'bad_randomness',  # Use of Cryptographically Weak PRNG
    'CWE-335': 'bad_randomness',  # Incorrect Usage of Seeds in PRNG
    'CWE-336': 'bad_randomness',  # Same Seed in PRNG

    # ========================
    # 6. DENIAL OF SERVICE
    # ========================
    'CWE-400': 'denial_of_service',  # Uncontrolled Resource Consumption
    'CWE-835': 'denial_of_service',  # Loop with Unreachable Exit Condition
    'CWE-770': 'denial_of_service',  # Allocation without Limits
    'CWE-834': 'denial_of_service',  # Excessive Iteration
    'CWE-405': 'denial_of_service',  # Asymmetric Resource Consumption

    # ========================
    # 7. FRONT RUNNING
    # ========================
    'CWE-362': 'front_running',   # Race Condition (overlaps with reentrancy)
    'CWE-663': 'front_running',   # Use of a Non-reentrant Function

    # ========================
    # 8. TIME MANIPULATION
    # ========================
    'CWE-829': 'time_manipulation',  # Inclusion of Functionality from Untrusted Source
    'CWE-347': 'time_manipulation',  # Improper Verification of Cryptographic Signature
    'CWE-367': 'time_manipulation',  # Time-of-check Time-of-use (TOCTOU)

    # ========================
    # 9. SHORT ADDRESSES
    # ========================
    'CWE-130': 'short_addresses',    # Improper Handling of Length Parameter
    'CWE-129': 'short_addresses',    # Improper Validation of Array Index

    # ========================
    # 10. OTHER (unmapped CWEs)
    # ========================
    'CWE-710': 'other',           # Coding Standard Violation
    'CWE-664': 'other',           # Improper Control of Resource
    'CWE-693': 'other',           # Protection Mechanism Failure
    'CWE-20': 'other',            # Improper Input Validation
    'CWE-435': 'other',           # Improper Interaction Between Multiple Entities
    'CWE-1041': 'other',          # Unnecessary Code
    'CWE-1068': 'other',          # Inconsistency Between Implementation and Documentation
    'CWE-1076': 'other',          # Insufficient Adherence to Expected Conventions
    'CWE-1164': 'other',          # Unused Variable
    'CWE-561': 'other',           # Dead Code
    'CWE-563': 'other',           # Unused Variable
}


def map_cwe_to_class(cwes: List[str]) -> str:
    """
    Map a list of CWE codes to a single vulnerability class
    Uses priority: specific vulnerabilities > generic
    """
    if not cwes:
        return 'other'

    # Priority order (more specific first)
    priority_classes = [
        'reentrancy',
        'arithmetic',
        'bad_randomness',
        'time_manipulation',
        'short_addresses',
        'front_running',
        'denial_of_service',
        'unchecked_low_level_calls',
        'access_control',
        'other'
    ]

    # Get all mapped classes
    mapped_classes = set()
    for cwe in cwes:
        if cwe in CWE_TO_CLASS:
            mapped_classes.add(CWE_TO_CLASS[cwe])

    # Return highest priority class
    for priority_class in priority_classes:
        if priority_class in mapped_classes:
            return priority_class

    return 'other'


def prepare_forge_dataset(
    forge_dir: str,
    output_dir: str,
    samples_per_class: Dict[str, int],
    split_ratio: tuple = (0.7, 0.15, 0.15),
    seed: int = 42
):
    """
    Prepare FORGE dataset with train/val/test splits

    Args:
        forge_dir: Path to FORGE-Artifacts/dataset
        output_dir: Output directory for organized dataset
        samples_per_class: Target number of samples per class
        split_ratio: (train, val, test) split ratios
        seed: Random seed for reproducibility
    """

    random.seed(seed)

    forge_path = Path(forge_dir)
    results_dir = forge_path / "results"
    contracts_dir = forge_path / "contracts"
    output_path = Path(output_dir)

    logger.info("="*80)
    logger.info("FORGE DATASET PREPARATION")
    logger.info("="*80)
    logger.info(f"Source: {forge_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target samples per class: {samples_per_class}")
    logger.info(f"Split ratio: {split_ratio}")
    logger.info("")

    # ========================================================================
    # STEP 1: Parse all JSON files and map to 10 classes
    # ========================================================================

    logger.info("Step 1: Parsing FORGE JSON files...")

    class_to_projects = defaultdict(list)  # class -> list of (project_name, contract_paths)
    cwe_counts = Counter()
    class_counts = Counter()

    json_files = list(results_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} audit reports")

    for json_file in json_files:
        try:
            with open(json_file) as f:
                data = json.load(f)

            findings = data.get('findings', [])
            project_info = data.get('project_info', {})
            project_paths = project_info.get('project_path', {})

            # Extract all CWE codes from findings
            all_cwes = []
            for finding in findings:
                category = finding.get('category', {})
                for level, cwes in category.items():
                    all_cwes.extend(cwes)
                    for cwe in cwes:
                        cwe_counts[cwe] += 1

            # Map to vulnerability class
            if not findings or not all_cwes:
                vuln_class = 'safe'
            else:
                vuln_class = map_cwe_to_class(all_cwes)

            # Get contract file paths
            contract_files = []
            for project_name, rel_path in project_paths.items():
                full_path = contracts_dir / rel_path
                if full_path.exists():
                    sol_files = list(full_path.rglob("*.sol"))
                    contract_files.extend(sol_files)

            if contract_files:
                class_to_projects[vuln_class].append({
                    'project_name': json_file.stem,
                    'contracts': contract_files,
                    'cwes': all_cwes
                })
                class_counts[vuln_class] += len(contract_files)

        except Exception as e:
            logger.warning(f"Error processing {json_file.name}: {e}")
            continue

    logger.info("")
    logger.info("="*80)
    logger.info("MAPPING RESULTS")
    logger.info("="*80)
    logger.info(f"Total projects processed: {len(json_files)}")
    logger.info(f"Total contracts found: {sum(class_counts.values())}")
    logger.info("")
    logger.info("Contracts per class:")
    for vuln_class in sorted(class_counts.keys()):
        count = class_counts[vuln_class]
        projects = len(class_to_projects[vuln_class])
        logger.info(f"  {vuln_class:30s}: {count:5d} contracts ({projects:4d} projects)")

    logger.info("")
    logger.info("Top 10 CWE codes found:")
    for cwe, count in cwe_counts.most_common(10):
        mapped_class = CWE_TO_CLASS.get(cwe, 'other')
        logger.info(f"  {cwe:15s} → {mapped_class:25s}: {count:4d} findings")

    # ========================================================================
    # STEP 2: Sample contracts for balanced dataset
    # ========================================================================

    logger.info("")
    logger.info("="*80)
    logger.info("Step 2: Creating balanced dataset...")
    logger.info("="*80)

    sampled_contracts = defaultdict(list)  # class -> list of contract paths

    for vuln_class, target_samples in samples_per_class.items():
        available_projects = class_to_projects.get(vuln_class, [])

        # Collect all contracts from all projects
        all_contracts = []
        for project in available_projects:
            all_contracts.extend(project['contracts'])

        available = len(all_contracts)

        if available == 0:
            logger.warning(f"  {vuln_class}: No contracts available!")
            continue

        # Sample or use all
        if available >= target_samples:
            sampled = random.sample(all_contracts, target_samples)
            logger.info(f"  {vuln_class:30s}: Sampled {len(sampled)}/{available} contracts")
        else:
            sampled = all_contracts
            logger.warning(f"  {vuln_class:30s}: Only {available}/{target_samples} available (using all)")

        sampled_contracts[vuln_class] = sampled

    # ========================================================================
    # STEP 3: Split into train/val/test
    # ========================================================================

    logger.info("")
    logger.info("="*80)
    logger.info("Step 3: Creating train/val/test splits...")
    logger.info("="*80)

    splits = {'train': {}, 'val': {}, 'test': {}}
    train_ratio, val_ratio, test_ratio = split_ratio

    for vuln_class, contracts in sampled_contracts.items():
        n_total = len(contracts)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        random.shuffle(contracts)

        splits['train'][vuln_class] = contracts[:n_train]
        splits['val'][vuln_class] = contracts[n_train:n_train+n_val]
        splits['test'][vuln_class] = contracts[n_train+n_val:]

        logger.info(f"  {vuln_class:30s}: {n_train:4d} train, {n_val:4d} val, {n_test:4d} test")

    # ========================================================================
    # STEP 4: Copy files to output directory
    # ========================================================================

    logger.info("")
    logger.info("="*80)
    logger.info("Step 4: Copying files...")
    logger.info("="*80)

    total_copied = 0

    for split_name, split_data in splits.items():
        logger.info(f"\nCopying {split_name} split...")

        for vuln_class, contracts in split_data.items():
            # Create output directory
            output_class_dir = output_path / split_name / vuln_class
            output_class_dir.mkdir(parents=True, exist_ok=True)

            # Copy contracts
            for contract_path in contracts:
                try:
                    # Create unique filename (project_name + contract_name)
                    project_name = contract_path.parent.name
                    new_filename = f"{project_name}_{contract_path.name}"
                    dest_path = output_class_dir / new_filename

                    shutil.copy2(contract_path, dest_path)
                    total_copied += 1

                except Exception as e:
                    logger.warning(f"Error copying {contract_path}: {e}")
                    continue

            logger.info(f"  {vuln_class:30s}: {len(contracts):4d} contracts copied")

    # ========================================================================
    # STEP 5: Create summary
    # ========================================================================

    logger.info("")
    logger.info("="*80)
    logger.info("DATASET CREATION COMPLETE!")
    logger.info("="*80)
    logger.info(f"Total contracts copied: {total_copied}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Dataset structure:")
    logger.info(f"  {output_dir}/")
    logger.info(f"    ├── train/")
    logger.info(f"    ├── val/")
    logger.info(f"    └── test/")
    logger.info(f"         ├── safe/")
    logger.info(f"         ├── access_control/")
    logger.info(f"         ├── arithmetic/")
    logger.info(f"         ├── reentrancy/")
    logger.info(f"         └── ... (10 classes total)")
    logger.info("")
    logger.info("Next steps:")
    logger.info(f"  1. Verify: ls {output_dir}/train/")
    logger.info(f"  2. Train: python scripts/train_complete_pipeline.py --train-dir {output_dir}/train")
    logger.info("")

    # Save summary JSON
    summary = {
        'total_contracts': total_copied,
        'splits': {
            split_name: {
                vuln_class: len(contracts)
                for vuln_class, contracts in split_data.items()
            }
            for split_name, split_data in splits.items()
        },
        'cwe_mapping': CWE_TO_CLASS,
        'seed': seed
    }

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FORGE dataset for training")
    parser.add_argument(
        "--forge-dir",
        default="/data/llm_projects/triton_datasets/FORGE-Artifacts/dataset",
        help="Path to FORGE-Artifacts/dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        default="data/datasets/forge_balanced",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Target samples per class
    samples_per_class = {
        'safe': 1000,
        'access_control': 1000,
        'arithmetic': 1000,
        'unchecked_low_level_calls': 1000,
        'reentrancy': 800,
        'bad_randomness': 300,
        'denial_of_service': 300,
        'front_running': 300,
        'time_manipulation': 300,
        'short_addresses': 200,
        'other': 500
    }

    prepare_forge_dataset(
        forge_dir=args.forge_dir,
        output_dir=args.output_dir,
        samples_per_class=samples_per_class,
        split_ratio=(0.7, 0.15, 0.15),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
