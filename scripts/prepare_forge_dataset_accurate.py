#!/usr/bin/env python3
"""
Prepare FORGE Dataset with ACCURATE CWE Mapping
Based on official CWE database + smart contract security research
Maps 303 CWE codes → 10 vulnerability classes
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
# COMPREHENSIVE CWE → 10 CLASS MAPPING
# Based on: CWE database + Smart Contract Security Patterns + FORGE analysis
# ============================================================================

CWE_TO_CLASS = {
    # ========================================================================
    # 1. ACCESS_CONTROL
    # Improper access control, privilege management, authorization
    # ========================================================================
    'CWE-284': 'access_control',  # Improper Access Control (BASE - 6138 findings)
    'CWE-269': 'access_control',  # Improper Privilege Management (2827 findings)
    'CWE-285': 'access_control',  # Improper Authorization (1038 findings)
    'CWE-862': 'access_control',  # Missing Authorization (493 findings)
    'CWE-863': 'access_control',  # Incorrect Authorization (272 findings)
    'CWE-732': 'access_control',  # Incorrect Permission Assignment (253 findings)
    'CWE-266': 'access_control',  # Incorrect Privilege Assignment (576 findings)
    'CWE-287': 'access_control',  # Improper Authentication (34 findings)
    'CWE-306': 'access_control',  # Missing Authentication
    'CWE-639': 'access_control',  # Authorization Bypass
    'CWE-282': 'access_control',  # Improper Ownership Management (312 findings)
    'CWE-250': 'access_control',  # Execution with Unnecessary Privileges (113 findings)
    'CWE-267': 'access_control',  # Privilege Defined with Unsafe Actions (94 findings)
    'CWE-749': 'access_control',  # Exposed Dangerous Method (147 findings)
    'CWE-766': 'access_control',  # Critical Data Element Without Access Control (128 findings)

    # ========================================================================
    # 2. ARITHMETIC
    # Integer overflow/underflow, incorrect calculations, wrap-around
    # ========================================================================
    'CWE-682': 'arithmetic',      # Incorrect Calculation (3250 findings)
    'CWE-190': 'arithmetic',      # Integer Overflow/Wraparound (202 findings)
    'CWE-191': 'arithmetic',      # Integer Underflow (43 findings)
    'CWE-369': 'arithmetic',      # Divide by Zero (52 findings)
    'CWE-128': 'arithmetic',      # Wrap-around Error
    'CWE-1339': 'arithmetic',     # Insufficient Precision/Accuracy (283 findings)
    'CWE-193': 'arithmetic',      # Off-by-one Error (105 findings)
    'CWE-680': 'arithmetic',      # Integer Overflow to Buffer Overflow
    'CWE-1335': 'arithmetic',     # Incorrect Bitwise Shift

    # ========================================================================
    # 3. UNCHECKED_LOW_LEVEL_CALLS
    # Unchecked return values, improper error handling, external call failures
    # ========================================================================
    'CWE-703': 'unchecked_low_level_calls',  # Improper Error Handling (3763 findings)
    'CWE-252': 'unchecked_low_level_calls',  # Unchecked Return Value (472 findings)
    'CWE-476': 'unchecked_low_level_calls',  # NULL Pointer Dereference
    'CWE-754': 'unchecked_low_level_calls',  # Improper Check for Unusual Conditions (2653 findings)
    'CWE-755': 'unchecked_low_level_calls',  # Improper Exception Handling (915 findings)
    'CWE-758': 'unchecked_low_level_calls',  # Undefined/Unspecified Behavior (164 findings)
    'CWE-705': 'unchecked_low_level_calls',  # Incorrect Control Flow (99 findings)
    'CWE-253': 'unchecked_low_level_calls',  # Incorrect Check of Function Return Value (36 findings)
    'CWE-394': 'unchecked_low_level_calls',  # Unexpected Status Code/Return Value (79 findings)
    'CWE-390': 'unchecked_low_level_calls',  # Detection of Error Without Action (110 findings)
    'CWE-392': 'unchecked_low_level_calls',  # Missing Report of Error Condition (83 findings)
    'CWE-393': 'unchecked_low_level_calls',  # Return of Wrong Status Code (70 findings)
    'CWE-error': 'unchecked_low_level_calls',  # Generic error handling issues

    # ========================================================================
    # 4. REENTRANCY
    # Race conditions, improper locking, unexpected reentrant calls, TOCTOU
    # ========================================================================
    'CWE-841': 'reentrancy',      # Improper Enforcement of Behavioral Workflow (62 findings)
    'CWE-362': 'reentrancy',      # Race Condition (98 findings) - PRIMARY
    'CWE-667': 'reentrancy',      # Improper Locking
    'CWE-691': 'reentrancy',      # Insufficient Control Flow (1432 findings)
    'CWE-1265': 'reentrancy',     # Unintended Reentrant Invocation (281 findings)
    'CWE-366': 'reentrancy',      # Race Condition within Thread
    'CWE-367': 'reentrancy',      # Time-of-check Time-of-use (TOCTOU)
    'CWE-663': 'reentrancy',      # Use of Non-reentrant Function
    'CWE-662': 'reentrancy',      # Improper Synchronization (39 findings)
    'CWE-366': 'reentrancy',      # Race Condition within Thread
    'CWE-1223': 'reentrancy',     # Race Condition for Write-Once Attributes

    # ========================================================================
    # 5. BAD_RANDOMNESS
    # Weak or predictable randomness, improper use of randomness
    # ========================================================================
    'CWE-330': 'bad_randomness',  # Use of Insufficiently Random Values
    'CWE-338': 'bad_randomness',  # Use of Cryptographically Weak PRNG
    'CWE-335': 'bad_randomness',  # Incorrect Usage of Seeds in PRNG
    'CWE-336': 'bad_randomness',  # Same Seed in PRNG
    'CWE-337': 'bad_randomness',  # Predictable Seed in PRNG
    'CWE-340': 'bad_randomness',  # Generation of Predictable Numbers/IDs
    'CWE-343': 'bad_randomness',  # Predictable Value Range from Previous Values

    # ========================================================================
    # 6. DENIAL_OF_SERVICE
    # Resource exhaustion, unbounded loops, gas limit issues
    # ========================================================================
    'CWE-400': 'denial_of_service',  # Uncontrolled Resource Consumption (409 findings)
    'CWE-835': 'denial_of_service',  # Loop with Unreachable Exit
    'CWE-770': 'denial_of_service',  # Allocation without Limits (234 findings)
    'CWE-834': 'denial_of_service',  # Excessive Iteration (217 findings)
    'CWE-405': 'denial_of_service',  # Asymmetric Resource Consumption
    'CWE-674': 'denial_of_service',  # Uncontrolled Recursion
    'CWE-772': 'denial_of_service',  # Missing Release of Resource (90 findings)
    'CWE-404': 'denial_of_service',  # Improper Resource Shutdown (138 findings)
    'CWE-476': 'denial_of_service',  # NULL Pointer Dereference (can cause DoS)
    'CWE-617': 'denial_of_service',  # Reachable Assertion
    'CWE-909': 'denial_of_service',  # Missing Initialization (73 findings)

    # ========================================================================
    # 7. FRONT_RUNNING
    # Transaction ordering, MEV, race conditions in transaction ordering
    # ========================================================================
    'CWE-362': 'front_running',   # Race Condition (overlaps - context dependent)
    'CWE-663': 'front_running',   # Use of Non-reentrant Function
    'CWE-829': 'front_running',   # Inclusion of Functionality from Untrusted Source (89 findings)
    'CWE-807': 'front_running',   # Reliance on Untrusted Inputs (104 findings)
    'CWE-841': 'front_running',   # Improper Enforcement of Behavioral Workflow

    # ========================================================================
    # 8. TIME_MANIPULATION
    # Timestamp dependence, block number manipulation
    # ========================================================================
    'CWE-829': 'time_manipulation',  # Inclusion of Untrusted Functionality (89 findings)
    'CWE-347': 'time_manipulation',  # Improper Verification of Signatures (29 findings)
    'CWE-367': 'time_manipulation',  # TOCTOU Race Condition
    'CWE-345': 'time_manipulation',  # Insufficient Verification of Data (42 findings)
    'CWE-346': 'time_manipulation',  # Origin Validation Error (36 findings)
    'CWE-354': 'time_manipulation',  # Improper Validation of Integrity Check (49 findings)

    # ========================================================================
    # 9. SHORT_ADDRESSES
    # Length parameter issues, array index validation
    # ========================================================================
    'CWE-130': 'short_addresses',  # Improper Handling of Length Parameter
    'CWE-129': 'short_addresses',  # Improper Validation of Array Index
    'CWE-787': 'short_addresses',  # Out-of-bounds Write
    'CWE-125': 'short_addresses',  # Out-of-bounds Read
    'CWE-805': 'short_addresses',  # Buffer Access with Incorrect Length

    # ========================================================================
    # 10. OTHER
    # Code quality, documentation, best practices, uncategorized
    # ========================================================================
    'CWE-710': 'other',           # Coding Standard Violation (8885 findings - most common!)
    'CWE-664': 'other',           # Improper Control of Resource (1756 findings)
    'CWE-693': 'other',           # Protection Mechanism Failure (626 findings)
    'CWE-20': 'other',            # Improper Input Validation (594 findings)
    'CWE-435': 'other',           # Improper Interaction (626 findings)
    'CWE-1041': 'other',          # Unnecessary Code (2638 findings)
    'CWE-1068': 'other',          # Inconsistency Between Code/Docs (804 findings)
    'CWE-1076': 'other',          # Insufficient Conventions (533 findings)
    'CWE-1164': 'other',          # Unused Variable (485 findings)
    'CWE-561': 'other',           # Dead Code (210 findings)
    'CWE-563': 'other',           # Unused Variable (172 findings)
    'CWE-670': 'other',           # Always-Incorrect Control Flow (398 findings)
    'CWE-697': 'other',           # Incorrect Comparison (396 findings)
    'CWE-1126': 'other',          # Declaration of Variable with Unnecessarily Wide Scope (335 findings)
    'CWE-1059': 'other',          # Incomplete Documentation (309 findings)
    'CWE-707': 'other',           # Improper Neutralization (247 findings)
    'CWE-1025': 'other',          # Comparison Using Wrong Factors (221 findings)
    'CWE-221': 'other',           # Information Loss (188 findings)
    'CWE-666': 'other',           # Operation on Resource (179 findings)
    'CWE-431': 'other',           # Missing Handler (171 findings)
    'CWE-1329': 'other',          # Reliance on Component That is Not Updateable (169 findings)
    'CWE-477': 'other',           # Use of Obsolete Function (165 findings)
    'CWE-436': 'other',           # Interpretation Conflict (151 findings)
    'CWE-665': 'other',           # Improper Initialization (124 findings)
    'CWE-1061': 'other',          # Insufficient Encapsulation (123 findings)
    'CWE-778': 'other',           # Insufficient Logging (122 findings)
    'CWE-223': 'other',           # Omission of Security-relevant Information (117 findings)
    'CWE-193': 'other',           # Off-by-one Error (105 findings)
    'CWE-372': 'other',           # Incomplete Internal State Distinction (77 findings)
    'CWE-1120': 'other',          # Excessive Code Complexity (77 findings)
    'CWE-1023': 'other',          # Incomplete Comparison (73 findings)
    'CWE-1078': 'other',          # Inappropriate Source Code Style (73 findings)
    'CWE-672': 'other',           # Operation on Resource after Expiration (68 findings)
    'CWE-228': 'other',           # Improper Handling of Syntactically Invalid Structure (65 findings)
    'CWE-483': 'other',           # Incorrect Block Delimitation (60 findings)
    'CWE-708': 'other',           # Incorrect Ownership Assignment (59 findings)
    'CWE-1357': 'other',          # Reliance on Insufficiently Trustworthy Component (59 findings)
    'CWE-437': 'other',           # Incomplete Model of Endpoint Features (58 findings)
    'CWE-799': 'other',           # Improper Control of Interaction Frequency (51 findings)
    'CWE-456': 'other',           # Missing Initialization (50 findings)
    'CWE-684': 'other',           # Incorrect Provision of Specified Functionality (48 findings)
    'CWE-459': 'other',           # Incomplete Cleanup (43 findings)
    'CWE-1024': 'other',          # Comparison of Incompatible Types (39 findings)
    'CWE-1053': 'other',          # Missing Documentation (38 findings)
    'CWE-1127': 'other',          # Compilation with Insufficient Warnings (38 findings)
    'CWE-358': 'other',           # Improperly Implemented Security Check (38 findings)
    'CWE-115': 'other',           # Misinterpretation of Input (35 findings)
    'CWE-826': 'other',           # Premature Release of Resource (34 findings)
    'CWE-1099': 'other',          # Inconsistent Naming Conventions (33 findings)
    'CWE-676': 'other',           # Use of Potentially Dangerous Function (32 findings)
    'CWE-1110': 'other',          # Incomplete Design Documentation (29 findings)
    'CWE-940': 'other',           # Improper Verification of Source (27 findings)
    'CWE-1038': 'other',          # Insecure Automated Optimizations (25 findings)
}


def map_cwe_to_class(cwes: List[str]) -> str:
    """
    Map a list of CWE codes to a single vulnerability class

    Priority order (most specific to least specific):
    1. Reentrancy (critical for smart contracts)
    2. Arithmetic (critical for smart contracts)
    3. Bad randomness (critical for smart contracts)
    4. Time manipulation (smart contract specific)
    5. Short addresses (smart contract specific)
    6. Front running (smart contract specific)
    7. Denial of service
    8. Unchecked calls
    9. Access control
    10. Other
    """
    if not cwes:
        return 'other'

    # Priority order (most specific vulnerabilities first)
    priority_classes = [
        'reentrancy',           # Most critical
        'arithmetic',           # Most critical
        'bad_randomness',       # Smart contract specific
        'time_manipulation',    # Smart contract specific
        'short_addresses',      # Smart contract specific
        'front_running',        # Smart contract specific
        'denial_of_service',    # High severity
        'unchecked_low_level_calls',  # Common but less severe
        'access_control',       # Common but context-dependent
        'other'                 # Catch-all
    ]

    # Collect all mapped classes
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
    logger.info("FORGE DATASET PREPARATION - ACCURATE CWE MAPPING")
    logger.info("="*80)
    logger.info(f"Source: {forge_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Target samples per class: {samples_per_class}")
    logger.info(f"Split ratio: {split_ratio}")
    logger.info("")

    # ========================================================================
    # STEP 1: Parse all JSON files and map to 10 classes
    # ========================================================================

    logger.info("Step 1: Parsing FORGE JSON files with accurate CWE mapping...")

    class_to_contracts = defaultdict(list)  # class -> list of contract paths
    cwe_counts = Counter()
    class_counts = Counter()
    unmapped_cwes = Counter()

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
                        if cwe not in CWE_TO_CLASS:
                            unmapped_cwes[cwe] += 1

            # Map to vulnerability class
            if not findings or not all_cwes:
                vuln_class = 'safe'
            else:
                vuln_class = map_cwe_to_class(all_cwes)

            # Get contract file paths
            contract_files = []
            for project_name, rel_path in project_paths.items():
                # rel_path is relative to dataset root (e.g., "contracts/$joke/JOKECOMMUNITY")
                full_path = forge_path / rel_path
                if full_path.exists():
                    sol_files = list(full_path.rglob("*.sol"))
                    contract_files.extend(sol_files)

            if contract_files:
                class_to_contracts[vuln_class].extend(contract_files)
                class_counts[vuln_class] += len(contract_files)

        except Exception as e:
            logger.warning(f"Error processing {json_file.name}: {e}")
            continue

    logger.info("")
    logger.info("="*80)
    logger.info("ACCURATE MAPPING RESULTS")
    logger.info("="*80)
    logger.info(f"Total projects processed: {len(json_files)}")
    logger.info(f"Total contracts found: {sum(class_counts.values())}")
    logger.info(f"Total unique CWEs: {len(cwe_counts)}")
    logger.info(f"Mapped CWEs: {len(CWE_TO_CLASS)}")
    logger.info(f"Unmapped CWEs: {len(unmapped_cwes)}")
    logger.info("")
    logger.info("Contracts per class (after accurate mapping):")
    for vuln_class in sorted(class_counts.keys()):
        count = class_counts[vuln_class]
        pct = 100 * count / sum(class_counts.values())
        logger.info(f"  {vuln_class:30s}: {count:6d} contracts ({pct:5.1f}%)")

    if unmapped_cwes:
        logger.info("")
        logger.info("Top 10 unmapped CWEs (will be categorized as 'other'):")
        for cwe, count in unmapped_cwes.most_common(10):
            logger.info(f"  {cwe:15s}: {count:4d} findings")

    # ========================================================================
    # STEP 2: Sample contracts for balanced dataset
    # ========================================================================

    logger.info("")
    logger.info("="*80)
    logger.info("Step 2: Creating balanced dataset...")
    logger.info("="*80)

    sampled_contracts = defaultdict(list)

    for vuln_class, target_samples in samples_per_class.items():
        available_contracts = class_to_contracts.get(vuln_class, [])
        available = len(available_contracts)

        if available == 0:
            logger.warning(f"  {vuln_class:30s}: No contracts available!")
            continue

        # Sample or use all
        if available >= target_samples:
            sampled = random.sample(available_contracts, target_samples)
            logger.info(f"  {vuln_class:30s}: Sampled {len(sampled):4d}/{available:5d} contracts")
        else:
            sampled = available_contracts
            logger.warning(f"  {vuln_class:30s}: Only {available:4d}/{target_samples:4d} available (using all)")

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
            output_class_dir = output_path / split_name / vuln_class
            output_class_dir.mkdir(parents=True, exist_ok=True)

            for contract_path in contracts:
                try:
                    # Create unique filename
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
    logger.info("Next steps:")
    logger.info(f"  1. Verify: ls {output_dir}/train/")
    logger.info(f"  2. Train: python scripts/train_complete_pipeline.py --train-dir {output_dir}/train")
    logger.info("")

    # Save summary
    summary = {
        'total_contracts': total_copied,
        'splits': {
            split_name: {
                vuln_class: len(contracts)
                for vuln_class, contracts in split_data.items()
            }
            for split_name, split_data in splits.items()
        },
        'cwe_mapping_coverage': {
            'total_cwes_in_dataset': len(cwe_counts),
            'mapped_cwes': len(CWE_TO_CLASS),
            'unmapped_cwes': len(unmapped_cwes)
        },
        'seed': seed
    }

    summary_path = output_path / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare FORGE dataset with accurate CWE mapping")
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

    # Target samples per class (adjusted based on availability)
    samples_per_class = {
        'safe': 1000,                      # ~1,141 available
        'access_control': 1000,            # ~10,000+ available
        'arithmetic': 1000,                # ~3,500+ available
        'unchecked_low_level_calls': 1000, # ~8,000+ available
        'reentrancy': 800,                 # ~1,500+ available
        'bad_randomness': 300,             # Limited availability
        'denial_of_service': 500,          # ~1,000+ available
        'front_running': 300,              # Limited availability
        'time_manipulation': 300,          # Limited availability
        'short_addresses': 200,            # Very limited
        'other': 1000                      # ~12,000+ available
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
