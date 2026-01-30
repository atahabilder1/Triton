#!/usr/bin/env python3
"""
Binary Classification Training Script for Triton

This implements the two-stage approach:
Stage 1: Binary classification (Vulnerable vs Safe) using ML
Stage 2: Tool-based type classification using Slither/Mythril

Key Features:
1. Binary classification (2 classes: vulnerable=1, safe=0)
2. Focal Loss with high recall focus
3. Threshold tuning for recall optimization
4. Integration with tool-based type classification
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, precision_recall_curve
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.cross_modal_fusion import CrossModalFusion
from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder


# =============================================================================
# STEP 1: Focal Loss for Binary Classification (High Recall Focus)
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss optimized for binary classification with HIGH RECALL focus.

    Key insight: Missing a vulnerability (False Negative) is much worse than
    a false alarm (False Positive) in security applications.

    Formula: L = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Parameters:
        alpha_vulnerable (float): Weight for vulnerable class (default 2.0 for higher recall)
        alpha_safe (float): Weight for safe class (default 0.25 for lower importance)
        gamma (float): Focusing parameter (default 2.0)
    """

    def __init__(
        self,
        alpha_vulnerable: float = 2.0,  # High weight for vulnerabilities
        alpha_safe: float = 0.25,        # Low weight for safe class
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super(BinaryFocalLoss, self).__init__()
        # alpha[0] = safe, alpha[1] = vulnerable
        self.alpha = torch.tensor([alpha_safe, alpha_vulnerable])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits of shape (batch_size, 2)
            targets: Binary labels of shape (batch_size,) where 0=safe, 1=vulnerable
        """
        # Move alpha to same device as inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # Standard cross-entropy
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')

        # Get probability of true class
        pt = torch.exp(-ce_loss)

        # Apply focal term
        focal_term = (1 - pt) ** self.gamma

        # Apply alpha weighting (per sample based on target class)
        alpha_t = self.alpha[targets]

        # Final focal loss
        focal_loss = alpha_t * focal_term * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# STEP 2: Binary Label Converter
# =============================================================================

class BinaryLabelConverter:
    """
    Converts multi-class vulnerability labels to binary labels.

    Binary mapping:
        - Vulnerable (any type) = 1
        - Safe = 0
    """

    # Original 11-class mapping
    MULTICLASS_LABELS = {
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

    # Binary mapping
    BINARY_LABELS = {
        'vulnerable': 1,
        'safe': 0
    }

    @classmethod
    def to_binary(cls, label: int) -> int:
        """Convert multi-class label to binary label."""
        # If label is 10 (safe), return 0; otherwise return 1 (vulnerable)
        return 0 if label == 10 else 1

    @classmethod
    def to_binary_from_name(cls, label_name: str) -> int:
        """Convert vulnerability name to binary label."""
        return 0 if label_name.lower() == 'safe' else 1

    @classmethod
    def batch_to_binary(cls, labels: torch.Tensor) -> torch.Tensor:
        """Convert a batch of multi-class labels to binary."""
        # 10 = safe -> 0, everything else -> 1
        return (labels != 10).long()


# =============================================================================
# STEP 3: Tool-Based Type Classifier (Stage 2)
# =============================================================================

class ToolBasedTypeClassifier:
    """
    Comprehensive tool-based vulnerability type classification.

    Supports multiple security analysis tools:
    - Slither (static analysis)
    - Mythril (symbolic execution)
    - Securify2 (formal verification)
    - Solhint (linting & best practices)
    - Manticore (symbolic execution)
    - Echidna (fuzzing - indirect patterns)

    This is Stage 2 of the pipeline - only called when Stage 1 (ML binary)
    predicts "vulnerable".
    """

    # =========================================================================
    # SLITHER MAPPINGS (93 detectors)
    # =========================================================================
    SLITHER_MAPPING = {
        # --- Reentrancy (5 detectors) ---
        'reentrancy-eth': 'reentrancy',
        'reentrancy-no-eth': 'reentrancy',
        'reentrancy-benign': 'reentrancy',
        'reentrancy-events': 'reentrancy',
        'reentrancy-unlimited-gas': 'reentrancy',

        # --- Access Control (15 detectors) ---
        'unprotected-upgrade': 'access_control',
        'protected-vars': 'access_control',
        'suicidal': 'access_control',
        'arbitrary-send-eth': 'access_control',
        'arbitrary-send-erc20': 'access_control',
        'arbitrary-send-erc20-permit': 'access_control',
        'tx-origin': 'access_control',
        'uninitialized-state': 'access_control',
        'uninitialized-storage': 'access_control',
        'uninitialized-local': 'access_control',
        'missing-zero-check': 'access_control',
        'controlled-array-length': 'access_control',
        'write-after-write': 'access_control',
        'msg-value-loop': 'access_control',
        'delegatecall-loop': 'access_control',

        # --- Arithmetic (8 detectors) ---
        'divide-before-multiply': 'arithmetic',
        'unchecked-transfer': 'arithmetic',
        'incorrect-equality': 'arithmetic',
        'tautology': 'arithmetic',
        'boolean-cst': 'arithmetic',
        'incorrect-unary': 'arithmetic',
        'unused-return': 'arithmetic',
        'shadowing-builtin': 'arithmetic',

        # --- Denial of Service (10 detectors) ---
        'locked-ether': 'denial_of_service',
        'calls-loop': 'denial_of_service',
        'costly-loop': 'denial_of_service',
        'array-by-reference': 'denial_of_service',
        'incorrect-modifier': 'denial_of_service',
        'mapping-deletion': 'denial_of_service',
        'unimplemented-functions': 'denial_of_service',
        'void-cst': 'denial_of_service',
        'assert-state-change': 'denial_of_service',
        'constant-function-state': 'denial_of_service',

        # --- Time Manipulation (3 detectors) ---
        'timestamp': 'time_manipulation',
        'block-timestamp': 'time_manipulation',
        'incorrect-block-hash': 'time_manipulation',

        # --- Bad Randomness (2 detectors) ---
        'weak-prng': 'bad_randomness',
        'blockhash': 'bad_randomness',

        # --- Unchecked Low Level Calls (6 detectors) ---
        'unchecked-lowlevel': 'unchecked_low_level_calls',
        'low-level-calls': 'unchecked_low_level_calls',
        'unchecked-send': 'unchecked_low_level_calls',
        'unchecked-transfer': 'unchecked_low_level_calls',
        'multiple-send': 'unchecked_low_level_calls',
        'return-bomb': 'unchecked_low_level_calls',

        # --- Front Running (3 detectors) ---
        'erc20-interface': 'front_running',
        'erc721-interface': 'front_running',
        'locked-ether': 'front_running',

        # --- Short Addresses (2 detectors) ---
        'erc20-indexed': 'short_addresses',
        'missing-inheritance': 'short_addresses',

        # --- Other/Delegatecall (15+ detectors) ---
        'controlled-delegatecall': 'other',
        'deprecated-standards': 'other',
        'shadowing-state': 'other',
        'shadowing-local': 'other',
        'shadowing-abstract': 'other',
        'constable-states': 'other',
        'external-function': 'other',
        'naming-convention': 'other',
        'pragma': 'other',
        'solc-version': 'other',
        'dead-code': 'other',
        'reentrancy-no-eth': 'other',
        'assembly': 'other',
        'encode-packed-collision': 'other',
        'incorrect-using-for': 'other',
        'public-mappings-nested': 'other',
    }

    # =========================================================================
    # MYTHRIL MAPPINGS (SWC Registry - 37 weakness types)
    # =========================================================================
    MYTHRIL_MAPPING = {
        # --- Reentrancy ---
        'SWC-107': 'reentrancy',
        'Reentrancy': 'reentrancy',
        'External Call': 'reentrancy',
        'External Call To User-Supplied Address': 'reentrancy',
        'State change after external call': 'reentrancy',

        # --- Arithmetic ---
        'SWC-101': 'arithmetic',
        'Integer Overflow': 'arithmetic',
        'Integer Underflow': 'arithmetic',
        'Integer Overflow and Underflow': 'arithmetic',
        'SWC-129': 'arithmetic',  # Typographical Error

        # --- Access Control ---
        'SWC-106': 'access_control',
        'Unprotected Selfdestruct': 'access_control',
        'Unprotected SUICIDE': 'access_control',
        'SWC-115': 'access_control',  # Authorization through tx.origin
        'SWC-105': 'access_control',  # Unprotected Ether Withdrawal
        'SWC-112': 'access_control',  # Delegatecall to Untrusted Callee
        'SWC-124': 'access_control',  # Write to Arbitrary Storage Location
        'Unprotected Ether Withdrawal': 'access_control',
        'Dependence on tx.origin': 'access_control',

        # --- Unchecked Low Level ---
        'SWC-104': 'unchecked_low_level_calls',
        'Unchecked Return Value': 'unchecked_low_level_calls',
        'Unchecked Call Return Value': 'unchecked_low_level_calls',
        'SWC-113': 'unchecked_low_level_calls',  # DoS with Failed Call

        # --- Time/Block Dependency ---
        'SWC-116': 'time_manipulation',
        'Block Timestamp': 'time_manipulation',
        'Timestamp Dependence': 'time_manipulation',
        'Block values as a proxy for time': 'time_manipulation',

        # --- Bad Randomness ---
        'SWC-120': 'bad_randomness',
        'Weak Sources of Randomness': 'bad_randomness',
        'Weak Randomness': 'bad_randomness',
        'SWC-136': 'bad_randomness',  # Unencrypted Private Data On-Chain

        # --- Denial of Service ---
        'SWC-128': 'denial_of_service',
        'DoS With Block Gas Limit': 'denial_of_service',
        'DoS with Failed Call': 'denial_of_service',
        'SWC-126': 'denial_of_service',  # Insufficient Gas Griefing
        'SWC-134': 'denial_of_service',  # Message call with hardcoded gas amount

        # --- Front Running ---
        'SWC-114': 'front_running',  # Transaction Order Dependence
        'Transaction Order Dependence': 'front_running',
        'Race Condition': 'front_running',

        # --- Other ---
        'SWC-100': 'other',  # Function Default Visibility
        'SWC-102': 'other',  # Outdated Compiler Version
        'SWC-103': 'other',  # Floating Pragma
        'SWC-108': 'other',  # State Variable Default Visibility
        'SWC-109': 'other',  # Uninitialized Storage Pointer
        'SWC-110': 'other',  # Assert Violation
        'SWC-111': 'other',  # Use of Deprecated Solidity Functions
        'SWC-117': 'other',  # Signature Malleability
        'SWC-118': 'other',  # Incorrect Constructor Name
        'SWC-119': 'other',  # Shadowing State Variables
        'SWC-121': 'other',  # Missing Protection against Signature Replay
        'SWC-122': 'other',  # Lack of Proper Signature Verification
        'SWC-123': 'other',  # Requirement Violation
        'SWC-125': 'other',  # Incorrect Inheritance Order
        'SWC-127': 'other',  # Arbitrary Jump with Function Type Variable
        'SWC-130': 'other',  # Right-To-Left-Override control character
        'SWC-131': 'other',  # Presence of unused variables
        'SWC-132': 'other',  # Unexpected Ether balance
        'SWC-133': 'other',  # Hash Collisions With Multiple Variable Length Arguments
        'SWC-135': 'other',  # Code With No Effects
        'SWC-136': 'other',  # Unencrypted Private Data On-Chain
    }

    # =========================================================================
    # SECURIFY2 MAPPINGS (38 patterns)
    # =========================================================================
    SECURIFY2_MAPPING = {
        # --- Reentrancy ---
        'DAO': 'reentrancy',
        'DAOConstantGas': 'reentrancy',
        'Reentrancy': 'reentrancy',
        'ReentrancyNoETH': 'reentrancy',
        'ReentrancyBenign': 'reentrancy',

        # --- Access Control ---
        'UnrestrictedWrite': 'access_control',
        'UnrestrictedEtherFlow': 'access_control',
        'UnrestrictedSelfdestruct': 'access_control',
        'MissingInputValidation': 'access_control',
        'UnhandledException': 'access_control',
        'TxOrigin': 'access_control',
        'ShadowedStateVariable': 'access_control',

        # --- Arithmetic ---
        'TODAmount': 'arithmetic',
        'TODReceiver': 'arithmetic',
        'DivisionByZero': 'arithmetic',
        'Overflow': 'arithmetic',
        'Underflow': 'arithmetic',

        # --- Denial of Service ---
        'LockedEther': 'denial_of_service',
        'UnboundedLoop': 'denial_of_service',
        'CostlyLoop': 'denial_of_service',
        'GasLimitReached': 'denial_of_service',

        # --- Time Manipulation ---
        'TimestampDependence': 'time_manipulation',
        'BlockNumberDependence': 'time_manipulation',

        # --- Bad Randomness ---
        'WeakRandomness': 'bad_randomness',
        'PredictableRandomness': 'bad_randomness',
        'Blockhash': 'bad_randomness',

        # --- Unchecked Low Level ---
        'UncheckedLowLevelCall': 'unchecked_low_level_calls',
        'UncheckedSend': 'unchecked_low_level_calls',
        'UncheckedTransfer': 'unchecked_low_level_calls',
        'MissingReturnValue': 'unchecked_low_level_calls',

        # --- Front Running ---
        'TODTransfer': 'front_running',
        'TransactionOrderDependence': 'front_running',

        # --- Other ---
        'Delegatecall': 'other',
        'UnusedReturn': 'other',
        'DeadCode': 'other',
        'UninitializedStorage': 'other',
        'UninitializedLocal': 'other',
        'AssemblyUsage': 'other',
    }

    # =========================================================================
    # SOLHINT MAPPINGS (Security rules)
    # =========================================================================
    SOLHINT_MAPPING = {
        # --- Reentrancy ---
        'reentrancy': 'reentrancy',
        'check-send-result': 'reentrancy',

        # --- Access Control ---
        'avoid-tx-origin': 'access_control',
        'not-rely-on-time': 'access_control',
        'avoid-suicide': 'access_control',
        'avoid-sha3': 'access_control',

        # --- Arithmetic ---
        'no-unused-vars': 'arithmetic',
        'avoid-low-level-calls': 'arithmetic',

        # --- Denial of Service ---
        'avoid-throw': 'denial_of_service',
        'no-complex-fallback': 'denial_of_service',

        # --- Time Manipulation ---
        'not-rely-on-block-hash': 'time_manipulation',

        # --- Bad Randomness ---
        'no-block-members': 'bad_randomness',

        # --- Unchecked Low Level ---
        'check-send-result': 'unchecked_low_level_calls',
        'avoid-call-value': 'unchecked_low_level_calls',

        # --- Front Running ---
        'state-visibility': 'front_running',
        'func-visibility': 'front_running',

        # --- Other ---
        'compiler-version': 'other',
        'no-inline-assembly': 'other',
        'mark-callable-contracts': 'other',
        'multiple-sends': 'other',
        'no-unused-import': 'other',
        'const-name-snakecase': 'other',
        'contract-name-camelcase': 'other',
        'event-name-camelcase': 'other',
        'func-name-mixedcase': 'other',
        'var-name-mixedcase': 'other',
    }

    # =========================================================================
    # MANTICORE MAPPINGS (Symbolic execution findings)
    # =========================================================================
    MANTICORE_MAPPING = {
        # --- Reentrancy ---
        'external_call_to_untrusted_contract': 'reentrancy',
        'reentrancy': 'reentrancy',
        'state_change_after_call': 'reentrancy',

        # --- Arithmetic ---
        'integer_overflow': 'arithmetic',
        'integer_underflow': 'arithmetic',
        'divide_by_zero': 'arithmetic',

        # --- Access Control ---
        'unprotected_selfdestruct': 'access_control',
        'arbitrary_write': 'access_control',
        'unprotected_ether_withdrawal': 'access_control',
        'tx_origin': 'access_control',

        # --- Unchecked Low Level ---
        'unchecked_return_value': 'unchecked_low_level_calls',
        'unused_return_value': 'unchecked_low_level_calls',

        # --- Denial of Service ---
        'dos_with_unbounded_operations': 'denial_of_service',
        'gas_limit': 'denial_of_service',

        # --- Time Manipulation ---
        'timestamp_dependence': 'time_manipulation',
        'block_number_dependence': 'time_manipulation',

        # --- Bad Randomness ---
        'weak_randomness': 'bad_randomness',
    }

    # =========================================================================
    # VULNERABILITY TYPE COVERAGE SUMMARY
    # =========================================================================
    # 1. access_control      - 35+ detectors across all tools
    # 2. arithmetic          - 20+ detectors (overflow, underflow, etc.)
    # 3. bad_randomness      - 10+ detectors (weak PRNG, blockhash)
    # 4. denial_of_service   - 15+ detectors (gas limit, loops, locks)
    # 5. front_running       - 10+ detectors (TOD, race conditions)
    # 6. reentrancy          - 15+ detectors (most critical)
    # 7. short_addresses     - 5+ detectors (ERC20 related)
    # 8. time_manipulation   - 10+ detectors (timestamp, block)
    # 9. unchecked_low_level - 15+ detectors (calls, sends)
    # 10. other              - 40+ detectors (code quality, misc)
    # =========================================================================

    # Vulnerability type to ID mapping
    VULN_TYPE_TO_ID = {
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
    }

    # Reverse mapping: ID to vulnerability type
    ID_TO_VULN_TYPE = {v: k for k, v in VULN_TYPE_TO_ID.items()}

    @classmethod
    def classify_from_slither(cls, slither_findings: List[Dict]) -> Optional[str]:
        """
        Classify vulnerability type from Slither findings.

        Args:
            slither_findings: List of Slither detector findings

        Returns:
            Vulnerability type name or None if no mapping found
        """
        for finding in slither_findings:
            detector = finding.get('check', finding.get('detector', ''))
            if detector in cls.SLITHER_MAPPING:
                return cls.SLITHER_MAPPING[detector]
        return None

    @classmethod
    def classify_from_mythril(cls, mythril_findings: List[Dict]) -> Optional[str]:
        """
        Classify vulnerability type from Mythril findings.

        Args:
            mythril_findings: List of Mythril issue findings

        Returns:
            Vulnerability type name or None if no mapping found
        """
        for finding in mythril_findings:
            swc_id = finding.get('swc-id', '')
            title = finding.get('title', '')

            # Check SWC ID first
            swc_key = f'SWC-{swc_id}' if swc_id else ''
            if swc_key in cls.MYTHRIL_MAPPING:
                return cls.MYTHRIL_MAPPING[swc_key]

            # Check title
            for key, vuln_type in cls.MYTHRIL_MAPPING.items():
                if key.lower() in title.lower():
                    return vuln_type
        return None

    @classmethod
    def classify_from_securify(cls, securify_findings: List[Dict]) -> Optional[str]:
        """
        Classify vulnerability type from Securify2 findings.

        Args:
            securify_findings: List of Securify2 pattern findings

        Returns:
            Vulnerability type name or None if no mapping found
        """
        for finding in securify_findings:
            pattern = finding.get('pattern', finding.get('name', ''))
            if pattern in cls.SECURIFY2_MAPPING:
                return cls.SECURIFY2_MAPPING[pattern]
        return None

    @classmethod
    def classify_from_solhint(cls, solhint_findings: List[Dict]) -> Optional[str]:
        """
        Classify vulnerability type from Solhint findings.

        Args:
            solhint_findings: List of Solhint rule violations

        Returns:
            Vulnerability type name or None if no mapping found
        """
        for finding in solhint_findings:
            rule = finding.get('ruleId', finding.get('rule', ''))
            if rule in cls.SOLHINT_MAPPING:
                return cls.SOLHINT_MAPPING[rule]
        return None

    @classmethod
    def classify_from_manticore(cls, manticore_findings: List[Dict]) -> Optional[str]:
        """
        Classify vulnerability type from Manticore findings.

        Args:
            manticore_findings: List of Manticore symbolic execution findings

        Returns:
            Vulnerability type name or None if no mapping found
        """
        for finding in manticore_findings:
            issue_type = finding.get('type', finding.get('issue', ''))
            if issue_type in cls.MANTICORE_MAPPING:
                return cls.MANTICORE_MAPPING[issue_type]
            # Also check description for keywords
            desc = finding.get('description', '').lower()
            for key, vuln_type in cls.MANTICORE_MAPPING.items():
                if key.lower() in desc:
                    return vuln_type
        return None

    @classmethod
    def classify(
        cls,
        slither_findings: Optional[List[Dict]] = None,
        mythril_findings: Optional[List[Dict]] = None,
        securify_findings: Optional[List[Dict]] = None,
        solhint_findings: Optional[List[Dict]] = None,
        manticore_findings: Optional[List[Dict]] = None
    ) -> Tuple[str, int]:
        """
        Classify vulnerability type from multiple tool findings.

        Priority order (based on reliability):
        1. Slither (fastest, most detectors)
        2. Mythril (good for runtime issues)
        3. Securify2 (formal verification)
        4. Manticore (deep symbolic execution)
        5. Solhint (code quality)

        Returns:
            Tuple of (vulnerability_type_name, vulnerability_type_id)
        """
        vuln_type = None

        # Try each tool in priority order
        if slither_findings and vuln_type is None:
            vuln_type = cls.classify_from_slither(slither_findings)

        if mythril_findings and vuln_type is None:
            vuln_type = cls.classify_from_mythril(mythril_findings)

        if securify_findings and vuln_type is None:
            vuln_type = cls.classify_from_securify(securify_findings)

        if manticore_findings and vuln_type is None:
            vuln_type = cls.classify_from_manticore(manticore_findings)

        if solhint_findings and vuln_type is None:
            vuln_type = cls.classify_from_solhint(solhint_findings)

        # Default to 'other' if no classification found
        if vuln_type is None:
            vuln_type = 'other'

        return vuln_type, cls.VULN_TYPE_TO_ID.get(vuln_type, 9)

    @classmethod
    def get_all_findings_types(
        cls,
        slither_findings: Optional[List[Dict]] = None,
        mythril_findings: Optional[List[Dict]] = None,
        securify_findings: Optional[List[Dict]] = None,
        solhint_findings: Optional[List[Dict]] = None,
        manticore_findings: Optional[List[Dict]] = None
    ) -> List[Tuple[str, int, str]]:
        """
        Get all vulnerability types found by all tools.

        Returns:
            List of (vulnerability_type, vulnerability_id, source_tool) tuples
        """
        results = []

        if slither_findings:
            vuln_type = cls.classify_from_slither(slither_findings)
            if vuln_type:
                results.append((vuln_type, cls.VULN_TYPE_TO_ID.get(vuln_type, 9), 'slither'))

        if mythril_findings:
            vuln_type = cls.classify_from_mythril(mythril_findings)
            if vuln_type:
                results.append((vuln_type, cls.VULN_TYPE_TO_ID.get(vuln_type, 9), 'mythril'))

        if securify_findings:
            vuln_type = cls.classify_from_securify(securify_findings)
            if vuln_type:
                results.append((vuln_type, cls.VULN_TYPE_TO_ID.get(vuln_type, 9), 'securify'))

        if manticore_findings:
            vuln_type = cls.classify_from_manticore(manticore_findings)
            if vuln_type:
                results.append((vuln_type, cls.VULN_TYPE_TO_ID.get(vuln_type, 9), 'manticore'))

        if solhint_findings:
            vuln_type = cls.classify_from_solhint(solhint_findings)
            if vuln_type:
                results.append((vuln_type, cls.VULN_TYPE_TO_ID.get(vuln_type, 9), 'solhint'))

        return results

    @classmethod
    def get_coverage_summary(cls) -> Dict[str, int]:
        """
        Get summary of detector coverage per vulnerability type.

        Returns:
            Dictionary mapping vulnerability type to number of detectors
        """
        coverage = {vuln_type: 0 for vuln_type in cls.VULN_TYPE_TO_ID.keys()}

        for mapping in [cls.SLITHER_MAPPING, cls.MYTHRIL_MAPPING,
                       cls.SECURIFY2_MAPPING, cls.SOLHINT_MAPPING, cls.MANTICORE_MAPPING]:
            for vuln_type in mapping.values():
                if vuln_type in coverage:
                    coverage[vuln_type] += 1

        return coverage


# =============================================================================
# STEP 4: Binary Classification Trainer with Recall Optimization
# =============================================================================

class BinaryClassificationTrainer:
    """
    Trainer for binary vulnerability classification.

    Features:
    - Binary classification (vulnerable vs safe)
    - Focal Loss with high recall focus
    - Threshold tuning for optimal recall
    - Integration with tool-based type classification
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        alpha_vulnerable: float = 2.0,
        alpha_safe: float = 0.25,
        gamma: float = 2.0,
        target_recall: float = 0.90,  # Target 90% recall
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.target_recall = target_recall

        # Binary Focal Loss with high recall focus
        self.criterion = BinaryFocalLoss(
            alpha_vulnerable=alpha_vulnerable,
            alpha_safe=alpha_safe,
            gamma=gamma
        )

        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # Threshold for binary classification (tuned for high recall)
        self.threshold = 0.3  # Lower threshold = higher recall

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_recall': [],
            'val_precision': [],
            'val_f1': [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            # Get data
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)

            # Convert to binary labels
            binary_labels = BinaryLabelConverter.batch_to_binary(labels)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(features)

            # Get logits (assuming model returns dict with 'vulnerability_logits')
            if isinstance(outputs, dict):
                logits = outputs['vulnerability_logits']
            else:
                logits = outputs

            # Compute loss
            loss = self.criterion(logits, binary_labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate(
        self,
        val_loader: DataLoader,
        find_optimal_threshold: bool = False
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()

        all_logits = []
        all_labels = []
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                binary_labels = BinaryLabelConverter.batch_to_binary(labels)

                outputs = self.model(features)
                if isinstance(outputs, dict):
                    logits = outputs['vulnerability_logits']
                else:
                    logits = outputs

                loss = self.criterion(logits, binary_labels)
                total_loss += loss.item()
                num_batches += 1

                all_logits.append(logits.cpu())
                all_labels.append(binary_labels.cpu())

        # Concatenate all predictions
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Get probabilities
        probs = torch.softmax(all_logits, dim=1)[:, 1]  # Probability of vulnerable

        # Find optimal threshold for target recall
        if find_optimal_threshold:
            self.threshold = self._find_threshold_for_recall(
                probs.numpy(), all_labels.numpy(), self.target_recall
            )

        # Make predictions using threshold
        predictions = (probs >= self.threshold).long()

        # Calculate metrics
        metrics = {
            'loss': total_loss / max(num_batches, 1),
            'accuracy': accuracy_score(all_labels, predictions),
            'recall': recall_score(all_labels, predictions, zero_division=0),
            'precision': precision_score(all_labels, predictions, zero_division=0),
            'f1': f1_score(all_labels, predictions, zero_division=0),
            'threshold': self.threshold,
        }

        return metrics

    def _find_threshold_for_recall(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        target_recall: float
    ) -> float:
        """Find threshold that achieves target recall."""
        precision, recall, thresholds = precision_recall_curve(labels, probs)

        # Find threshold that achieves target recall
        valid_indices = recall >= target_recall
        if valid_indices.any():
            # Get the threshold with highest precision among valid ones
            valid_thresholds = thresholds[valid_indices[:-1]]
            valid_precisions = precision[valid_indices]
            if len(valid_thresholds) > 0:
                best_idx = np.argmax(valid_precisions[:-1])
                return float(valid_thresholds[best_idx])

        # If no threshold achieves target recall, use lowest threshold
        return float(thresholds[0]) if len(thresholds) > 0 else 0.5

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.
        """
        best_recall = 0
        patience_counter = 0

        print(f"\n{'='*60}")
        print("Binary Classification Training")
        print(f"Target Recall: {self.target_recall*100:.0f}%")
        print(f"{'='*60}\n")

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Evaluate (find optimal threshold every 5 epochs)
            metrics = self.evaluate(val_loader, find_optimal_threshold=(epoch % 5 == 0))

            # Update scheduler
            self.scheduler.step(metrics['recall'])

            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(metrics['loss'])
            self.history['val_accuracy'].append(metrics['accuracy'])
            self.history['val_recall'].append(metrics['recall'])
            self.history['val_precision'].append(metrics['precision'])
            self.history['val_f1'].append(metrics['f1'])

            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {metrics['loss']:.4f}")
            print(f"  Val Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"  Val Recall: {metrics['recall']*100:.2f}% (Target: {self.target_recall*100:.0f}%)")
            print(f"  Val Precision: {metrics['precision']*100:.2f}%")
            print(f"  Val F1: {metrics['f1']*100:.2f}%")
            print(f"  Threshold: {metrics['threshold']:.3f}")
            print()

            # Early stopping based on recall
            if metrics['recall'] > best_recall:
                best_recall = metrics['recall']
                patience_counter = 0

                # Save best model
                if save_path:
                    self._save_checkpoint(save_path, epoch, metrics)
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        print(f"\nTraining complete. Best Recall: {best_recall*100:.2f}%")
        return self.history

    def _save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'metrics': metrics,
        }
        torch.save(checkpoint, path)
        print(f"  Saved checkpoint with recall: {metrics['recall']*100:.2f}%")

    def predict(
        self,
        features: torch.Tensor,
        slither_findings: Optional[List[Dict]] = None,
        mythril_findings: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Full two-stage prediction pipeline.

        Stage 1: Binary classification (ML)
        Stage 2: Tool-based type classification (if vulnerable)
        """
        self.model.eval()

        with torch.no_grad():
            features = features.to(self.device)
            outputs = self.model(features)

            if isinstance(outputs, dict):
                logits = outputs['vulnerability_logits']
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            vuln_prob = probs[:, 1].item()  # Probability of vulnerable

            # Stage 1: Binary prediction
            is_vulnerable = vuln_prob >= self.threshold

            result = {
                'stage1_prediction': 'vulnerable' if is_vulnerable else 'safe',
                'stage1_confidence': vuln_prob if is_vulnerable else (1 - vuln_prob),
                'vulnerability_probability': vuln_prob,
                'threshold_used': self.threshold,
            }

            # Stage 2: Type classification (only if vulnerable)
            if is_vulnerable:
                vuln_type, vuln_id = ToolBasedTypeClassifier.classify(
                    slither_findings, mythril_findings
                )
                result['stage2_vulnerability_type'] = vuln_type
                result['stage2_vulnerability_id'] = vuln_id
            else:
                result['stage2_vulnerability_type'] = None
                result['stage2_vulnerability_id'] = None

            return result


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    """Main training function."""
    print("="*60)
    print("Triton Binary Classification Training")
    print("="*60)
    print("\nImplemented Features:")
    print("1. Binary classification (Vulnerable vs Safe)")
    print("2. Focal Loss with high recall focus")
    print("3. Threshold tuning for >90% recall")
    print("4. Tool-based type classification integration")
    print("\n" + "="*60)

    # Configuration
    config = {
        'num_classes': 2,  # Binary
        'learning_rate': 1e-4,
        'alpha_vulnerable': 2.0,  # High weight for recall
        'alpha_safe': 0.25,
        'gamma': 2.0,
        'target_recall': 0.90,
        'batch_size': 32,
        'num_epochs': 50,
    }

    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create model (binary mode)
    print("\nInitializing model for binary classification...")
    model = CrossModalFusion(
        static_dim=768,
        dynamic_dim=512,
        semantic_dim=768,
        num_vulnerability_types=2,  # Binary
        binary_mode=True
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Create trainer
    trainer = BinaryClassificationTrainer(
        model=model,
        learning_rate=config['learning_rate'],
        alpha_vulnerable=config['alpha_vulnerable'],
        alpha_safe=config['alpha_safe'],
        gamma=config['gamma'],
        target_recall=config['target_recall'],
    )

    print("\nTrainer initialized with:")
    print(f"  - Focal Loss (alpha_vuln={config['alpha_vulnerable']}, alpha_safe={config['alpha_safe']}, gamma={config['gamma']})")
    print(f"  - Target Recall: {config['target_recall']*100:.0f}%")
    print(f"  - Initial Threshold: {trainer.threshold}")

    print("\n" + "="*60)
    print("Ready for training!")
    print("To train, provide DataLoaders and call trainer.train()")
    print("="*60)

    # Example usage (with placeholder data)
    print("\n--- Example Two-Stage Prediction ---")
    example_slither = [{'check': 'reentrancy-eth', 'impact': 'High'}]
    example_mythril = [{'swc-id': '107', 'title': 'External Call To User-Supplied Address'}]

    vuln_type, vuln_id = ToolBasedTypeClassifier.classify(
        slither_findings=example_slither,
        mythril_findings=example_mythril
    )
    print(f"Slither findings: {example_slither}")
    print(f"Mythril findings: {example_mythril}")
    print(f"Classified as: {vuln_type} (ID: {vuln_id})")

    return trainer


if __name__ == "__main__":
    main()
