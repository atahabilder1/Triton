#!/usr/bin/env python3
"""
Triton: Agentic Multimodal Representation for Smart Contract Vulnerability Detection

Main entry point for the Triton vulnerability detection system.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion
from orchestrator.agentic_workflow import AgenticOrchestrator
from utils.metrics import VulnerabilityMetrics


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('triton.log')
        ]
    )


def load_contract(file_path: str) -> str:
    """Load smart contract source code from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logging.error(f"Failed to load contract from {file_path}: {e}")
        raise


def initialize_triton(
    device: str = 'cpu',
    confidence_threshold: float = 0.9,
    max_iterations: int = 5
) -> AgenticOrchestrator:
    """Initialize the Triton vulnerability detection system."""

    logging.info("Initializing Triton components...")

    # Initialize encoders
    static_encoder = StaticEncoder(
        node_feature_dim=128,
        hidden_dim=256,
        output_dim=768,
        dropout=0.2
    ).to(device)

    dynamic_encoder = DynamicEncoder(
        vocab_size=50,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout=0.2
    ).to(device)

    semantic_encoder = SemanticEncoder(
        model_name="microsoft/graphcodebert-base",
        output_dim=768,
        max_length=512,
        dropout=0.1
    ).to(device)

    # Initialize fusion module
    fusion_module = CrossModalFusion(
        static_dim=768,
        dynamic_dim=512,
        semantic_dim=768,
        hidden_dim=512,
        output_dim=768,
        dropout=0.1
    ).to(device)

    # Initialize orchestrator
    orchestrator = AgenticOrchestrator(
        static_encoder=static_encoder,
        dynamic_encoder=dynamic_encoder,
        semantic_encoder=semantic_encoder,
        fusion_module=fusion_module,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations
    )

    logging.info("Triton initialization complete")
    return orchestrator


def analyze_contract(
    orchestrator: AgenticOrchestrator,
    source_code: str,
    contract_name: Optional[str] = None,
    target_vulnerability: Optional[str] = None,
    output_file: Optional[str] = None
) -> dict:
    """Analyze a smart contract for vulnerabilities."""

    logging.info(f"Starting analysis for contract: {contract_name or 'unnamed'}")

    if target_vulnerability:
        logging.info(f"Targeting vulnerability type: {target_vulnerability}")

    # Perform analysis
    result = orchestrator.analyze_contract(
        source_code=source_code,
        contract_name=contract_name,
        target_vulnerability=target_vulnerability
    )

    # Log results
    final_result = result['final_result']
    workflow_summary = result['workflow_summary']

    logging.info(f"Analysis complete:")
    logging.info(f"  - Vulnerability detected: {final_result.vulnerability_detected}")
    logging.info(f"  - Vulnerability type: {final_result.vulnerability_type}")
    logging.info(f"  - Confidence: {final_result.confidence:.3f}")
    logging.info(f"  - Total iterations: {workflow_summary['total_iterations']}")
    logging.info(f"  - Early stopping: {workflow_summary['early_stopping']}")

    # Save results if output file specified
    if output_file:
        save_results(result, output_file)

    return result


def save_results(result: dict, output_file: str):
    """Save analysis results to JSON file."""

    # Convert result to serializable format
    serializable_result = {
        'final_result': {
            'vulnerability_detected': result['final_result'].vulnerability_detected,
            'vulnerability_type': result['final_result'].vulnerability_type,
            'confidence': result['final_result'].confidence,
            'reasoning': result['final_result'].reasoning,
            'phase': result['final_result'].phase.value,
            'modality_contributions': result['final_result'].modality_contributions
        },
        'workflow_summary': result['workflow_summary'],
        'analysis_history': [
            {
                'phase': r.phase.value,
                'confidence': r.confidence,
                'vulnerability_type': r.vulnerability_type,
                'reasoning': r.reasoning,
                'modality_contributions': r.modality_contributions
            }
            for r in result['analysis_history']
        ]
    }

    try:
        with open(output_file, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        logging.info(f"Results saved to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save results: {e}")


def print_summary(result: dict):
    """Print a formatted summary of the analysis results."""

    final_result = result['final_result']
    workflow_summary = result['workflow_summary']

    print("\n" + "="*50)
    print("TRITON VULNERABILITY ANALYSIS SUMMARY")
    print("="*50)

    print(f"Vulnerability Detected: {'YES' if final_result.vulnerability_detected else 'NO'}")
    print(f"Vulnerability Type: {final_result.vulnerability_type}")
    print(f"Confidence Score: {final_result.confidence:.3f}")
    print(f"Analysis Iterations: {workflow_summary['total_iterations']}")
    print(f"Early Stopping: {'YES' if workflow_summary['early_stopping'] else 'NO'}")

    print(f"\nModality Contributions:")
    for modality, contribution in final_result.modality_contributions.items():
        print(f"  - {modality.capitalize()}: {contribution:.3f}")

    print(f"\nReasoning: {final_result.reasoning}")

    print(f"\nPhases Executed: {', '.join(workflow_summary['phases_executed'])}")

    print("="*50)


def main():
    """Main entry point."""

    parser = argparse.ArgumentParser(
        description="Triton: Agentic Multimodal Smart Contract Vulnerability Detection"
    )

    parser.add_argument(
        'contract_file',
        help='Path to the Solidity contract file'
    )

    parser.add_argument(
        '--contract-name',
        help='Name of the contract to analyze (optional)'
    )

    parser.add_argument(
        '--target-vulnerability',
        choices=[
            'reentrancy', 'overflow', 'underflow', 'access_control',
            'unchecked_call', 'timestamp_dependency', 'tx_origin',
            'delegatecall', 'self_destruct', 'gas_limit'
        ],
        help='Specific vulnerability type to target (optional)'
    )

    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.9,
        help='Confidence threshold for early stopping (default: 0.9)'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=5,
        help='Maximum number of analysis iterations (default: 5)'
    )

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to use for computation (default: cpu)'
    )

    parser.add_argument(
        '--output',
        help='Output file to save results (JSON format)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        # Check if CUDA is available if requested
        if args.device == 'cuda' and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, falling back to CPU")
            args.device = 'cpu'

        # Load contract
        source_code = load_contract(args.contract_file)

        # Initialize Triton
        orchestrator = initialize_triton(
            device=args.device,
            confidence_threshold=args.confidence_threshold,
            max_iterations=args.max_iterations
        )

        # Analyze contract
        result = analyze_contract(
            orchestrator=orchestrator,
            source_code=source_code,
            contract_name=args.contract_name,
            target_vulnerability=args.target_vulnerability,
            output_file=args.output
        )

        # Print summary
        print_summary(result)

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()