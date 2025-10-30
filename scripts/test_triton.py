#!/usr/bin/env python3
"""
Triton Testing Script
Tests Triton on benchmark datasets and generates performance reports.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion
from orchestrator.agentic_workflow import AgenticOrchestrator
from utils.data_loader import SmartContractDataset

import torch
import glob


class TritonSystem:
    """Wrapper class for Triton system"""
    def __init__(self, device='cpu', checkpoint_dir='models/checkpoints'):
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Initialize encoders
        self.static_encoder = StaticEncoder(
            node_feature_dim=128,
            hidden_dim=256,
            output_dim=768,
            dropout=0.2
        ).to(device)

        self.dynamic_encoder = DynamicEncoder(
            vocab_size=50,
            embedding_dim=128,
            hidden_dim=256,
            output_dim=512,
            dropout=0.2
        ).to(device)

        self.semantic_encoder = SemanticEncoder(
            model_name="microsoft/graphcodebert-base",
            output_dim=768,
            max_length=512,
            dropout=0.1
        ).to(device)

        # Initialize fusion module
        self.fusion_module = CrossModalFusion(
            static_dim=768,
            dynamic_dim=512,
            semantic_dim=768,
            hidden_dim=512,
            output_dim=768,
            dropout=0.1
        ).to(device)

        # Load trained weights if available
        self._load_checkpoints()

        # Initialize orchestrator
        self.orchestrator = AgenticOrchestrator(
            static_encoder=self.static_encoder,
            dynamic_encoder=self.dynamic_encoder,
            semantic_encoder=self.semantic_encoder,
            fusion_module=self.fusion_module,
            confidence_threshold=0.9,
            max_iterations=5
        )

    def _load_checkpoints(self):
        """Load trained model weights from checkpoint directory"""
        checkpoint_path = Path(self.checkpoint_dir)

        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint directory not found: {checkpoint_path}")
            logger.warning("Using untrained models. Run training first for better results.")
            return

        try:
            # Find the latest checkpoint files
            semantic_ckpts = sorted(glob.glob(str(checkpoint_path / "semantic_encoder_epoch*.pt")))
            fusion_ckpts = sorted(glob.glob(str(checkpoint_path / "fusion_module_epoch*.pt")))
            static_ckpts = sorted(glob.glob(str(checkpoint_path / "static_encoder_epoch*.pt")))
            dynamic_ckpts = sorted(glob.glob(str(checkpoint_path / "dynamic_encoder_epoch*.pt")))

            # Load semantic encoder (best checkpoint)
            if semantic_ckpts:
                semantic_ckpt = semantic_ckpts[-1]  # Use the latest epoch
                logger.info(f"Loading semantic encoder from: {semantic_ckpt}")
                checkpoint = torch.load(semantic_ckpt, map_location=self.device)

                # Extract state dict from checkpoint (handles both formats)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                self.semantic_encoder.load_state_dict(state_dict)
                logger.info("✓ Semantic encoder loaded successfully")
            else:
                logger.warning("No semantic encoder checkpoint found")

            # Load fusion module (best checkpoint)
            if fusion_ckpts:
                fusion_ckpt = fusion_ckpts[-1]  # Use the latest epoch
                logger.info(f"Loading fusion module from: {fusion_ckpt}")
                checkpoint = torch.load(fusion_ckpt, map_location=self.device)

                # Extract state dict from checkpoint (handles both formats)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                self.fusion_module.load_state_dict(state_dict)
                logger.info("✓ Fusion module loaded successfully")
            else:
                logger.warning("No fusion module checkpoint found")

            # Load static encoder (best checkpoint)
            if static_ckpts:
                static_ckpt = static_ckpts[-1]  # Use the latest epoch
                logger.info(f"Loading static encoder from: {static_ckpt}")
                checkpoint = torch.load(static_ckpt, map_location=self.device)

                # Extract state dict from checkpoint (handles both formats)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                self.static_encoder.load_state_dict(state_dict)
                logger.info("✓ Static encoder loaded successfully")
            else:
                logger.warning("No static encoder checkpoint found")

            # Load dynamic encoder (best checkpoint)
            if dynamic_ckpts:
                dynamic_ckpt = dynamic_ckpts[-1]  # Use the latest epoch
                logger.info(f"Loading dynamic encoder from: {dynamic_ckpt}")
                checkpoint = torch.load(dynamic_ckpt, map_location=self.device)

                # Extract state dict from checkpoint (handles both formats)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint

                self.dynamic_encoder.load_state_dict(state_dict)
                logger.info("✓ Dynamic encoder loaded successfully")
            else:
                logger.warning("No dynamic encoder checkpoint found")

            logger.info("=" * 80)
            logger.info("All trained models loaded successfully!")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Error loading checkpoints: {e}")
            logger.warning("Continuing with untrained models")

    def analyze_contract(self, source_code: str, contract_name: str = None) -> dict:
        """Analyze a contract and return results"""
        result = self.orchestrator.analyze_contract(
            source_code=source_code,
            contract_name=contract_name
        )

        # Convert to simpler format for testing
        final_result = result['final_result']

        return {
            "vulnerabilities": [final_result.vulnerability_type] if final_result.vulnerability_detected else [],
            "confidence_scores": {final_result.vulnerability_type: final_result.confidence} if final_result.vulnerability_detected else {},
            "modality_contributions": final_result.modality_contributions,
            "reasoning": final_result.reasoning,
            "workflow_summary": result['workflow_summary']
        }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TritonTester:
    def __init__(
        self,
        triton_system: Optional[TritonSystem] = None,
        output_dir: str = "results"
    ):
        self.triton = triton_system or TritonSystem()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "system": "Triton v2.0"
            },
            "datasets": {},
            "overall_metrics": {}
        }

    def test_single_contract(
        self,
        contract_path: str,
        ground_truth: Optional[Dict] = None
    ) -> Dict:
        """Test Triton on a single contract"""

        try:
            # Read contract
            with open(contract_path, 'r') as f:
                source_code = f.read()

            # Analyze with Triton
            start_time = time.time()
            result = self.triton.analyze_contract(source_code)
            analysis_time = time.time() - start_time

            # Prepare result
            test_result = {
                "contract_path": str(contract_path),
                "analysis_time": analysis_time,
                "vulnerabilities_found": result.get("vulnerabilities", []),
                "confidence_scores": result.get("confidence_scores", {}),
                "modality_contributions": result.get("modality_contributions", {}),
                "success": True
            }

            # Compare with ground truth if available
            if ground_truth:
                test_result["ground_truth"] = ground_truth
                test_result["metrics"] = self._calculate_metrics(
                    result.get("vulnerabilities", []),
                    ground_truth.get("vulnerabilities", [])
                )

            return test_result

        except Exception as e:
            logger.error(f"Error analyzing {contract_path}: {e}")
            return {
                "contract_path": str(contract_path),
                "success": False,
                "error": str(e)
            }

    def test_smartbugs(self, dataset_dir: str = "data/datasets/smartbugs-curated/dataset"):
        """Test on SmartBugs Curated dataset"""
        logger.info("Testing on SmartBugs Curated dataset...")

        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            logger.error(f"SmartBugs Curated dataset not found at {dataset_path}")
            logger.error(f"Please download it first: git clone https://github.com/smartbugs/smartbugs-curated.git")
            return

        # SmartBugs structure: dataset/<vulnerability_type>/*.sol
        vuln_categories = [
            "access_control",
            "arithmetic",
            "bad_randomness",
            "denial_of_service",
            "front_running",
            "reentrancy",
            "short_addresses",
            "time_manipulation",
            "unchecked_low_level_calls",
            "other"
        ]

        results = {
            "dataset": "SmartBugs",
            "total_contracts": 0,
            "successful_analyses": 0,
            "total_time": 0,
            "by_vulnerability": {},
            "contracts": []
        }

        for vuln_type in vuln_categories:
            vuln_dir = dataset_path / vuln_type
            if not vuln_dir.exists():
                continue

            logger.info(f"Testing {vuln_type} contracts...")

            sol_files = list(vuln_dir.glob("*.sol"))
            results["by_vulnerability"][vuln_type] = {
                "total": len(sol_files),
                "detected": 0,
                "missed": 0,
                "false_positives": 0
            }

            for contract_file in sol_files:
                ground_truth = {
                    "vulnerabilities": [vuln_type],
                    "has_vulnerability": True
                }

                test_result = self.test_single_contract(
                    str(contract_file),
                    ground_truth
                )

                results["contracts"].append(test_result)
                results["total_contracts"] += 1

                if test_result["success"]:
                    results["successful_analyses"] += 1
                    results["total_time"] += test_result["analysis_time"]

                    # Check if vulnerability was detected
                    detected_vulns = test_result.get("vulnerabilities_found", [])
                    if any(vuln_type in str(v).lower() for v in detected_vulns):
                        results["by_vulnerability"][vuln_type]["detected"] += 1
                    else:
                        results["by_vulnerability"][vuln_type]["missed"] += 1

        # Calculate overall metrics
        results["metrics"] = self._calculate_overall_metrics(results)

        self.results["datasets"]["smartbugs"] = results

        logger.info(f"SmartBugs testing complete: {results['successful_analyses']}/{results['total_contracts']} contracts analyzed")

        return results

    def test_solidifi(self, dataset_dir: str = "data/datasets/solidifi/Benchmarks"):
        """Test on SolidiFI dataset"""
        logger.info("Testing on SolidiFI dataset...")

        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            logger.error(f"SolidiFI dataset not found at {dataset_path}")
            return

        results = {
            "dataset": "SolidiFI",
            "total_contracts": 0,
            "successful_analyses": 0,
            "total_time": 0,
            "contracts": []
        }

        # SolidiFI has buggy and patched versions
        sol_files = list(dataset_path.rglob("*.sol"))
        logger.info(f"Found {len(sol_files)} contracts")

        # Limit to first 100 for initial testing
        sol_files = sol_files[:100]

        for contract_file in sol_files:
            # Check if this is a buggy or patched version
            is_buggy = "buggy" in str(contract_file).lower()

            ground_truth = {
                "has_vulnerability": is_buggy,
                "vulnerabilities": [] if not is_buggy else ["injected"]
            }

            test_result = self.test_single_contract(
                str(contract_file),
                ground_truth
            )

            results["contracts"].append(test_result)
            results["total_contracts"] += 1

            if test_result["success"]:
                results["successful_analyses"] += 1
                results["total_time"] += test_result["analysis_time"]

        results["metrics"] = self._calculate_overall_metrics(results)

        self.results["datasets"]["solidifi"] = results

        logger.info(f"SolidiFI testing complete: {results['successful_analyses']}/{results['total_contracts']} contracts analyzed")

        return results

    def test_custom_contracts(self, contracts_dir: str):
        """Test on custom contracts"""
        logger.info(f"Testing on custom contracts from {contracts_dir}...")

        contracts_path = Path(contracts_dir)
        if not contracts_path.exists():
            logger.error(f"Custom contracts directory not found: {contracts_path}")
            return

        results = {
            "dataset": "Custom",
            "total_contracts": 0,
            "successful_analyses": 0,
            "total_time": 0,
            "contracts": []
        }

        sol_files = list(contracts_path.rglob("*.sol"))
        logger.info(f"Found {len(sol_files)} contracts")

        for contract_file in sol_files:
            test_result = self.test_single_contract(str(contract_file))

            results["contracts"].append(test_result)
            results["total_contracts"] += 1

            if test_result["success"]:
                results["successful_analyses"] += 1
                results["total_time"] += test_result["analysis_time"]

        self.results["datasets"]["custom"] = results

        logger.info(f"Custom contracts testing complete: {results['successful_analyses']}/{results['total_contracts']} contracts analyzed")

        return results

    def _calculate_metrics(
        self,
        predicted: List[str],
        ground_truth: List[str]
    ) -> Dict:
        """Calculate precision, recall, F1 for a single contract"""

        if not ground_truth:
            # No ground truth available
            return {}

        # Convert to sets for comparison
        pred_set = set(str(v).lower() for v in predicted)
        true_set = set(str(v).lower() for v in ground_truth)

        true_positives = len(pred_set & true_set)
        false_positives = len(pred_set - true_set)
        false_negatives = len(true_set - pred_set)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }

    def _calculate_overall_metrics(self, results: Dict) -> Dict:
        """Calculate overall metrics for a dataset"""

        all_metrics = [
            contract.get("metrics", {})
            for contract in results.get("contracts", [])
            if contract.get("metrics")
        ]

        if not all_metrics:
            return {}

        avg_precision = np.mean([m["precision"] for m in all_metrics])
        avg_recall = np.mean([m["recall"] for m in all_metrics])
        avg_f1 = np.mean([m["f1"] for m in all_metrics])

        avg_time = results["total_time"] / results["successful_analyses"] if results["successful_analyses"] > 0 else 0

        return {
            "average_precision": float(avg_precision),
            "average_recall": float(avg_recall),
            "average_f1": float(avg_f1),
            "average_analysis_time": float(avg_time),
            "throughput": results["successful_analyses"] / results["total_time"] if results["total_time"] > 0 else 0
        }

    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("Generating test report...")

        # Calculate overall metrics across all datasets
        all_metrics = []
        for dataset_name, dataset_results in self.results["datasets"].items():
            if "metrics" in dataset_results:
                all_metrics.append(dataset_results["metrics"])

        if all_metrics:
            self.results["overall_metrics"] = {
                "average_precision": float(np.mean([m.get("average_precision", 0) for m in all_metrics])),
                "average_recall": float(np.mean([m.get("average_recall", 0) for m in all_metrics])),
                "average_f1": float(np.mean([m.get("average_f1", 0) for m in all_metrics])),
                "average_analysis_time": float(np.mean([m.get("average_analysis_time", 0) for m in all_metrics])),
                "total_contracts_tested": sum(
                    dataset.get("total_contracts", 0)
                    for dataset in self.results["datasets"].values()
                )
            }

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"triton_test_results_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Detailed results saved to {results_file}")

        # Generate summary report
        summary_file = self.output_dir / f"triton_test_summary_{timestamp}.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("TRITON VULNERABILITY DETECTION - COMPREHENSIVE SUMMARY\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"Timestamp: {self.results['metadata']['timestamp']}\n")
            f.write(f"System: {self.results['metadata']['system']}\n\n")

            # Overall metrics
            if self.results["overall_metrics"]:
                f.write("OVERALL METRICS\n")
                f.write("-" * 100 + "\n")
                for key, value in self.results["overall_metrics"].items():
                    f.write(f"{key.replace('_', ' ').title()}: {value:.4f}\n")
                f.write("\n")

            # Dataset-specific results with detailed table
            for dataset_name, dataset_results in self.results["datasets"].items():
                f.write(f"\n{dataset_name.upper()} DATASET\n")
                f.write("=" * 100 + "\n")
                f.write(f"Total Contracts: {dataset_results.get('total_contracts', 0)}\n")
                f.write(f"Successful Analyses: {dataset_results.get('successful_analyses', 0)}\n")
                f.write(f"Total Time: {dataset_results.get('total_time', 0):.2f}s\n\n")

                if "metrics" in dataset_results:
                    f.write("Overall Dataset Metrics:\n")
                    f.write("-" * 100 + "\n")
                    for key, value in dataset_results["metrics"].items():
                        f.write(f"  {key.replace('_', ' ').title()}: {value:.4f}\n")
                    f.write("\n")

                # Enhanced vulnerability table
                if "by_vulnerability" in dataset_results:
                    f.write("VULNERABILITY DETECTION BREAKDOWN\n")
                    f.write("=" * 100 + "\n\n")

                    # Table header
                    f.write(f"{'Vulnerability Type':<30} | {'Total':<8} | {'Detected':<10} | {'Missed':<8} | {'Detection %':<12}\n")
                    f.write("-" * 100 + "\n")

                    total_vulns = 0
                    total_detected = 0

                    for vuln_type, stats in sorted(dataset_results["by_vulnerability"].items()):
                        total = stats.get("total", 0)
                        detected = stats.get("detected", 0)
                        missed = stats.get("missed", 0)

                        detection_rate = (detected / total * 100) if total > 0 else 0

                        total_vulns += total
                        total_detected += detected

                        vuln_display = vuln_type.replace('_', ' ').title()
                        f.write(f"{vuln_display:<30} | {total:<8} | {detected:<10} | {missed:<8} | {detection_rate:>10.2f}%\n")

                    f.write("-" * 100 + "\n")

                    overall_detection_rate = (total_detected / total_vulns * 100) if total_vulns > 0 else 0
                    f.write(f"{'TOTAL':<30} | {total_vulns:<8} | {total_detected:<10} | {total_vulns - total_detected:<8} | {overall_detection_rate:>10.2f}%\n")
                    f.write("=" * 100 + "\n\n")

        logger.info(f"Summary report saved to {summary_file}")

        # Generate markdown table for easy copy-paste
        markdown_file = self.output_dir / f"triton_results_table_{timestamp}.md"

        with open(markdown_file, 'w') as f:
            f.write("# Triton Vulnerability Detection Results\n\n")

            for dataset_name, dataset_results in self.results["datasets"].items():
                if "by_vulnerability" in dataset_results:
                    f.write(f"## {dataset_name.upper()} Dataset\n\n")

                    f.write("| Vulnerability Type | Total | Detected | Missed | Detection Rate |\n")
                    f.write("|-------------------|-------|----------|--------|----------------|\n")

                    total_vulns = 0
                    total_detected = 0

                    for vuln_type, stats in sorted(dataset_results["by_vulnerability"].items()):
                        total = stats.get("total", 0)
                        detected = stats.get("detected", 0)
                        missed = stats.get("missed", 0)
                        detection_rate = (detected / total * 100) if total > 0 else 0

                        total_vulns += total
                        total_detected += detected

                        vuln_display = vuln_type.replace('_', ' ').title()
                        f.write(f"| {vuln_display} | {total} | {detected} | {missed} | {detection_rate:.2f}% |\n")

                    overall_detection_rate = (total_detected / total_vulns * 100) if total_vulns > 0 else 0
                    f.write(f"| **TOTAL** | **{total_vulns}** | **{total_detected}** | **{total_vulns - total_detected}** | **{overall_detection_rate:.2f}%** |\n\n")

        logger.info(f"Markdown table saved to {markdown_file}")

        # Print summary to console with table
        print("\n" + "=" * 100)
        print("TRITON VULNERABILITY DETECTION - SUMMARY")
        print("=" * 100)

        if self.results["overall_metrics"]:
            print("\nOverall Metrics:")
            for key, value in self.results["overall_metrics"].items():
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

        # Print vulnerability table
        for dataset_name, dataset_results in self.results["datasets"].items():
            if "by_vulnerability" in dataset_results:
                print(f"\n{dataset_name.upper()} Dataset - Vulnerability Detection:")
                print("-" * 100)
                print(f"{'Vulnerability Type':<30} | {'Total':<8} | {'Detected':<10} | {'Missed':<8} | {'Detection %':<12}")
                print("-" * 100)

                total_vulns = 0
                total_detected = 0

                for vuln_type, stats in sorted(dataset_results["by_vulnerability"].items()):
                    total = stats.get("total", 0)
                    detected = stats.get("detected", 0)
                    missed = stats.get("missed", 0)
                    detection_rate = (detected / total * 100) if total > 0 else 0

                    total_vulns += total
                    total_detected += detected

                    vuln_display = vuln_type.replace('_', ' ').title()
                    print(f"{vuln_display:<30} | {total:<8} | {detected:<10} | {missed:<8} | {detection_rate:>10.2f}%")

                print("-" * 100)
                overall_detection_rate = (total_detected / total_vulns * 100) if total_vulns > 0 else 0
                print(f"{'TOTAL':<30} | {total_vulns:<8} | {total_detected:<10} | {total_vulns - total_detected:<8} | {overall_detection_rate:>10.2f}%")

        print("\n" + "=" * 100)
        print(f"\nResults saved to:")
        print(f"  - Detailed JSON: {results_file}")
        print(f"  - Text Summary: {summary_file}")
        print(f"  - Markdown Table: {markdown_file}")
        print("=" * 100 + "\n")

        return results_file, summary_file, markdown_file


def main():
    parser = argparse.ArgumentParser(description="Test Triton on benchmark datasets")
    parser.add_argument(
        "--dataset",
        choices=["all", "smartbugs", "solidifi", "custom"],
        default="smartbugs",
        help="Which dataset to test on"
    )
    parser.add_argument(
        "--custom-dir",
        help="Path to custom contracts directory"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = TritonTester(output_dir=args.output_dir)

    # Run tests
    if args.dataset == "all":
        tester.test_smartbugs()
        tester.test_solidifi()
    elif args.dataset == "smartbugs":
        tester.test_smartbugs()
    elif args.dataset == "solidifi":
        tester.test_solidifi()
    elif args.dataset == "custom":
        if not args.custom_dir:
            logger.error("Please provide --custom-dir for custom contracts")
            sys.exit(1)
        tester.test_custom_contracts(args.custom_dir)

    # Generate report
    tester.generate_report()


if __name__ == "__main__":
    main()
