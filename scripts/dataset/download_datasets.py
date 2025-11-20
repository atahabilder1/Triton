#!/usr/bin/env python3
"""
Dataset Download Script for Triton
Downloads and prepares benchmark datasets for smart contract vulnerability detection.

Datasets:
1. SmartBugs: 143 vulnerable contracts (9 vulnerability types)
2. SolidiFI: 9,369 contracts with injected vulnerabilities
3. SmartBugs Wild: Large-scale real-world contracts
4. Real-world audits from GitHub
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self, base_dir: str = "data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_smartbugs(self):
        """Download SmartBugs dataset (143 vulnerable contracts)"""
        logger.info("Downloading SmartBugs dataset...")

        smartbugs_dir = self.base_dir / "smartbugs"
        smartbugs_dir.mkdir(exist_ok=True)

        # Clone SmartBugs repository
        repo_url = "https://github.com/smartbugs/smartbugs.git"

        if (smartbugs_dir / "dataset").exists():
            logger.info("SmartBugs already downloaded. Skipping...")
            return

        try:
            subprocess.run(
                ["git", "clone", repo_url, str(smartbugs_dir)],
                check=True,
                capture_output=True
            )
            logger.info(f"SmartBugs downloaded to {smartbugs_dir}")

            # The actual vulnerable contracts are in dataset/
            contracts_path = smartbugs_dir / "dataset"
            if contracts_path.exists():
                logger.info(f"Found {len(list(contracts_path.rglob('*.sol')))} Solidity contracts")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download SmartBugs: {e}")

    def download_solidifi(self):
        """Download SolidiFI dataset (9,369 contracts with injected vulnerabilities)"""
        logger.info("Downloading SolidiFI dataset...")

        solidifi_dir = self.base_dir / "solidifi"
        solidifi_dir.mkdir(exist_ok=True)

        repo_url = "https://github.com/DependableSystemsLab/SolidiFI.git"

        if (solidifi_dir / "Benchmarks").exists():
            logger.info("SolidiFI already downloaded. Skipping...")
            return

        try:
            subprocess.run(
                ["git", "clone", repo_url, str(solidifi_dir)],
                check=True,
                capture_output=True
            )
            logger.info(f"SolidiFI downloaded to {solidifi_dir}")

            # SolidiFI has different vulnerability categories
            benchmarks_path = solidifi_dir / "Benchmarks"
            if benchmarks_path.exists():
                logger.info(f"Found SolidiFI benchmark contracts")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download SolidiFI: {e}")

    def download_smartbugs_wild(self):
        """Download SmartBugs Wild dataset (47,398 real-world contracts)"""
        logger.info("Downloading SmartBugs Wild dataset...")

        wild_dir = self.base_dir / "smartbugs_wild"
        wild_dir.mkdir(exist_ok=True)

        repo_url = "https://github.com/smartbugs/smartbugs-wild.git"

        if (wild_dir / "contracts").exists():
            logger.info("SmartBugs Wild already downloaded. Skipping...")
            return

        try:
            subprocess.run(
                ["git", "clone", repo_url, str(wild_dir)],
                check=True,
                capture_output=True
            )
            logger.info(f"SmartBugs Wild downloaded to {wild_dir}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download SmartBugs Wild: {e}")

    def download_audit_datasets(self):
        """Download real-world audit datasets"""
        logger.info("Downloading audit datasets...")

        audit_dir = self.base_dir / "audits"
        audit_dir.mkdir(exist_ok=True)

        # Not-So-Smart-Contracts (Trail of Bits)
        logger.info("Downloading Not-So-Smart-Contracts (Trail of Bits)...")
        nssc_dir = audit_dir / "not_so_smart_contracts"
        repo_url = "https://github.com/crytic/not-so-smart-contracts.git"

        if not nssc_dir.exists():
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(nssc_dir)],
                    check=True,
                    capture_output=True
                )
                logger.info(f"Not-So-Smart-Contracts downloaded to {nssc_dir}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download Not-So-Smart-Contracts: {e}")

        # DeFi vulnerabilities
        logger.info("Downloading DeFi vulnerability examples...")
        defi_dir = audit_dir / "defi_vulnerabilities"
        repo_url = "https://github.com/sirhashalot/SCV-List.git"

        if not defi_dir.exists():
            try:
                subprocess.run(
                    ["git", "clone", repo_url, str(defi_dir)],
                    check=True,
                    capture_output=True
                )
                logger.info(f"DeFi vulnerabilities downloaded to {defi_dir}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download DeFi vulnerabilities: {e}")

    def download_securify_dataset(self):
        """Download Securify dataset"""
        logger.info("Downloading Securify dataset...")

        securify_dir = self.base_dir / "securify"
        securify_dir.mkdir(exist_ok=True)

        repo_url = "https://github.com/eth-sri/securify2.git"

        if (securify_dir / "tests").exists():
            logger.info("Securify already downloaded. Skipping...")
            return

        try:
            subprocess.run(
                ["git", "clone", repo_url, str(securify_dir)],
                check=True,
                capture_output=True
            )
            logger.info(f"Securify downloaded to {securify_dir}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download Securify: {e}")

    def create_dataset_summary(self):
        """Create a summary of all downloaded datasets"""
        logger.info("Creating dataset summary...")

        summary = {
            "datasets": [],
            "total_contracts": 0,
            "vulnerability_types": []
        }

        # SmartBugs
        smartbugs_dir = self.base_dir / "smartbugs" / "dataset"
        if smartbugs_dir.exists():
            sol_files = list(smartbugs_dir.rglob("*.sol"))
            summary["datasets"].append({
                "name": "SmartBugs",
                "path": str(smartbugs_dir),
                "contracts": len(sol_files),
                "description": "143 vulnerable contracts (9 vulnerability types)"
            })
            summary["total_contracts"] += len(sol_files)

        # SolidiFI
        solidifi_dir = self.base_dir / "solidifi" / "Benchmarks"
        if solidifi_dir.exists():
            sol_files = list(solidifi_dir.rglob("*.sol"))
            summary["datasets"].append({
                "name": "SolidiFI",
                "path": str(solidifi_dir),
                "contracts": len(sol_files),
                "description": "9,369 contracts with injected vulnerabilities"
            })
            summary["total_contracts"] += len(sol_files)

        # SmartBugs Wild
        wild_dir = self.base_dir / "smartbugs_wild" / "contracts"
        if wild_dir.exists():
            sol_files = list(wild_dir.rglob("*.sol"))
            summary["datasets"].append({
                "name": "SmartBugs Wild",
                "path": str(wild_dir),
                "contracts": len(sol_files),
                "description": "47,398 real-world contracts"
            })
            summary["total_contracts"] += len(sol_files)

        # Save summary
        summary_path = self.base_dir / "dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Dataset summary saved to {summary_path}")
        logger.info(f"Total contracts: {summary['total_contracts']}")

        return summary

    def download_all(self):
        """Download all datasets"""
        logger.info("Starting download of all datasets...")

        self.download_smartbugs()
        self.download_solidifi()
        self.download_smartbugs_wild()
        self.download_audit_datasets()
        self.download_securify_dataset()

        summary = self.create_dataset_summary()

        logger.info("All datasets downloaded successfully!")
        logger.info(f"Total datasets: {len(summary['datasets'])}")
        logger.info(f"Total contracts: {summary['total_contracts']}")


def main():
    parser = argparse.ArgumentParser(description="Download benchmark datasets for Triton")
    parser.add_argument(
        "--dataset",
        choices=["all", "smartbugs", "solidifi", "wild", "audits", "securify"],
        default="all",
        help="Which dataset to download"
    )
    parser.add_argument(
        "--base-dir",
        default="data/datasets",
        help="Base directory for datasets"
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(args.base_dir)

    if args.dataset == "all":
        downloader.download_all()
    elif args.dataset == "smartbugs":
        downloader.download_smartbugs()
    elif args.dataset == "solidifi":
        downloader.download_solidifi()
    elif args.dataset == "wild":
        downloader.download_smartbugs_wild()
    elif args.dataset == "audits":
        downloader.download_audit_datasets()
    elif args.dataset == "securify":
        downloader.download_securify_dataset()

    downloader.create_dataset_summary()


if __name__ == "__main__":
    main()
