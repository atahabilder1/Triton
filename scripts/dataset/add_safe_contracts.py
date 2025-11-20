#!/usr/bin/env python3
"""
Add Safe Contracts to Dataset
Adds verified safe contracts from OpenZeppelin and other trusted sources
"""

import sys
import requests
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# OpenZeppelin verified safe contracts (well-audited, no known vulnerabilities)
SAFE_CONTRACTS = [
    # ERC20 Standard (Safe)
    {
        'name': 'ERC20_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v4.9.0/contracts/token/ERC20/ERC20.sol',
    },
    # ERC721 Standard (Safe NFT)
    {
        'name': 'ERC721_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v4.9.0/contracts/token/ERC721/ERC721.sol',
    },
    # Ownable (Safe access control)
    {
        'name': 'Ownable_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v4.9.0/contracts/access/Ownable.sol',
    },
    # AccessControl (Safe role-based access)
    {
        'name': 'AccessControl_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v4.9.0/contracts/access/AccessControl.sol',
    },
    # ReentrancyGuard (Safe reentrancy protection)
    {
        'name': 'ReentrancyGuard_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v4.9.0/contracts/security/ReentrancyGuard.sol',
    },
    # Pausable (Safe pause mechanism)
    {
        'name': 'Pausable_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v4.9.0/contracts/security/Pausable.sol',
    },
    # SafeMath (Safe arithmetic)
    {
        'name': 'SafeMath_Safe.sol',
        'url': 'https://raw.githubusercontent.com/OpenZeppelin/openzeppelin-contracts/v3.4.0/contracts/math/SafeMath.sol',
    },
]

def download_contract(url: str, output_path: Path) -> bool:
    """Download a contract from URL"""
    try:
        logger.info(f"  Downloading from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        content = response.text

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"  ✓ Saved to {output_path.name}")
        return True

    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        return False


def create_simple_safe_contracts(output_dir: Path):
    """Create simple, verified-safe contracts manually"""

    contracts = {
        'SimpleToken_Safe.sol': '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleToken {
    string public name = "Safe Token";
    string public symbol = "SAFE";
    uint8 public decimals = 18;
    uint256 public totalSupply;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    constructor(uint256 _initialSupply) {
        totalSupply = _initialSupply * 10 ** uint256(decimals);
        balanceOf[msg.sender] = totalSupply;
    }

    function transfer(address _to, uint256 _value) public returns (bool success) {
        require(balanceOf[msg.sender] >= _value, "Insufficient balance");
        require(_to != address(0), "Invalid address");

        balanceOf[msg.sender] -= _value;
        balanceOf[_to] += _value;
        emit Transfer(msg.sender, _to, _value);
        return true;
    }

    function approve(address _spender, uint256 _value) public returns (bool success) {
        require(_spender != address(0), "Invalid address");
        allowance[msg.sender][_spender] = _value;
        emit Approval(msg.sender, _spender, _value);
        return true;
    }

    function transferFrom(address _from, address _to, uint256 _value) public returns (bool success) {
        require(_value <= balanceOf[_from], "Insufficient balance");
        require(_value <= allowance[_from][msg.sender], "Insufficient allowance");
        require(_to != address(0), "Invalid address");

        balanceOf[_from] -= _value;
        balanceOf[_to] += _value;
        allowance[_from][msg.sender] -= _value;
        emit Transfer(_from, _to, _value);
        return true;
    }
}
''',
        'SafeVault_Safe.sol': '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SafeVault {
    mapping(address => uint256) public balances;
    address public owner;
    bool public paused = false;

    event Deposit(address indexed user, uint256 amount);
    event Withdrawal(address indexed user, uint256 amount);

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "Contract is paused");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function deposit() public payable whenNotPaused {
        require(msg.value > 0, "Must deposit something");
        balances[msg.sender] += msg.value;
        emit Deposit(msg.sender, msg.value);
    }

    function withdraw(uint256 amount) public whenNotPaused {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount;
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        emit Withdrawal(msg.sender, amount);
    }

    function pause() public onlyOwner {
        paused = true;
    }

    function unpause() public onlyOwner {
        paused = false;
    }
}
''',
        'SimpleMultiSig_Safe.sol': '''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SimpleMultiSig {
    address[] public owners;
    uint256 public required;

    struct Transaction {
        address to;
        uint256 value;
        bytes data;
        bool executed;
        uint256 confirmations;
    }

    Transaction[] public transactions;
    mapping(uint256 => mapping(address => bool)) public isConfirmed;

    modifier onlyOwner() {
        bool isOwner = false;
        for (uint i = 0; i < owners.length; i++) {
            if (owners[i] == msg.sender) {
                isOwner = true;
                break;
            }
        }
        require(isOwner, "Not an owner");
        _;
    }

    constructor(address[] memory _owners, uint256 _required) {
        require(_owners.length > 0, "Owners required");
        require(_required > 0 && _required <= _owners.length, "Invalid required confirmations");

        owners = _owners;
        required = _required;
    }

    function submitTransaction(address _to, uint256 _value, bytes memory _data) public onlyOwner returns (uint256) {
        uint256 txId = transactions.length;
        transactions.push(Transaction({
            to: _to,
            value: _value,
            data: _data,
            executed: false,
            confirmations: 0
        }));
        return txId;
    }

    function confirmTransaction(uint256 _txId) public onlyOwner {
        require(_txId < transactions.length, "Transaction does not exist");
        require(!isConfirmed[_txId][msg.sender], "Already confirmed");
        require(!transactions[_txId].executed, "Already executed");

        isConfirmed[_txId][msg.sender] = true;
        transactions[_txId].confirmations += 1;
    }

    function executeTransaction(uint256 _txId) public onlyOwner {
        require(_txId < transactions.length, "Transaction does not exist");
        require(!transactions[_txId].executed, "Already executed");
        require(transactions[_txId].confirmations >= required, "Not enough confirmations");

        Transaction storage txn = transactions[_txId];
        txn.executed = true;
        (bool success, ) = txn.to.call{value: txn.value}(txn.data);
        require(success, "Transaction failed");
    }
}
'''
    }

    logger.info("\nCreating simple safe contracts...")
    for filename, content in contracts.items():
        output_path = output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"  ✓ Created {filename}")

    return len(contracts)


def main():
    logger.info("="*80)
    logger.info("ADDING SAFE CONTRACTS TO DATASET")
    logger.info("="*80)

    # Output directory
    output_dir = Path("data/datasets/safe_contracts_source")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nOutput directory: {output_dir}\n")

    # Download OpenZeppelin contracts
    logger.info("Downloading OpenZeppelin safe contracts...")
    downloaded = 0
    for contract in SAFE_CONTRACTS:
        output_path = output_dir / contract['name']
        if download_contract(contract['url'], output_path):
            downloaded += 1

    logger.info(f"\n✓ Downloaded {downloaded}/{len(SAFE_CONTRACTS)} OpenZeppelin contracts")

    # Create simple safe contracts
    created = create_simple_safe_contracts(output_dir)
    logger.info(f"✓ Created {created} simple safe contracts")

    logger.info(f"\n" + "="*80)
    logger.info(f"TOTAL: {downloaded + created} safe contracts")
    logger.info(f"Location: {output_dir}")
    logger.info("="*80)

    logger.info("\nNext: Run organize_by_class.py to add these to the dataset")


if __name__ == "__main__":
    main()
