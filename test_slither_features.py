#!/usr/bin/env python3
"""Test script to see what features Slither extracts from functions"""

from slither import Slither
import tempfile
import os

# Simple test contract
test_contract = '''
pragma solidity ^0.8.0;

contract TestContract {
    uint256 public balance;
    address public owner;

    constructor() {
        owner = msg.sender;
    }

    function withdraw(uint256 amount) external payable {
        require(msg.sender == owner, "Not owner");
        require(amount <= balance, "Insufficient balance");
        balance -= amount;
        payable(msg.sender).call{value: amount}("");
    }

    function deposit() external payable {
        balance += msg.value;
    }

    function getBalance() external view returns (uint256) {
        return balance;
    }
}
'''

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
    f.write(test_contract)
    temp_file = f.name

try:
    # Analyze with Slither
    slither = Slither(temp_file, solc_disable_warnings=True)

    for contract in slither.contracts:
        print(f"\n=== Contract: {contract.name} ===")

        for function in contract.functions:
            print(f"\n--- Function: {function.name} ---")
            print(f"  is_constructor: {function.is_constructor}")
            print(f"  is_fallback: {function.is_fallback}")
            print(f"  is_receive: {function.is_receive}")
            print(f"  visibility: {function.visibility}")
            print(f"  view: {function.view}")
            print(f"  pure: {function.pure}")
            print(f"  payable: {function.payable}")
            print(f"  can_reenter: {function.can_reenter()}")
            print(f"  can_send_eth: {function.can_send_eth()}")
            print(f"  has_assembly: {function.contains_assembly}")
            print(f"  state_variables_read: {[v.name for v in function.state_variables_read]}")
            print(f"  state_variables_written: {[v.name for v in function.state_variables_written]}")
            print(f"  internal_calls: {len(function.internal_calls)}")
            print(f"  external_calls: {len(function.external_calls_as_expressions)}")
            print(f"  low_level_calls: {len(function.low_level_calls)}")
            print(f"  solidity_calls: {len(function.solidity_calls)}")
            print(f"  modifiers: {[m.name for m in function.modifiers]}")

            # Check for loops
            has_loop = False
            for node in function.nodes:
                if node.type.name in ['STARTLOOP', 'IFLOOP']:
                    has_loop = True
                    break
            print(f"  has_loop: {has_loop}")

            # Count conditionals
            conditionals = sum(1 for node in function.nodes if node.type.name in ['IF', 'IFLOOP'])
            print(f"  conditionals: {conditionals}")

            # Check for require/assert
            has_require = any('require' in str(node.expression) for node in function.nodes if node.expression)
            has_assert = any('assert' in str(node.expression) for node in function.nodes if node.expression)
            print(f"  has_require: {has_require}")
            print(f"  has_assert: {has_assert}")

finally:
    if os.path.exists(temp_file):
        os.unlink(temp_file)
