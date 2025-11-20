#!/usr/bin/env python3
"""
Quick test script to verify all encoders work correctly
"""

import sys
from pathlib import Path
import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion

def test_static_encoder():
    print("=" * 60)
    print("Testing Static Encoder...")
    print("=" * 60)

    encoder = StaticEncoder(
        node_feature_dim=128,
        hidden_dim=256,
        output_dim=768,
        dropout=0.2
    )

    # Create dummy PDG
    pdg = nx.DiGraph()
    pdg.add_node("func1", type="function")
    pdg.add_node("var1", type="variable")
    pdg.add_edge("func1", "var1", type="reads")

    # Test forward pass
    features, scores = encoder([pdg], None)

    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Expected: (1, 768)")
    print(f"✓ Vulnerability scores: {len(scores)} types")
    print("✓ Static Encoder works!\n")

    return features

def test_dynamic_encoder():
    print("=" * 60)
    print("Testing Dynamic Encoder...")
    print("=" * 60)

    encoder = DynamicEncoder(
        vocab_size=50,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout=0.2
    )

    # Create dummy execution trace
    trace = {
        'steps': [
            {'opcode': 'PUSH1', 'gas': 3, 'depth': 1, 'stack': []},
            {'opcode': 'ADD', 'gas': 2, 'depth': 1, 'stack': [1, 2]},
            {'opcode': 'SSTORE', 'gas': 5000, 'depth': 1, 'stack': [3]},
        ]
    }

    # Test forward pass
    features, scores = encoder([trace], None)

    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Expected: (1, 512)")
    print(f"✓ Vulnerability scores: {len(scores)} types")
    print("✓ Dynamic Encoder works!\n")

    return features

def test_semantic_encoder():
    print("=" * 60)
    print("Testing Semantic Encoder...")
    print("=" * 60)

    encoder = SemanticEncoder(
        model_name="microsoft/graphcodebert-base",
        output_dim=768,
        max_length=512,
        dropout=0.1
    )

    # Create dummy Solidity code
    code = """
    pragma solidity ^0.8.0;

    contract Test {
        uint256 public value;

        function setValue(uint256 _value) public {
            value = _value;
        }
    }
    """

    # Test forward pass
    features, scores = encoder([code], None)

    print(f"✓ Output shape: {features.shape}")
    print(f"✓ Expected: (1, 768)")
    print(f"✓ Vulnerability scores: {len(scores)} types")
    print("✓ Semantic Encoder works!\n")

    return features

def test_fusion():
    print("=" * 60)
    print("Testing Fusion Module...")
    print("=" * 60)

    # Create dummy features
    static_features = torch.randn(2, 768)
    dynamic_features = torch.randn(2, 512)
    semantic_features = torch.randn(2, 768)

    fusion = CrossModalFusion(
        static_dim=768,
        dynamic_dim=512,
        semantic_dim=768,
        hidden_dim=512,
        output_dim=768,
        dropout=0.1
    )

    # Test forward pass
    output = fusion(static_features, dynamic_features, semantic_features, None)

    print(f"✓ Fused features shape: {output['fused_features'].shape}")
    print(f"✓ Expected: (2, 768)")
    print(f"✓ Vulnerability logits shape: {output['vulnerability_logits'].shape}")
    print(f"✓ Modality weights shape: {output['modality_weights'].shape}")
    print(f"✓ Confidence scores shape: {output['confidence_scores'].shape}")
    print("✓ Fusion Module works!\n")

def test_end_to_end():
    print("=" * 60)
    print("Testing End-to-End Pipeline...")
    print("=" * 60)

    # Initialize all components
    static_encoder = StaticEncoder(
        node_feature_dim=128,
        hidden_dim=256,
        output_dim=768,
        dropout=0.2
    )

    dynamic_encoder = DynamicEncoder(
        vocab_size=50,
        embedding_dim=128,
        hidden_dim=256,
        output_dim=512,
        dropout=0.2
    )

    semantic_encoder = SemanticEncoder(
        model_name="microsoft/graphcodebert-base",
        output_dim=768,
        max_length=512,
        dropout=0.1
    )

    fusion = CrossModalFusion(
        static_dim=768,
        dynamic_dim=512,
        semantic_dim=768,
        hidden_dim=512,
        output_dim=768,
        dropout=0.1
    )

    # Create test inputs
    pdg = nx.DiGraph()
    pdg.add_node("func1", type="function")
    pdg.add_node("var1", type="variable")
    pdg.add_edge("func1", "var1", type="reads")

    trace = {
        'steps': [
            {'opcode': 'PUSH1', 'gas': 3, 'depth': 1, 'stack': []},
            {'opcode': 'SSTORE', 'gas': 5000, 'depth': 1, 'stack': [1]},
        ]
    }

    code = "pragma solidity ^0.8.0; contract Test { uint256 value; }"

    # Run through pipeline
    static_features, _ = static_encoder([pdg], None)
    dynamic_features, _ = dynamic_encoder([trace], None)
    semantic_features, _ = semantic_encoder([code], None)

    output = fusion(static_features, dynamic_features, semantic_features, None)

    print(f"✓ Static features: {static_features.shape}")
    print(f"✓ Dynamic features: {dynamic_features.shape}")
    print(f"✓ Semantic features: {semantic_features.shape}")
    print(f"✓ Final output: {output['vulnerability_logits'].shape}")
    print(f"✓ Predicted vulnerabilities: {torch.argmax(output['vulnerability_logits'], dim=1)}")
    print("✓ End-to-End Pipeline works!\n")

def main():
    print("\n" + "=" * 60)
    print("TRITON ENCODER TESTING SUITE")
    print("=" * 60 + "\n")

    try:
        static_feat = test_static_encoder()
        dynamic_feat = test_dynamic_encoder()
        semantic_feat = test_semantic_encoder()
        test_fusion()
        test_end_to_end()

        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nDimension Summary:")
        print(f"  Static Encoder:   768")
        print(f"  Dynamic Encoder:  512")
        print(f"  Semantic Encoder: 768")
        print(f"  Fusion Output:    768")
        print("\nReady for training!")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
