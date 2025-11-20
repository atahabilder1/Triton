#!/usr/bin/env python3
"""
Diagnostic script to check what the models are actually predicting
"""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion

def load_models():
    """Load trained models"""
    device = 'cpu'

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

    fusion_module = CrossModalFusion(
        static_dim=768,
        dynamic_dim=512,
        semantic_dim=768,
        hidden_dim=512,
        output_dim=768,
        dropout=0.1
    ).to(device)

    # Load weights
    checkpoint_dir = Path("models/checkpoints")

    # Helper function to load checkpoint (handles both formats)
    def load_checkpoint(model, path):
        checkpoint = torch.load(path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    load_checkpoint(semantic_encoder, checkpoint_dir / "semantic_encoder_best.pt")
    load_checkpoint(fusion_module, checkpoint_dir / "fusion_module_best.pt")
    load_checkpoint(static_encoder, checkpoint_dir / "static_encoder_best.pt")
    load_checkpoint(dynamic_encoder, checkpoint_dir / "dynamic_encoder_best.pt")

    return static_encoder, dynamic_encoder, semantic_encoder, fusion_module, device

def test_predictions():
    """Test what models predict on different vulnerability types"""

    print("Loading models...")
    static_encoder, dynamic_encoder, semantic_encoder, fusion_module, device = load_models()

    static_encoder.eval()
    dynamic_encoder.eval()
    semantic_encoder.eval()
    fusion_module.eval()

    # Test contracts from different categories
    test_dir = Path("data/datasets/smartbugs-curated/dataset")

    vuln_types = [
        "access_control", "arithmetic", "reentrancy",
        "unchecked_low_level_calls", "bad_randomness"
    ]

    vuln_type_list = [
        'access_control', 'arithmetic', 'bad_randomness', 'denial_of_service',
        'front_running', 'reentrancy', 'short_addresses', 'time_manipulation',
        'unchecked_low_level_calls', 'other'
    ]

    print("\n" + "=" * 100)
    print("TESTING PREDICTIONS ON DIFFERENT VULNERABILITY TYPES")
    print("=" * 100 + "\n")

    for vuln_type in vuln_types:
        vuln_dir = test_dir / vuln_type
        if not vuln_dir.exists():
            continue

        contracts = list(vuln_dir.glob("*.sol"))[:2]  # Test 2 contracts per type

        for contract_file in contracts:
            with open(contract_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()

            with torch.no_grad():
                # Get features from all encoders
                semantic_features, vuln_scores = semantic_encoder([source_code], None)

                # Use dummy features for static/dynamic (since tools fail)
                static_tensor = torch.randn(1, 768) * 0.1
                dynamic_tensor = torch.randn(1, 512) * 0.1

                # Fuse features
                fusion_result = fusion_module(
                    static_tensor,
                    dynamic_tensor,
                    semantic_features,
                    None
                )

                # Get predictions
                vulnerability_logits = fusion_result['vulnerability_logits']
                predicted_class = torch.argmax(vulnerability_logits, dim=1).item()
                predicted_vulnerability = vuln_type_list[predicted_class] if predicted_class < len(vuln_type_list) else "other"

                # Get all logit values
                logits_np = vulnerability_logits[0].cpu().numpy()

                print(f"Contract: {contract_file.name:50s}")
                print(f"  Ground Truth: {vuln_type:30s}")
                print(f"  Predicted:    {predicted_vulnerability:30s} (class {predicted_class})")
                print(f"  Logits: {' '.join([f'{v:6.3f}' for v in logits_np])}")
                print(f"  Max logit: {logits_np.max():.3f}, Min logit: {logits_np.min():.3f}")

                # Check if all logits are similar (sign of model collapse)
                logit_std = logits_np.std()
                print(f"  Logit std dev: {logit_std:.4f} {'[COLLAPSED!]' if logit_std < 0.1 else ''}")
                print()

if __name__ == "__main__":
    test_predictions()
