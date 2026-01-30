"""
Advanced Multimodal Fusion Techniques for Smart Contract Vulnerability Detection

Based on state-of-the-art research including:
- VulnFusion: Interleaving-based fusion (89% F1 score)
- Tensor Fusion Networks: Higher-order feature interactions
- Gated Multimodal Units: Selective information flow
- Hierarchical Cross-Modal Attention: Multi-level feature alignment

References:
- VulnFusion (BCCA 2024): Interleaving fusion outperforms concatenation and cross-attention
- VulnSense (IJIS 2025): BERT + BiLSTM + GNN multimodal approach
- Multimodal Feature Fusion Framework (DLCV 2025): GAT-based weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class InterleavingFusion(nn.Module):
    """
    Interleaving-based fusion from VulnFusion paper.

    Instead of simple concatenation, this strategically interleaves features
    from different modalities, placing similar features together to capture
    intricate cross-modal relationships.

    Achieved 89% F1 score in VulnFusion experiments - best among all fusion methods.
    """

    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()

        # Project all modalities to same dimension for interleaving
        self.hidden_dim = 256  # Common dimension for interleaving

        self.static_proj = nn.Linear(static_dim, self.hidden_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, self.hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, self.hidden_dim)

        # Dense layers after interleaving (following VulnFusion architecture)
        interleaved_dim = self.hidden_dim * 3

        self.dense_blocks = nn.Sequential(
            nn.Linear(interleaved_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.ReLU()
        )

        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Interleave features from three modalities.

        Formula from VulnFusion paper:
        E[i,j] = A[i,k] if j = 3k
               = B[i,k] if j = 3k + 1
               = C[i,k] if j = 3k + 2
        """
        batch_size = static_features.size(0)

        # Project to common dimension
        static_proj = self.static_proj(static_features)    # [B, hidden_dim]
        dynamic_proj = self.dynamic_proj(dynamic_features)  # [B, hidden_dim]
        semantic_proj = self.semantic_proj(semantic_features)  # [B, hidden_dim]

        # Interleave features (key innovation from VulnFusion)
        # Instead of [static, dynamic, semantic], we do [s0,d0,sem0, s1,d1,sem1, ...]
        interleaved = torch.zeros(batch_size, self.hidden_dim * 3, device=static_features.device)

        for k in range(self.hidden_dim):
            interleaved[:, 3*k] = static_proj[:, k]      # j = 3k
            interleaved[:, 3*k + 1] = dynamic_proj[:, k]  # j = 3k + 1
            interleaved[:, 3*k + 2] = semantic_proj[:, k] # j = 3k + 2

        # Pass through dense blocks
        output = self.dense_blocks(interleaved)
        output = self.layer_norm(output)

        return output


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU) for selective information fusion.

    Uses learnable gates to control information flow from each modality,
    allowing the model to adaptively weight contributions based on input.
    """

    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project to common dimension
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        # Gating networks for each modality
        # Gate decides how much of each modality to use
        total_dim = hidden_dim * 3

        self.gate_static = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_dynamic = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.gate_semantic = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.Sigmoid()
        )

        # Transform gated features
        self.transform_static = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.transform_dynamic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        self.transform_semantic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated fusion.

        Returns:
            fused_features: Combined representation
            gate_values: Gate activations for interpretability
        """
        # Project to common dimension
        h_static = self.static_proj(static_features)
        h_dynamic = self.dynamic_proj(dynamic_features)
        h_semantic = self.semantic_proj(semantic_features)

        # Concatenate for gate computation
        concat = torch.cat([h_static, h_dynamic, h_semantic], dim=-1)

        # Compute gates
        g_static = self.gate_static(concat)
        g_dynamic = self.gate_dynamic(concat)
        g_semantic = self.gate_semantic(concat)

        # Apply gates and transform
        t_static = g_static * self.transform_static(h_static)
        t_dynamic = g_dynamic * self.transform_dynamic(h_dynamic)
        t_semantic = g_semantic * self.transform_semantic(h_semantic)

        # Combine gated features
        combined = torch.cat([t_static, t_dynamic, t_semantic], dim=-1)
        output = self.output_proj(combined)

        # Return gate values for interpretability
        gate_values = torch.stack([
            g_static.mean(dim=-1),
            g_dynamic.mean(dim=-1),
            g_semantic.mean(dim=-1)
        ], dim=-1)

        return output, gate_values


class TensorFusionNetwork(nn.Module):
    """
    Tensor Fusion Network for capturing higher-order modality interactions.

    Computes outer product of modality features to capture all possible
    uni-modal, bi-modal, and tri-modal interactions.

    Particularly effective for capturing complex vulnerability patterns
    that emerge from interactions between code structure, execution, and semantics.
    """

    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        hidden_dim: int = 64,  # Reduced for tensor product
        output_dim: int = 768,
        dropout: float = 0.1,
        rank: int = 8  # Low-rank approximation to reduce computation
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.rank = rank

        # Project to lower dimension for tractable tensor fusion
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        # Low-rank tensor decomposition factors
        # Instead of full tensor, use factorized form: W = sum_r (a_r ⊗ b_r ⊗ c_r)
        self.factor_static = nn.Parameter(torch.randn(rank, hidden_dim + 1))
        self.factor_dynamic = nn.Parameter(torch.randn(rank, hidden_dim + 1))
        self.factor_semantic = nn.Parameter(torch.randn(rank, hidden_dim + 1))

        nn.init.xavier_uniform_(self.factor_static)
        nn.init.xavier_uniform_(self.factor_dynamic)
        nn.init.xavier_uniform_(self.factor_semantic)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(rank, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute low-rank tensor fusion.

        Uses factorized tensor to capture tri-modal interactions efficiently.
        """
        batch_size = static_features.size(0)

        # Project and add bias term (append 1 for bias)
        h_static = self.static_proj(static_features)
        h_dynamic = self.dynamic_proj(dynamic_features)
        h_semantic = self.semantic_proj(semantic_features)

        # Add bias term (1) to enable uni-modal and bi-modal terms
        ones = torch.ones(batch_size, 1, device=static_features.device)
        h_static = torch.cat([h_static, ones], dim=-1)    # [B, hidden_dim+1]
        h_dynamic = torch.cat([h_dynamic, ones], dim=-1)
        h_semantic = torch.cat([h_semantic, ones], dim=-1)

        # Low-rank tensor fusion
        # For each rank r: compute element-wise product of projections
        fusion_output = []
        for r in range(self.rank):
            # Project each modality with rank-r factor
            proj_s = (h_static * self.factor_static[r]).sum(dim=-1, keepdim=True)
            proj_d = (h_dynamic * self.factor_dynamic[r]).sum(dim=-1, keepdim=True)
            proj_sem = (h_semantic * self.factor_semantic[r]).sum(dim=-1, keepdim=True)

            # Element-wise product captures interaction
            fusion_output.append(proj_s * proj_d * proj_sem)

        # Concatenate all rank outputs
        fusion = torch.cat(fusion_output, dim=-1)  # [B, rank]

        # Project to output dimension
        output = self.output_proj(fusion)

        return output


class HierarchicalCrossModalAttention(nn.Module):
    """
    Hierarchical Cross-Modal Attention with multiple attention levels.

    Level 1: Intra-modal self-attention (refine each modality)
    Level 2: Pair-wise cross-modal attention (static↔dynamic, dynamic↔semantic, etc.)
    Level 3: Global cross-modal attention (all modalities together)

    Based on insights from VulnSense and Multimodal Feature Fusion Framework papers.
    """

    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Project to common dimension
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        # Level 1: Intra-modal refinement (self-attention within each modality)
        self.intra_static = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.intra_dynamic = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.intra_semantic = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )

        # Level 2: Pair-wise cross-modal attention
        self.cross_static_dynamic = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_dynamic_semantic = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_semantic_static = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Level 3: Global cross-modal attention
        self.global_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply hierarchical cross-modal attention.

        Returns:
            fused_features: Final fused representation
            attention_weights: Dictionary of attention weights for interpretability
        """
        # Project to common dimension and add sequence dim
        h_static = self.static_proj(static_features).unsqueeze(1)    # [B, 1, H]
        h_dynamic = self.dynamic_proj(dynamic_features).unsqueeze(1)
        h_semantic = self.semantic_proj(semantic_features).unsqueeze(1)

        # Level 1: Intra-modal refinement
        h_static = self.intra_static(h_static)
        h_dynamic = self.intra_dynamic(h_dynamic)
        h_semantic = self.intra_semantic(h_semantic)

        # Level 2: Pair-wise cross-modal attention
        # Static attends to Dynamic
        cross_sd, attn_sd = self.cross_static_dynamic(h_static, h_dynamic, h_dynamic)
        h_static = self.norm1(h_static + cross_sd)

        # Dynamic attends to Semantic
        cross_ds, attn_ds = self.cross_dynamic_semantic(h_dynamic, h_semantic, h_semantic)
        h_dynamic = self.norm2(h_dynamic + cross_ds)

        # Semantic attends to Static
        cross_ss, attn_ss = self.cross_semantic_static(h_semantic, h_static, h_static)
        h_semantic = self.norm3(h_semantic + cross_ss)

        # Level 3: Global attention
        all_modalities = torch.cat([h_static, h_dynamic, h_semantic], dim=1)  # [B, 3, H]
        global_out, attn_global = self.global_attention(
            all_modalities, all_modalities, all_modalities
        )

        # Combine with residual
        combined = global_out + all_modalities

        # Flatten and project
        combined_flat = combined.flatten(1)  # [B, 3*H]
        output = self.output_proj(combined_flat)

        attention_weights = {
            'static_to_dynamic': attn_sd,
            'dynamic_to_semantic': attn_ds,
            'semantic_to_static': attn_ss,
            'global': attn_global
        }

        return output, attention_weights


class AdvancedCrossModalFusion(nn.Module):
    """
    Advanced Cross-Modal Fusion combining multiple fusion strategies.

    Fusion Methods:
    1. Interleaving (VulnFusion) - Best performing in benchmarks
    2. Gated Multimodal Units - Selective information flow
    3. Tensor Fusion - Higher-order interactions
    4. Hierarchical Attention - Multi-level feature alignment

    Uses ensemble of methods with learned combination weights.
    """

    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_vulnerability_types: int = 2,
        fusion_method: str = 'ensemble',  # 'interleaving', 'gated', 'tensor', 'hierarchical', 'ensemble'
        binary_mode: bool = True
    ):
        super().__init__()

        self.fusion_method = fusion_method
        self.output_dim = output_dim

        # Initialize fusion modules
        self.interleaving_fusion = InterleavingFusion(
            static_dim, dynamic_dim, semantic_dim, output_dim, dropout
        )

        self.gated_fusion = GatedMultimodalUnit(
            static_dim, dynamic_dim, semantic_dim, hidden_dim, output_dim, dropout
        )

        self.tensor_fusion = TensorFusionNetwork(
            static_dim, dynamic_dim, semantic_dim,
            hidden_dim=64, output_dim=output_dim, dropout=dropout
        )

        self.hierarchical_fusion = HierarchicalCrossModalAttention(
            static_dim, dynamic_dim, semantic_dim, hidden_dim, output_dim, num_heads, dropout
        )

        # Ensemble combination weights (learnable)
        if fusion_method == 'ensemble':
            self.ensemble_weights = nn.Parameter(torch.ones(4) / 4)
            self.ensemble_proj = nn.Sequential(
                nn.Linear(output_dim * 4, output_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim)
            )

        # Vulnerability classifier
        self.vulnerability_classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, num_vulnerability_types)
        )

        # Confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid()
        )

        # Modality importance predictor
        self.modality_importance = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim + semantic_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor,
        vulnerability_type: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply advanced fusion.

        Args:
            static_features: From static encoder (PDG analysis)
            dynamic_features: From dynamic encoder (execution traces)
            semantic_features: From semantic encoder (code understanding)
            vulnerability_type: Optional hint for vulnerability-specific fusion

        Returns:
            Dictionary with fused features, logits, confidence, and weights
        """

        if self.fusion_method == 'interleaving':
            fused = self.interleaving_fusion(static_features, dynamic_features, semantic_features)
            gate_values = None

        elif self.fusion_method == 'gated':
            fused, gate_values = self.gated_fusion(static_features, dynamic_features, semantic_features)

        elif self.fusion_method == 'tensor':
            fused = self.tensor_fusion(static_features, dynamic_features, semantic_features)
            gate_values = None

        elif self.fusion_method == 'hierarchical':
            fused, attn_weights = self.hierarchical_fusion(static_features, dynamic_features, semantic_features)
            gate_values = None

        elif self.fusion_method == 'ensemble':
            # Get outputs from all fusion methods
            f_interleave = self.interleaving_fusion(static_features, dynamic_features, semantic_features)
            f_gated, gate_values = self.gated_fusion(static_features, dynamic_features, semantic_features)
            f_tensor = self.tensor_fusion(static_features, dynamic_features, semantic_features)
            f_hierarchical, _ = self.hierarchical_fusion(static_features, dynamic_features, semantic_features)

            # Normalize ensemble weights
            weights = F.softmax(self.ensemble_weights, dim=0)

            # Weighted combination
            all_fused = torch.cat([f_interleave, f_gated, f_tensor, f_hierarchical], dim=-1)
            fused = self.ensemble_proj(all_fused)

        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Compute modality importance
        concat_features = torch.cat([static_features, dynamic_features, semantic_features], dim=-1)
        modality_weights = self.modality_importance(concat_features)

        # Classification and confidence
        vulnerability_logits = self.vulnerability_classifier(fused)
        confidence_scores = self.confidence_estimator(fused)

        return {
            'fused_features': fused,
            'vulnerability_logits': vulnerability_logits,
            'confidence_scores': confidence_scores,
            'modality_weights': modality_weights,
            'gate_values': gate_values if gate_values is not None else modality_weights,
            'fusion_method': self.fusion_method
        }


def get_fusion_module(
    fusion_type: str = 'advanced',
    static_dim: int = 768,
    dynamic_dim: int = 512,
    semantic_dim: int = 768,
    output_dim: int = 768,
    num_vulnerability_types: int = 2,
    **kwargs
) -> nn.Module:
    """
    Factory function to get appropriate fusion module.

    Args:
        fusion_type: 'basic', 'interleaving', 'gated', 'tensor', 'hierarchical', 'ensemble', 'advanced'

    Returns:
        Fusion module
    """
    if fusion_type == 'basic':
        from fusion.cross_modal_fusion import CrossModalFusion
        return CrossModalFusion(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            semantic_dim=semantic_dim,
            output_dim=output_dim,
            num_vulnerability_types=num_vulnerability_types,
            **kwargs
        )

    fusion_method = fusion_type if fusion_type in ['interleaving', 'gated', 'tensor', 'hierarchical', 'ensemble'] else 'ensemble'

    return AdvancedCrossModalFusion(
        static_dim=static_dim,
        dynamic_dim=dynamic_dim,
        semantic_dim=semantic_dim,
        output_dim=output_dim,
        num_vulnerability_types=num_vulnerability_types,
        fusion_method=fusion_method,
        **kwargs
    )
