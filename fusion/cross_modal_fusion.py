import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class CrossModalAttention(nn.Module):
    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super(CrossModalAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, hidden_dim)
        self.semantic_proj = nn.Linear(semantic_dim, hidden_dim)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.static_to_others = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.dynamic_to_others = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)
        self.semantic_to_others = nn.MultiheadAttention(hidden_dim, num_heads, dropout, batch_first=True)

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        static_proj = self.static_proj(static_features)
        dynamic_proj = self.dynamic_proj(dynamic_features)
        semantic_proj = self.semantic_proj(semantic_features)

        batch_size = static_proj.size(0)

        static_proj = static_proj.unsqueeze(1)
        dynamic_proj = dynamic_proj.unsqueeze(1)
        semantic_proj = semantic_proj.unsqueeze(1)

        all_features = torch.cat([static_proj, dynamic_proj, semantic_proj], dim=1)

        static_attended, _ = self.static_to_others(static_proj, all_features, all_features)
        dynamic_attended, _ = self.dynamic_to_others(dynamic_proj, all_features, all_features)
        semantic_attended, _ = self.semantic_to_others(semantic_proj, all_features, all_features)

        static_enhanced = self.layer_norm(static_proj + self.dropout(static_attended))
        dynamic_enhanced = self.layer_norm(dynamic_proj + self.dropout(dynamic_attended))
        semantic_enhanced = self.layer_norm(semantic_proj + self.dropout(semantic_attended))

        return (
            static_enhanced.squeeze(1),
            dynamic_enhanced.squeeze(1),
            semantic_enhanced.squeeze(1)
        )


class AdaptiveModalityWeighting(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        num_vulnerability_types: int = 10,
        temperature: float = 1.0
    ):
        super(AdaptiveModalityWeighting, self).__init__()

        self.temperature = temperature

        self.vulnerability_embeddings = nn.Embedding(num_vulnerability_types, 64)

        self.weight_predictor = nn.Sequential(
            nn.Linear(input_dim * 3 + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.vulnerability_type_mapping = {
            'reentrancy': 0,
            'overflow': 1,
            'underflow': 2,
            'access_control': 3,
            'unchecked_call': 4,
            'timestamp_dependency': 5,
            'tx_origin': 6,
            'delegatecall': 7,
            'self_destruct': 8,
            'gas_limit': 9
        }

        self.default_weights = {
            'reentrancy': torch.tensor([0.2, 0.6, 0.2]),
            'access_control': torch.tensor([0.7, 0.1, 0.2]),
            'overflow': torch.tensor([0.3, 0.4, 0.3]),
            'timestamp_dependency': torch.tensor([0.4, 0.2, 0.4]),
            'delegatecall': torch.tensor([0.3, 0.5, 0.2]),
            'default': torch.tensor([0.33, 0.33, 0.34])
        }

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor,
        vulnerability_type: Optional[str] = None
    ) -> torch.Tensor:

        combined = torch.cat([static_features, dynamic_features, semantic_features], dim=-1)

        if vulnerability_type and vulnerability_type in self.vulnerability_type_mapping:
            vuln_id = self.vulnerability_type_mapping[vulnerability_type]
            vuln_embedding = self.vulnerability_embeddings(
                torch.tensor([vuln_id], device=combined.device).expand(combined.size(0))
            )
            combined = torch.cat([combined, vuln_embedding], dim=-1)

            weights = self.weight_predictor(combined)
            weights = F.softmax(weights / self.temperature, dim=-1)
        else:
            if vulnerability_type and vulnerability_type in self.default_weights:
                default_weight = self.default_weights[vulnerability_type]
            else:
                default_weight = self.default_weights['default']

            weights = default_weight.to(combined.device).unsqueeze(0).expand(combined.size(0), -1)

        return weights


class ContextAwareFusion(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 768,
        fusion_layers: int = 3,
        dropout: float = 0.1
    ):
        super(ContextAwareFusion, self).__init__()

        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else output_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.LayerNorm(output_dim)
            )
            for i in range(fusion_layers)
        ])

        self.gate = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.Sigmoid()
        )

        self.residual_projection = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        x = fused_features

        for i, layer in enumerate(self.fusion_layers):
            if i == 0:
                residual = self.residual_projection(x)
            else:
                residual = x

            x = layer(x)

            gate_values = self.gate(x)
            x = gate_values * x + (1 - gate_values) * residual

        return x


class CrossModalFusion(nn.Module):
    def __init__(
        self,
        static_dim: int = 768,
        dynamic_dim: int = 512,
        semantic_dim: int = 768,
        hidden_dim: int = 512,
        output_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_vulnerability_types: int = 10
    ):
        super(CrossModalFusion, self).__init__()

        self.cross_attention = CrossModalAttention(
            static_dim, dynamic_dim, semantic_dim, hidden_dim, num_heads, dropout
        )

        self.adaptive_weighting = AdaptiveModalityWeighting(
            hidden_dim, num_vulnerability_types
        )

        self.context_fusion = ContextAwareFusion(
            hidden_dim, output_dim, fusion_layers=3, dropout=dropout
        )

        self.final_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

        self.vulnerability_classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, num_vulnerability_types)
        )

        self.confidence_estimator = nn.Sequential(
            nn.Linear(output_dim, output_dim // 4),
            nn.ReLU(),
            nn.Linear(output_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        semantic_features: torch.Tensor,
        vulnerability_type: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:

        static_enhanced, dynamic_enhanced, semantic_enhanced = self.cross_attention(
            static_features, dynamic_features, semantic_features
        )

        modality_weights = self.adaptive_weighting(
            static_enhanced, dynamic_enhanced, semantic_enhanced, vulnerability_type
        )

        weighted_features = (
            modality_weights[:, 0:1] * static_enhanced +
            modality_weights[:, 1:2] * dynamic_enhanced +
            modality_weights[:, 2:3] * semantic_enhanced
        )

        fused_features = self.context_fusion(weighted_features)

        final_features = self.final_projection(fused_features)

        vulnerability_logits = self.vulnerability_classifier(final_features)
        confidence_scores = self.confidence_estimator(final_features)

        return {
            'fused_features': final_features,
            'vulnerability_logits': vulnerability_logits,
            'confidence_scores': confidence_scores,
            'modality_weights': modality_weights,
            'individual_features': {
                'static': static_enhanced,
                'dynamic': dynamic_enhanced,
                'semantic': semantic_enhanced
            }
        }


class MultiScaleFusion(nn.Module):
    def __init__(
        self,
        input_dims: Dict[str, int],
        output_dim: int = 768,
        scales: List[int] = [1, 2, 4],
        dropout: float = 0.1
    ):
        super(MultiScaleFusion, self).__init__()

        self.scales = scales
        self.projections = nn.ModuleDict()

        for modality, dim in input_dims.items():
            self.projections[modality] = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim, output_dim // scale),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
                for scale in scales
            ])

        self.scale_attention = nn.ModuleList([
            nn.MultiheadAttention(
                output_dim // scale,
                num_heads=max(1, (output_dim // scale) // 64),
                dropout=dropout,
                batch_first=True
            )
            for scale in scales
        ])

        self.fusion_projection = nn.Linear(
            sum(output_dim // scale for scale in scales) * len(input_dims),
            output_dim
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        scale_features = []

        for scale_idx, scale in enumerate(self.scales):
            modality_features = []

            for modality, feature in features.items():
                projected = self.projections[modality][scale_idx](feature)
                modality_features.append(projected.unsqueeze(1))

            if len(modality_features) > 1:
                stacked = torch.cat(modality_features, dim=1)
                attended, _ = self.scale_attention[scale_idx](stacked, stacked, stacked)
                scale_features.append(attended.flatten(1))
            else:
                scale_features.append(modality_features[0].squeeze(1))

        all_scales = torch.cat(scale_features, dim=-1)
        fused = self.fusion_projection(all_scales)

        return fused


def compute_modality_importance(
    static_scores: torch.Tensor,
    dynamic_scores: torch.Tensor,
    semantic_scores: torch.Tensor,
    vulnerability_type: str
) -> Dict[str, float]:

    importance_maps = {
        'reentrancy': {'static': 0.2, 'dynamic': 0.6, 'semantic': 0.2},
        'access_control': {'static': 0.7, 'dynamic': 0.1, 'semantic': 0.2},
        'overflow': {'static': 0.3, 'dynamic': 0.4, 'semantic': 0.3},
        'timestamp_dependency': {'static': 0.4, 'dynamic': 0.2, 'semantic': 0.4},
        'delegatecall': {'static': 0.3, 'dynamic': 0.5, 'semantic': 0.2},
        'default': {'static': 0.33, 'dynamic': 0.33, 'semantic': 0.34}
    }

    base_weights = importance_maps.get(vulnerability_type, importance_maps['default'])

    score_variances = {
        'static': torch.var(static_scores).item(),
        'dynamic': torch.var(dynamic_scores).item(),
        'semantic': torch.var(semantic_scores).item()
    }

    total_variance = sum(score_variances.values())
    if total_variance > 0:
        uncertainty_weights = {
            k: v / total_variance for k, v in score_variances.items()
        }
    else:
        uncertainty_weights = {'static': 0.33, 'dynamic': 0.33, 'semantic': 0.34}

    alpha = 0.7
    final_weights = {}
    for modality in ['static', 'dynamic', 'semantic']:
        final_weights[modality] = (
            alpha * base_weights[modality] +
            (1 - alpha) * uncertainty_weights[modality]
        )

    total = sum(final_weights.values())
    final_weights = {k: v / total for k, v in final_weights.items()}

    return final_weights