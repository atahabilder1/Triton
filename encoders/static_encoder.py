import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import networkx as nx
from typing import Dict, List, Optional, Tuple
import numpy as np


class EdgeAwareGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 8,
        dropout: float = 0.2
    ):
        super(EdgeAwareGAT, self).__init__()

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=4
        )

        self.gat2 = GATConv(
            hidden_channels * num_heads,
            hidden_channels,
            heads=num_heads,
            dropout=dropout,
            edge_dim=4
        )

        self.gat3 = GATConv(
            hidden_channels * num_heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=4
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat2(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.dropout(x)

        x = self.gat3(x, edge_index, edge_attr)

        if batch is not None:
            x = global_mean_pool(x, batch)

        return x


class StaticEncoder(nn.Module):
    def __init__(
        self,
        node_feature_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 768,
        num_heads: int = 8,
        num_vulnerability_types: int = 10,
        dropout: float = 0.2
    ):
        super(StaticEncoder, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(5, node_feature_dim // 2),
            nn.ReLU(),
            nn.Linear(node_feature_dim // 2, node_feature_dim),
            nn.ReLU()
        )

        self.gat = EdgeAwareGAT(
            node_feature_dim,
            hidden_dim,
            hidden_dim,
            num_heads,
            dropout
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, output_dim)
        )

        self.vulnerability_heads = nn.ModuleDict({
            'reentrancy': nn.Linear(output_dim, 1),
            'overflow': nn.Linear(output_dim, 1),
            'access_control': nn.Linear(output_dim, 1),
            'timestamp': nn.Linear(output_dim, 1),
            'general': nn.Linear(output_dim, 1)
        })

    def pdg_to_geometric(self, pdg: nx.DiGraph) -> Data:
        if pdg.number_of_nodes() == 0:
            x = torch.zeros((1, 5), dtype=torch.float32)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 4), dtype=torch.float32)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        node_to_idx = {node: idx for idx, node in enumerate(pdg.nodes())}

        node_features = []
        for node in pdg.nodes():
            node_data = pdg.nodes[node]
            node_type = node_data.get('type', 'unknown')

            type_encoding = [0, 0, 0]
            if node_type == 'function':
                type_encoding = [1, 0, 0]
            elif node_type == 'variable':
                type_encoding = [0, 1, 0]
            elif node_type == 'modifier':
                type_encoding = [0, 0, 1]

            in_degree = pdg.in_degree(node)
            out_degree = pdg.out_degree(node)

            features = type_encoding + [in_degree / 10.0, out_degree / 10.0]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float32)

        edge_list = []
        edge_features = []
        edge_type_map = {'calls': 0, 'reads': 1, 'writes': 2, 'uses_modifier': 3}

        for source, target, data in pdg.edges(data=True):
            edge_list.append([node_to_idx[source], node_to_idx[target]])

            edge_type = data.get('type', 'unknown')
            edge_type_idx = edge_type_map.get(edge_type, 0)
            edge_encoding = [0, 0, 0, 0]
            edge_encoding[edge_type_idx] = 1
            edge_features.append(edge_encoding)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward(
        self,
        pdgs: List[nx.DiGraph],
        vulnerability_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        geometric_data = [self.pdg_to_geometric(pdg) for pdg in pdgs]
        batch = Batch.from_data_list(geometric_data)

        node_features = self.node_encoder(batch.x)

        graph_embeddings = self.gat(
            node_features,
            batch.edge_index,
            batch.edge_attr,
            batch.batch
        )

        static_features = self.projection(graph_embeddings)

        vulnerability_scores = {}
        for vuln_type, head in self.vulnerability_heads.items():
            vulnerability_scores[vuln_type] = torch.sigmoid(head(static_features))

        return static_features, vulnerability_scores


class PDGFeatureExtractor:
    def __init__(self):
        self.structural_patterns = {
            'reentrancy': self._detect_reentrancy_pattern,
            'access_control': self._detect_access_control_pattern,
            'unchecked_call': self._detect_unchecked_call_pattern
        }

    def extract_features(self, pdg: nx.DiGraph) -> np.ndarray:
        features = []

        features.append(pdg.number_of_nodes())
        features.append(pdg.number_of_edges())

        if pdg.number_of_nodes() > 0:
            degrees = dict(pdg.degree())
            features.append(np.mean(list(degrees.values())))
            features.append(np.max(list(degrees.values())))
        else:
            features.append(0)
            features.append(0)

        try:
            features.append(nx.diameter(pdg.to_undirected()) if pdg.number_of_nodes() > 0 else 0)
        except:
            features.append(0)

        features.append(nx.density(pdg))

        node_types = [pdg.nodes[n].get('type', 'unknown') for n in pdg.nodes()]
        features.append(node_types.count('function'))
        features.append(node_types.count('variable'))
        features.append(node_types.count('modifier'))

        edge_types = [pdg.edges[e].get('type', 'unknown') for e in pdg.edges()]
        features.append(edge_types.count('calls'))
        features.append(edge_types.count('reads'))
        features.append(edge_types.count('writes'))

        for pattern_name, pattern_func in self.structural_patterns.items():
            features.append(1 if pattern_func(pdg) else 0)

        return np.array(features, dtype=np.float32)

    def _detect_reentrancy_pattern(self, pdg: nx.DiGraph) -> bool:
        for node in pdg.nodes():
            if pdg.nodes[node].get('type') == 'function':
                successors = list(pdg.successors(node))

                has_external_call = any(
                    pdg.edges[(node, s)].get('type') == 'calls' and
                    'external' in str(s).lower()
                    for s in successors
                )

                has_state_change = any(
                    pdg.edges[(node, s)].get('type') == 'writes'
                    for s in successors
                )

                if has_external_call and has_state_change:
                    return True
        return False

    def _detect_access_control_pattern(self, pdg: nx.DiGraph) -> bool:
        for node in pdg.nodes():
            if pdg.nodes[node].get('type') == 'function':
                predecessors = list(pdg.predecessors(node))

                has_modifier = any(
                    pdg.nodes[p].get('type') == 'modifier'
                    for p in predecessors
                )

                if not has_modifier and 'admin' in str(node).lower():
                    return True
        return False

    def _detect_unchecked_call_pattern(self, pdg: nx.DiGraph) -> bool:
        for node in pdg.nodes():
            if pdg.nodes[node].get('type') == 'function':
                edges = pdg.edges(node, data=True)

                for _, target, data in edges:
                    if data.get('type') == 'calls' and 'call' in str(target).lower():
                        successors = list(pdg.successors(node))
                        if not any('require' in str(s).lower() or 'assert' in str(s).lower()
                                 for s in successors):
                            return True
        return False