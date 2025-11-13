import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


class OpcodeEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 50, embedding_dim: int = 128):
        super(OpcodeEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.position_encoding = PositionalEncoding(embedding_dim)

    def forward(self, opcodes: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(opcodes)
        return self.position_encoding(embedded)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class ExecutionTraceLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super(ExecutionTraceLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.attention = nn.MultiheadAttention(
            hidden_dim * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        lstm_out, (hidden, cell) = self.lstm(x)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out, key_padding_mask=mask)
        lstm_out = self.layer_norm(lstm_out + self.dropout(attn_out))

        if mask is not None:
            lstm_out = lstm_out * (~mask).unsqueeze(-1).float()
            lengths = (~mask).sum(dim=1)
            last_outputs = []
            for i, length in enumerate(lengths):
                if length > 0:
                    last_outputs.append(lstm_out[i, length - 1])
                else:
                    last_outputs.append(torch.zeros_like(lstm_out[i, 0]))
            return torch.stack(last_outputs)
        else:
            return lstm_out[:, -1, :]


class DynamicEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 3,
        max_trace_length: int = 512,
        dropout: float = 0.2
    ):
        super(DynamicEncoder, self).__init__()

        self.max_trace_length = max_trace_length

        self.opcode_embedding = OpcodeEmbedding(vocab_size, embedding_dim)

        self.context_encoder = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )

        self.trace_lstm = ExecutionTraceLSTM(
            embedding_dim * 2,
            hidden_dim,
            num_layers,
            dropout,
            bidirectional=True
        )

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.vulnerability_heads = nn.ModuleDict({
            'access_control': self._create_vuln_head(output_dim),
            'arithmetic': self._create_vuln_head(output_dim),
            'bad_randomness': self._create_vuln_head(output_dim),
            'denial_of_service': self._create_vuln_head(output_dim),
            'front_running': self._create_vuln_head(output_dim),
            'reentrancy': self._create_vuln_head(output_dim),
            'short_addresses': self._create_vuln_head(output_dim),
            'time_manipulation': self._create_vuln_head(output_dim),
            'unchecked_low_level_calls': self._create_vuln_head(output_dim),
            'other': self._create_vuln_head(output_dim)
        })

        self.opcode_to_id = self._create_opcode_mapping()

    def _create_vuln_head(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1)
        )

    def _create_opcode_mapping(self) -> Dict[str, int]:
        opcodes = [
            'STOP', 'ADD', 'MUL', 'SUB', 'DIV', 'MOD', 'EXP', 'NOT',
            'LT', 'GT', 'EQ', 'AND', 'OR', 'XOR', 'BYTE', 'SHL', 'SHR',
            'PUSH1', 'PUSH2', 'PUSH32', 'DUP1', 'DUP16', 'SWAP1', 'SWAP16',
            'MLOAD', 'MSTORE', 'SLOAD', 'SSTORE', 'JUMP', 'JUMPI', 'PC',
            'MSIZE', 'GAS', 'JUMPDEST', 'CALL', 'DELEGATECALL', 'STATICCALL',
            'RETURN', 'REVERT', 'SELFDESTRUCT', 'CREATE', 'CREATE2',
            'CALLDATALOAD', 'CALLDATASIZE', 'CALLDATACOPY', 'CODECOPY',
            'EXTCODESIZE', 'EXTCODECOPY', 'RETURNDATASIZE', 'RETURNDATACOPY'
        ]
        return {op: i + 1 for i, op in enumerate(opcodes)}

    def encode_traces(self, traces: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_opcodes = []
        batch_contexts = []
        batch_masks = []

        for trace in traces:
            opcodes = []
            contexts = []

            steps = trace.get('steps', [])[:self.max_trace_length]

            for step in steps:
                opcode = step.get('opcode', 'UNKNOWN')
                opcode_id = self.opcode_to_id.get(opcode, 0)
                opcodes.append(opcode_id)

                gas = min(step.get('gas', 0) / 1000000, 1.0)
                depth = min(step.get('depth', 0) / 10, 1.0)
                stack_size = min(len(step.get('stack', [])) / 100, 1.0)

                is_storage = 1.0 if opcode in ['SSTORE', 'SLOAD'] else 0.0
                is_call = 1.0 if 'CALL' in opcode else 0.0
                is_jump = 1.0 if 'JUMP' in opcode else 0.0
                is_arithmetic = 1.0 if opcode in ['ADD', 'SUB', 'MUL', 'DIV', 'MOD'] else 0.0

                context = [gas, depth, stack_size, is_storage, is_call, is_jump, is_arithmetic]
                contexts.append(context)

            while len(opcodes) < self.max_trace_length:
                opcodes.append(0)
                contexts.append([0] * 7)

            mask = [False] * len(steps) + [True] * (self.max_trace_length - len(steps))

            batch_opcodes.append(opcodes[:self.max_trace_length])
            batch_contexts.append(contexts[:self.max_trace_length])
            batch_masks.append(mask[:self.max_trace_length])

        return (
            torch.tensor(batch_opcodes, dtype=torch.long),
            torch.tensor(batch_contexts, dtype=torch.float32),
            torch.tensor(batch_masks, dtype=torch.bool)
        )

    def forward(
        self,
        execution_traces: List[Dict],
        vulnerability_type: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        opcodes, contexts, masks = self.encode_traces(execution_traces)

        # Move tensors to same device as model
        device = next(self.parameters()).device
        opcodes = opcodes.to(device)
        contexts = contexts.to(device)
        masks = masks.to(device)

        opcode_embeddings = self.opcode_embedding(opcodes)
        context_embeddings = self.context_encoder(contexts)

        combined = torch.cat([opcode_embeddings, context_embeddings], dim=-1)

        trace_features = self.trace_lstm(combined, masks)

        dynamic_features = self.projection(trace_features)

        vulnerability_scores = {}
        for vuln_type, head in self.vulnerability_heads.items():
            vulnerability_scores[vuln_type] = torch.sigmoid(head(dynamic_features))

        return dynamic_features, vulnerability_scores


class TracePatternDetector(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super(TracePatternDetector, self).__init__()

        self.reentrancy_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.loop_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.gas_pattern_detector = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, trace_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        patterns = {
            'reentrancy_risk': torch.sigmoid(self.reentrancy_detector(trace_features)),
            'loop_risk': torch.sigmoid(self.loop_detector(trace_features)),
            'gas_risk': torch.sigmoid(self.gas_pattern_detector(trace_features))
        }
        return patterns


def analyze_execution_patterns(traces: List[Dict]) -> Dict[str, float]:
    pattern_scores = {
        'reentrancy_pattern': 0.0,
        'excessive_loops': 0.0,
        'unbounded_gas': 0.0,
        'dangerous_delegatecall': 0.0,
        'storage_collision': 0.0
    }

    for trace in traces:
        steps = trace.get('steps', [])

        call_depth_changes = []
        storage_ops = []
        gas_consumption = []

        for i, step in enumerate(steps):
            opcode = step.get('opcode', '')
            depth = step.get('depth', 0)
            gas = step.get('gas', 0)

            if i > 0:
                prev_depth = steps[i-1].get('depth', 0)
                if depth != prev_depth:
                    call_depth_changes.append((i, depth - prev_depth))

            if opcode in ['SSTORE', 'SLOAD']:
                storage_ops.append((i, opcode, step.get('address', '')))

            gas_consumption.append(gas)

            if opcode == 'DELEGATECALL':
                pattern_scores['dangerous_delegatecall'] = max(
                    pattern_scores['dangerous_delegatecall'],
                    0.8
                )

        if len(call_depth_changes) > 2:
            for i in range(len(call_depth_changes) - 1):
                if call_depth_changes[i][1] > 0 and call_depth_changes[i+1][1] < 0:
                    for j in range(call_depth_changes[i][0], call_depth_changes[i+1][0]):
                        if steps[j].get('opcode') == 'SSTORE':
                            pattern_scores['reentrancy_pattern'] = max(
                                pattern_scores['reentrancy_pattern'],
                                0.9
                            )

        loop_indicators = 0
        for i in range(1, len(steps)):
            if steps[i].get('opcode') == 'JUMPI' and steps[i-1].get('opcode') in ['LT', 'GT', 'EQ']:
                loop_indicators += 1

        if loop_indicators > 10:
            pattern_scores['excessive_loops'] = min(loop_indicators / 20.0, 1.0)

        if gas_consumption and max(gas_consumption) > 8000000:
            pattern_scores['unbounded_gas'] = min(max(gas_consumption) / 10000000, 1.0)

        storage_addresses = {}
        for _, opcode, addr in storage_ops:
            if addr in storage_addresses and storage_addresses[addr] != opcode:
                pattern_scores['storage_collision'] = 0.7
            storage_addresses[addr] = opcode

    return pattern_scores