import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import re


class SemanticEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/graphcodebert-base",
        output_dim: int = 768,
        max_length: int = 512,
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        super(SemanticEncoder, self).__init__()

        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = output_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.vulnerability_embeddings = nn.Embedding(11, 64)

        self.projection = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + 64, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )

        self.vulnerability_heads = nn.ModuleDict({
            'access_control': self._create_classification_head(output_dim),
            'arithmetic': self._create_classification_head(output_dim),
            'bad_randomness': self._create_classification_head(output_dim),
            'denial_of_service': self._create_classification_head(output_dim),
            'front_running': self._create_classification_head(output_dim),
            'reentrancy': self._create_classification_head(output_dim),
            'short_addresses': self._create_classification_head(output_dim),
            'time_manipulation': self._create_classification_head(output_dim),
            'unchecked_low_level_calls': self._create_classification_head(output_dim),
            'other': self._create_classification_head(output_dim),
            'safe': self._create_classification_head(output_dim)
        })

        self.vulnerability_type_mapping = {
            'access_control': 0,
            'arithmetic': 1,
            'bad_randomness': 2,
            'denial_of_service': 3,
            'front_running': 4,
            'reentrancy': 5,
            'short_addresses': 6,
            'time_manipulation': 7,
            'unchecked_low_level_calls': 8,
            'other': 9,
            'safe': 10
        }

    def _create_classification_head(self, input_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, 1)
        )

    def preprocess_solidity_code(self, source_code: str) -> str:
        source_code = re.sub(r'//.*?\n', '\n', source_code)
        source_code = re.sub(r'/\*.*?\*/', '', source_code, flags=re.DOTALL)

        source_code = re.sub(r'\s+', ' ', source_code)

        keywords = [
            'function', 'modifier', 'contract', 'require', 'assert',
            'msg.sender', 'msg.value', 'block.timestamp', 'tx.origin',
            'call', 'delegatecall', 'selfdestruct', 'transfer', 'send'
        ]

        for keyword in keywords:
            source_code = source_code.replace(keyword, f' {keyword} ')

        source_code = re.sub(r'\s+', ' ', source_code).strip()

        return source_code

    def tokenize_batch(self, source_codes: List[str]) -> Dict[str, torch.Tensor]:
        processed_codes = [self.preprocess_solidity_code(code) for code in source_codes]

        encoded = self.tokenizer(
            processed_codes,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return encoded

    def forward(
        self,
        source_codes: List[str],
        vulnerability_types: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        encoded = self.tokenize_batch(source_codes)

        # Move inputs to the same device as the model
        device = next(self.bert.parameters()).device
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        bert_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        pooled_output = bert_outputs.pooler_output

        if vulnerability_types:
            vuln_type_ids = []
            for vuln_type in vulnerability_types:
                type_id = self.vulnerability_type_mapping.get(vuln_type, 0)
                vuln_type_ids.append(type_id)

            vuln_type_tensor = torch.tensor(vuln_type_ids, dtype=torch.long, device=pooled_output.device)
            vuln_embeddings = self.vulnerability_embeddings(vuln_type_tensor)

            combined = torch.cat([pooled_output, vuln_embeddings], dim=-1)
        else:
            zero_vuln = torch.zeros(len(source_codes), 64, device=pooled_output.device)
            combined = torch.cat([pooled_output, zero_vuln], dim=-1)

        semantic_features = self.projection(combined)

        vulnerability_scores = {}
        for vuln_type, head in self.vulnerability_heads.items():
            scores = torch.sigmoid(head(semantic_features))
            vulnerability_scores[vuln_type] = scores

        return semantic_features, vulnerability_scores


class VulnerabilityPatternMatcher:
    def __init__(self):
        self.patterns = {
            'reentrancy': [
                r'\.call\s*\(',
                r'\.send\s*\(',
                r'\.transfer\s*\(',
                r'external\s+.*\s+payable',
                r'balances\[.*\]\s*=.*after.*call'
            ],
            'overflow': [
                r'\+\+',
                r'--',
                r'\+(?!\+)',
                r'-(?!-)',
                r'\*',
                r'\/(?!\/)',
                r'unchecked\s*\{'
            ],
            'access_control': [
                r'onlyOwner',
                r'require\s*\(\s*msg\.sender\s*==',
                r'modifier\s+\w+.*msg\.sender',
                r'tx\.origin\s*==',
                r'selfdestruct\s*\('
            ],
            'timestamp_dependency': [
                r'block\.timestamp',
                r'block\.number',
                r'now\s',
                r'block\.difficulty'
            ],
            'delegatecall': [
                r'delegatecall\s*\(',
                r'\.delegatecall\s*\('
            ]
        }

    def analyze_code(self, source_code: str) -> Dict[str, float]:
        scores = {}

        for vuln_type, patterns in self.patterns.items():
            score = 0.0
            matches = 0

            for pattern in patterns:
                found = re.findall(pattern, source_code, re.IGNORECASE)
                matches += len(found)

            if matches > 0:
                score = min(matches / 5.0, 1.0)

            scores[vuln_type] = score

        return scores


class CodeStructureAnalyzer:
    def __init__(self):
        pass

    def extract_functions(self, source_code: str) -> List[Dict]:
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*([^{]*)\s*\{'
        functions = []

        for match in re.finditer(function_pattern, source_code):
            func_name = match.group(1)
            modifiers = match.group(2).strip()

            visibility = 'internal'
            if 'public' in modifiers:
                visibility = 'public'
            elif 'external' in modifiers:
                visibility = 'external'
            elif 'private' in modifiers:
                visibility = 'private'

            is_payable = 'payable' in modifiers
            is_view = 'view' in modifiers or 'pure' in modifiers

            start_pos = match.start()
            brace_count = 0
            i = match.end() - 1

            while i < len(source_code):
                if source_code[i] == '{':
                    brace_count += 1
                elif source_code[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        break
                i += 1

            func_body = source_code[match.start():i+1] if i < len(source_code) else source_code[match.start():]

            functions.append({
                'name': func_name,
                'visibility': visibility,
                'is_payable': is_payable,
                'is_view': is_view,
                'body': func_body,
                'start_pos': start_pos,
                'end_pos': i + 1 if i < len(source_code) else len(source_code)
            })

        return functions

    def analyze_contract_structure(self, source_code: str) -> Dict:
        functions = self.extract_functions(source_code)

        structure = {
            'total_functions': len(functions),
            'public_functions': sum(1 for f in functions if f['visibility'] == 'public'),
            'external_functions': sum(1 for f in functions if f['visibility'] == 'external'),
            'payable_functions': sum(1 for f in functions if f['is_payable']),
            'view_functions': sum(1 for f in functions if f['is_view']),
            'has_fallback': 'fallback' in source_code,
            'has_receive': 'receive' in source_code,
            'uses_assembly': 'assembly' in source_code,
            'imports_count': len(re.findall(r'import\s+', source_code)),
            'pragma_version': self._extract_pragma_version(source_code)
        }

        return structure

    def _extract_pragma_version(self, source_code: str) -> str:
        pragma_match = re.search(r'pragma\s+solidity\s+([^;]+);', source_code)
        return pragma_match.group(1).strip() if pragma_match else 'unknown'


def extract_semantic_features(source_code: str, vulnerability_type: Optional[str] = None) -> Dict:
    pattern_matcher = VulnerabilityPatternMatcher()
    structure_analyzer = CodeStructureAnalyzer()

    pattern_scores = pattern_matcher.analyze_code(source_code)
    structure_info = structure_analyzer.analyze_contract_structure(source_code)

    features = {
        'pattern_scores': pattern_scores,
        'structure': structure_info,
        'code_length': len(source_code),
        'complexity_score': structure_info['total_functions'] * 0.1 +
                          structure_info['public_functions'] * 0.2 +
                          structure_info['payable_functions'] * 0.3
    }

    if vulnerability_type and vulnerability_type in pattern_scores:
        features['target_pattern_score'] = pattern_scores[vulnerability_type]

    return features