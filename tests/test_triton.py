import pytest
import torch
import networkx as nx
from unittest.mock import Mock, patch

from encoders.static_encoder import StaticEncoder, PDGFeatureExtractor
from encoders.dynamic_encoder import DynamicEncoder, analyze_execution_patterns
from encoders.semantic_encoder import SemanticEncoder, VulnerabilityPatternMatcher
from fusion.cross_modal_fusion import CrossModalFusion, compute_modality_importance
from orchestrator.agentic_workflow import AgenticOrchestrator, AnalysisPhase
from tools.slither_wrapper import SlitherWrapper
from tools.mythril_wrapper import MythrilWrapper
from utils.metrics import VulnerabilityMetrics


class TestStaticEncoder:
    def setup_method(self):
        self.encoder = StaticEncoder()
        self.pdg_extractor = PDGFeatureExtractor()

    def test_pdg_to_geometric_empty(self):
        empty_pdg = nx.DiGraph()
        data = self.encoder.pdg_to_geometric(empty_pdg)

        assert data.x.shape == (1, 5)
        assert data.edge_index.shape == (2, 0)

    def test_pdg_to_geometric_with_nodes(self):
        pdg = nx.DiGraph()
        pdg.add_node("Contract.function1", type="function")
        pdg.add_node("Contract.var1", type="variable")
        pdg.add_edge("Contract.function1", "Contract.var1", type="reads")

        data = self.encoder.pdg_to_geometric(pdg)

        assert data.x.shape[0] == 2
        assert data.edge_index.shape[1] == 1

    def test_static_encoder_forward(self):
        pdg = nx.DiGraph()
        pdg.add_node("test_function", type="function")

        with torch.no_grad():
            features, vuln_scores = self.encoder([pdg])

        assert features.shape[1] == 768
        assert 'reentrancy' in vuln_scores
        assert vuln_scores['reentrancy'].shape[0] == 1

    def test_pdg_feature_extraction(self):
        pdg = nx.DiGraph()
        pdg.add_node("func1", type="function")
        pdg.add_node("var1", type="variable")
        pdg.add_edge("func1", "var1", type="writes")

        features = self.pdg_extractor.extract_features(pdg)

        assert len(features) > 10
        assert features[0] == 2  # num nodes
        assert features[1] == 1  # num edges


class TestDynamicEncoder:
    def setup_method(self):
        self.encoder = DynamicEncoder()

    def test_opcode_mapping(self):
        assert 'CALL' in self.encoder.opcode_to_id
        assert 'SSTORE' in self.encoder.opcode_to_id
        assert self.encoder.opcode_to_id['CALL'] > 0

    def test_trace_encoding(self):
        traces = [{
            'steps': [
                {'opcode': 'CALL', 'gas': 1000000, 'depth': 1},
                {'opcode': 'SSTORE', 'gas': 5000, 'depth': 1}
            ]
        }]

        opcodes, contexts, masks = self.encoder.encode_traces(traces)

        assert opcodes.shape[0] == 1
        assert contexts.shape[0] == 1
        assert masks.shape[0] == 1

    def test_dynamic_encoder_forward(self):
        traces = [{
            'steps': [
                {'opcode': 'CALL', 'gas': 1000000, 'depth': 1},
                {'opcode': 'RETURN', 'gas': 2000, 'depth': 0}
            ]
        }]

        with torch.no_grad():
            features, vuln_scores = self.encoder(traces)

        assert features.shape[1] == 512
        assert 'reentrancy' in vuln_scores

    def test_execution_pattern_analysis(self):
        traces = [{
            'steps': [
                {'opcode': 'CALL', 'gas': 1000000, 'depth': 2},
                {'opcode': 'SSTORE', 'gas': 5000, 'depth': 1},
                {'opcode': 'DELEGATECALL', 'gas': 8000000, 'depth': 1}
            ]
        }]

        patterns = analyze_execution_patterns(traces)

        assert 'dangerous_delegatecall' in patterns
        assert patterns['dangerous_delegatecall'] >= 0.8


class TestSemanticEncoder:
    def setup_method(self):
        self.pattern_matcher = VulnerabilityPatternMatcher()

    def test_vulnerability_pattern_matching(self):
        code = """
        contract Test {
            function withdraw() public {
                msg.sender.call{value: balance[msg.sender]}("");
                balance[msg.sender] = 0;
            }
        }
        """

        scores = self.pattern_matcher.analyze_code(code)

        assert 'reentrancy' in scores
        assert scores['reentrancy'] > 0

    def test_access_control_pattern(self):
        code = """
        contract Test {
            function onlyOwner() public {
                require(msg.sender == owner);
                selfdestruct(payable(owner));
            }
        }
        """

        scores = self.pattern_matcher.analyze_code(code)

        assert scores['access_control'] > 0

    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModel.from_pretrained')
    def test_semantic_encoder_preprocessing(self, mock_model, mock_tokenizer):
        mock_tokenizer.return_value = Mock()
        mock_model.return_value = Mock()

        encoder = SemanticEncoder()

        code = "contract Test { function test() public {} }"
        processed = encoder.preprocess_solidity_code(code)

        assert 'contract' in processed
        assert 'function' in processed


class TestCrossModalFusion:
    def setup_method(self):
        self.fusion = CrossModalFusion()

    def test_fusion_forward(self):
        static_features = torch.randn(2, 768)
        dynamic_features = torch.randn(2, 512)
        semantic_features = torch.randn(2, 768)

        with torch.no_grad():
            result = self.fusion(static_features, dynamic_features, semantic_features)

        assert 'fused_features' in result
        assert 'vulnerability_logits' in result
        assert 'confidence_scores' in result
        assert 'modality_weights' in result

        assert result['fused_features'].shape == (2, 768)
        assert result['modality_weights'].shape == (2, 3)

    def test_modality_importance_computation(self):
        static_scores = torch.tensor([0.8, 0.6, 0.9])
        dynamic_scores = torch.tensor([0.3, 0.7, 0.4])
        semantic_scores = torch.tensor([0.5, 0.8, 0.6])

        weights = compute_modality_importance(
            static_scores, dynamic_scores, semantic_scores, 'reentrancy'
        )

        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights['dynamic'] > weights['static']  # For reentrancy


class TestAgenticOrchestrator:
    def setup_method(self):
        self.static_encoder = Mock()
        self.dynamic_encoder = Mock()
        self.semantic_encoder = Mock()
        self.fusion_module = Mock()

        self.orchestrator = AgenticOrchestrator(
            self.static_encoder,
            self.dynamic_encoder,
            self.semantic_encoder,
            self.fusion_module
        )

    @patch('tools.slither_wrapper.extract_static_features')
    @patch('tools.mythril_wrapper.extract_dynamic_features')
    def test_initial_analysis(self, mock_dynamic, mock_static):
        mock_static.return_value = {'vulnerabilities': [], 'pdg': nx.DiGraph()}
        mock_dynamic.return_value = {'vulnerabilities': [], 'execution_traces': []}

        self.fusion_module.return_value = {
            'fused_features': torch.randn(1, 768),
            'vulnerability_logits': torch.randn(1, 10),
            'modality_weights': torch.tensor([[0.3, 0.4, 0.3]])
        }

        result = self.orchestrator._initial_analysis(
            "contract Test {}", None, "reentrancy"
        )

        assert result.phase == AnalysisPhase.INITIAL
        assert 'static' in result.modality_contributions

    def test_decision_engine(self):
        should_continue, reason = self.orchestrator.decision_engine.should_continue_analysis(
            0.95, 'reentrancy', 1, 5
        )

        assert not should_continue  # High confidence should stop

        should_continue, reason = self.orchestrator.decision_engine.should_continue_analysis(
            0.6, 'reentrancy', 1, 5
        )

        assert should_continue  # Low confidence should continue


class TestVulnerabilityMetrics:
    def setup_method(self):
        self.metrics = VulnerabilityMetrics()

    def test_metrics_update_and_compute(self):
        preds = torch.tensor([1, 0, 1, 1])
        labels = torch.tensor([1, 0, 0, 1])
        confidences = torch.tensor([0.9, 0.1, 0.6, 0.8])

        self.metrics.update(preds, labels, confidences)
        results = self.metrics.compute_metrics()

        assert 'accuracy' in results
        assert 'precision' in results
        assert 'recall' in results
        assert 'f1_score' in results
        assert 'false_positive_rate' in results

        expected_accuracy = 3/4  # 3 correct out of 4
        assert abs(results['accuracy'] - expected_accuracy) < 1e-6

    def test_metrics_reset(self):
        preds = torch.tensor([1, 0])
        labels = torch.tensor([1, 0])

        self.metrics.update(preds, labels)
        self.metrics.reset()

        assert len(self.metrics.predictions) == 0
        assert len(self.metrics.labels) == 0


class TestIntegration:
    @patch('tools.slither_wrapper.SlitherWrapper')
    @patch('tools.mythril_wrapper.MythrilWrapper')
    def test_end_to_end_workflow(self, mock_mythril_class, mock_slither_class):
        # Mock the analysis tools
        mock_slither = Mock()
        mock_slither.analyze_contract.return_value = {
            'success': True,
            'vulnerabilities': [],
            'pdg': nx.DiGraph(),
            'summary': {'total_issues': 0, 'high_severity': 0, 'medium_severity': 0, 'low_severity': 0}
        }
        mock_slither_class.return_value = mock_slither

        mock_mythril = Mock()
        mock_mythril.analyze_contract.return_value = {
            'success': True,
            'vulnerabilities': [],
            'execution_traces': [],
            'summary': {'total_issues': 0, 'critical': 0, 'medium': 0, 'low': 0, 'unique_vulnerability_types': 0}
        }
        mock_mythril_class.return_value = mock_mythril

        # Create real components
        static_encoder = StaticEncoder(output_dim=768)
        dynamic_encoder = DynamicEncoder(output_dim=512)
        fusion_module = CrossModalFusion()

        # Mock semantic encoder to avoid downloading models
        semantic_encoder = Mock()
        semantic_encoder.return_value = (torch.randn(1, 768), {'reentrancy': torch.tensor([[0.5]])})

        orchestrator = AgenticOrchestrator(
            static_encoder,
            dynamic_encoder,
            semantic_encoder,
            fusion_module,
            confidence_threshold=0.8,
            max_iterations=2
        )

        # Test contract
        test_contract = """
        pragma solidity ^0.8.0;
        contract Test {
            function test() public pure returns (uint256) {
                return 42;
            }
        }
        """

        # This should not crash and should return a result
        result = orchestrator.analyze_contract(test_contract, "Test", "reentrancy")

        assert 'final_result' in result
        assert 'analysis_history' in result
        assert 'workflow_summary' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])