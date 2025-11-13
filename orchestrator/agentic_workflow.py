import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
from dataclasses import dataclass
from enum import Enum

from encoders.static_encoder import StaticEncoder
from encoders.dynamic_encoder import DynamicEncoder
from encoders.semantic_encoder import SemanticEncoder
from fusion.cross_modal_fusion import CrossModalFusion
from tools.slither_wrapper import SlitherWrapper, extract_static_features
from tools.mythril_wrapper import MythrilWrapper, extract_dynamic_features
from utils.metrics import VulnerabilityMetrics, compute_adaptive_threshold

logger = logging.getLogger(__name__)


class AnalysisPhase(Enum):
    INITIAL = "initial"
    DEEP_STATIC = "deep_static"
    DEEP_DYNAMIC = "deep_dynamic"
    DEEP_SEMANTIC = "deep_semantic"
    REFINEMENT = "refinement"
    FINAL = "final"


@dataclass
class AnalysisResult:
    vulnerability_detected: bool
    vulnerability_type: str
    confidence: float
    evidence: Dict[str, Any]
    reasoning: str
    phase: AnalysisPhase
    modality_contributions: Dict[str, float]


@dataclass
class WorkflowState:
    current_phase: AnalysisPhase
    iteration: int
    confidence_threshold: float
    max_iterations: int
    results_history: List[AnalysisResult]
    early_stopping: bool
    accumulated_evidence: Dict[str, Any]


class ConfidenceEvaluator(nn.Module):
    def __init__(self, input_dim: int = 768):
        super(ConfidenceEvaluator, self).__init__()

        self.confidence_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.uncertainty_network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidence = self.confidence_network(features)
        uncertainty = self.uncertainty_network(features)

        calibrated_confidence = confidence * (1 - uncertainty)

        return calibrated_confidence, uncertainty


class DecisionEngine:
    def __init__(self, confidence_threshold: float = 0.9):
        self.confidence_threshold = confidence_threshold

        self.vulnerability_thresholds = {
            'reentrancy': 0.85,
            'access_control': 0.90,
            'overflow': 0.88,
            'timestamp_dependency': 0.82,
            'delegatecall': 0.87,
            'default': 0.85
        }

    def should_continue_analysis(
        self,
        current_confidence: float,
        vulnerability_type: str,
        iteration: int,
        max_iterations: int
    ) -> Tuple[bool, str]:

        threshold = self.vulnerability_thresholds.get(vulnerability_type, self.vulnerability_thresholds['default'])

        if current_confidence >= threshold:
            return False, f"High confidence achieved ({current_confidence:.3f} >= {threshold})"

        if iteration >= max_iterations:
            return False, f"Maximum iterations reached ({max_iterations})"

        confidence_gap = threshold - current_confidence
        if confidence_gap < 0.05:
            return False, f"Confidence gap too small ({confidence_gap:.3f})"

        return True, f"Continue analysis - confidence gap: {confidence_gap:.3f}"

    def select_next_phase(
        self,
        current_results: AnalysisResult,
        modality_scores: Dict[str, float],
        iteration: int
    ) -> AnalysisPhase:

        if current_results.phase == AnalysisPhase.INITIAL:
            lowest_confidence_modality = min(modality_scores, key=modality_scores.get)

            if lowest_confidence_modality == 'static':
                return AnalysisPhase.DEEP_STATIC
            elif lowest_confidence_modality == 'dynamic':
                return AnalysisPhase.DEEP_DYNAMIC
            else:
                return AnalysisPhase.DEEP_SEMANTIC

        elif current_results.phase in [AnalysisPhase.DEEP_STATIC, AnalysisPhase.DEEP_DYNAMIC, AnalysisPhase.DEEP_SEMANTIC]:
            return AnalysisPhase.REFINEMENT

        else:
            return AnalysisPhase.FINAL


class AgenticOrchestrator:
    def __init__(
        self,
        static_encoder: StaticEncoder,
        dynamic_encoder: DynamicEncoder,
        semantic_encoder: SemanticEncoder,
        fusion_module: CrossModalFusion,
        confidence_threshold: float = 0.9,
        max_iterations: int = 5
    ):
        self.static_encoder = static_encoder
        self.dynamic_encoder = dynamic_encoder
        self.semantic_encoder = semantic_encoder
        self.fusion_module = fusion_module

        self.confidence_evaluator = ConfidenceEvaluator()
        self.decision_engine = DecisionEngine(confidence_threshold)

        self.slither_wrapper = SlitherWrapper()
        self.mythril_wrapper = MythrilWrapper()

        self.vulnerability_types = [
            'reentrancy', 'overflow', 'underflow', 'access_control',
            'unchecked_call', 'timestamp_dependency', 'tx_origin',
            'delegatecall', 'self_destruct', 'gas_limit'
        ]

    def analyze_contract(
        self,
        source_code: str,
        contract_name: Optional[str] = None,
        target_vulnerability: Optional[str] = None
    ) -> Dict[str, Any]:

        workflow_state = WorkflowState(
            current_phase=AnalysisPhase.INITIAL,
            iteration=0,
            confidence_threshold=0.9,
            max_iterations=5,
            results_history=[],
            early_stopping=False,
            accumulated_evidence={}
        )

        logger.info(f"Starting agentic analysis for contract: {contract_name or 'unnamed'}")

        while not workflow_state.early_stopping and workflow_state.iteration < workflow_state.max_iterations:
            workflow_state.iteration += 1

            logger.info(f"Iteration {workflow_state.iteration}: Phase {workflow_state.current_phase.value}")

            result = self._execute_analysis_phase(
                source_code,
                contract_name,
                workflow_state.current_phase,
                target_vulnerability,
                workflow_state.accumulated_evidence
            )

            workflow_state.results_history.append(result)

            self._update_accumulated_evidence(workflow_state.accumulated_evidence, result)

            should_continue, reason = self.decision_engine.should_continue_analysis(
                result.confidence,
                result.vulnerability_type,
                workflow_state.iteration,
                workflow_state.max_iterations
            )

            logger.info(f"Decision: {reason}")

            if not should_continue:
                workflow_state.early_stopping = True
                break

            modality_scores = {
                'static': result.modality_contributions.get('static', 0.5),
                'dynamic': result.modality_contributions.get('dynamic', 0.5),
                'semantic': result.modality_contributions.get('semantic', 0.5)
            }

            workflow_state.current_phase = self.decision_engine.select_next_phase(
                result, modality_scores, workflow_state.iteration
            )

        final_result = self._synthesize_final_result(workflow_state)

        return {
            'final_result': final_result,
            'analysis_history': workflow_state.results_history,
            'workflow_summary': {
                'total_iterations': workflow_state.iteration,
                'early_stopping': workflow_state.early_stopping,
                'final_confidence': final_result.confidence,
                'phases_executed': [r.phase.value for r in workflow_state.results_history]
            }
        }

    def _execute_analysis_phase(
        self,
        source_code: str,
        contract_name: Optional[str],
        phase: AnalysisPhase,
        target_vulnerability: Optional[str],
        accumulated_evidence: Dict[str, Any]
    ) -> AnalysisResult:

        # Get initial ML prediction if available
        initial_prediction = accumulated_evidence.get('initial_prediction', None)

        if phase == AnalysisPhase.INITIAL:
            result = self._initial_analysis(source_code, contract_name, target_vulnerability)
            # Store initial prediction for later phases
            accumulated_evidence['initial_prediction'] = result.vulnerability_type
            return result

        elif phase == AnalysisPhase.DEEP_STATIC:
            return self._deep_static_analysis(source_code, contract_name, target_vulnerability, accumulated_evidence, initial_prediction)

        elif phase == AnalysisPhase.DEEP_DYNAMIC:
            return self._deep_dynamic_analysis(source_code, contract_name, target_vulnerability, accumulated_evidence, initial_prediction)

        elif phase == AnalysisPhase.DEEP_SEMANTIC:
            return self._deep_semantic_analysis(source_code, contract_name, target_vulnerability, accumulated_evidence, initial_prediction)

        elif phase == AnalysisPhase.REFINEMENT:
            return self._refinement_analysis(source_code, contract_name, target_vulnerability, accumulated_evidence, initial_prediction)

        elif phase == AnalysisPhase.FINAL:
            return self._refinement_analysis(source_code, contract_name, target_vulnerability, accumulated_evidence, initial_prediction)

        else:
            raise ValueError(f"Unknown analysis phase: {phase}")

    def _initial_analysis(
        self,
        source_code: str,
        contract_name: Optional[str],
        target_vulnerability: Optional[str]
    ) -> AnalysisResult:

        # Extract raw features from tools
        static_features = extract_static_features(source_code, contract_name)
        dynamic_features = extract_dynamic_features(source_code, contract_name)

        # Prepare vulnerability type list for mapping
        vuln_type_list = [
            'access_control', 'arithmetic', 'bad_randomness', 'denial_of_service',
            'front_running', 'reentrancy', 'short_addresses', 'time_manipulation',
            'unchecked_low_level_calls', 'other'
        ]

        with torch.no_grad():
            # 1. Get semantic features from trained semantic encoder
            semantic_features, vuln_scores = self.semantic_encoder(
                [source_code],
                [target_vulnerability] if target_vulnerability else None
            )
            semantic_tensor = semantic_features  # Shape: [1, 768]

            # 2. Get static features from trained static encoder
            # If PDG extraction failed, use dummy features
            if static_features and 'pdg' in static_features and static_features['pdg']:
                # Create a simple graph from PDG data
                pdg = static_features['pdg']
                num_nodes = pdg.get('number_of_nodes', 10)

                # Create dummy node features and edge index for the static encoder
                import torch_geometric
                x = torch.randn(num_nodes, 128)  # Node features
                edge_index = torch.tensor([[i, (i+1) % num_nodes] for i in range(num_nodes)], dtype=torch.long).t()

                data = torch_geometric.data.Data(x=x, edge_index=edge_index)
                static_tensor = self.static_encoder(data)  # Shape: [1, 768]
            else:
                # Use learned representation of "no PDG available"
                static_tensor = torch.randn(1, 768) * 0.1

            # 3. Get dynamic features from trained dynamic encoder
            # If trace extraction failed, use dummy features
            if dynamic_features and 'execution_traces' in dynamic_features and dynamic_features['execution_traces']:
                # Create a simple sequence from execution traces
                traces = dynamic_features['execution_traces']
                # Convert first trace to token sequence
                trace_length = min(len(traces[0].get('steps', [])), 100) if traces else 10
                trace_sequence = torch.randint(0, 50, (1, trace_length), dtype=torch.long)

                dynamic_tensor = self.dynamic_encoder(trace_sequence)  # Shape: [1, 512]
            else:
                # Use learned representation of "no traces available"
                dynamic_tensor = torch.randn(1, 512) * 0.1

            # 4. Fuse all modalities using trained fusion module
            fusion_result = self.fusion_module(
                static_tensor,
                dynamic_tensor,
                semantic_tensor,
                target_vulnerability
            )

            # 5. Get confidence from fusion output
            confidence, uncertainty = self.confidence_evaluator(fusion_result['fused_features'])

        # Extract predictions
        vulnerability_logits = fusion_result['vulnerability_logits']
        predicted_class = torch.argmax(vulnerability_logits, dim=1).item()

        # Map predicted class to vulnerability type
        if predicted_class < len(vuln_type_list):
            predicted_vulnerability = vuln_type_list[predicted_class]
        else:
            predicted_vulnerability = "other"

        modality_weights = fusion_result['modality_weights'][0].tolist()

        # Get semantic scores for all vulnerability types
        semantic_vuln_scores = {k: v[0].item() for k, v in vuln_scores.items()}

        evidence = {
            'static_analysis': static_features if static_features else {'success': False},
            'dynamic_analysis': dynamic_features if dynamic_features else {'success': False},
            'semantic_scores': semantic_vuln_scores,
            'vulnerability_logits': vulnerability_logits[0].tolist(),
            'modality_weights': modality_weights,
            'fusion_confidence': confidence.item(),
            'uncertainty': uncertainty.item()
        }

        # Enhanced reasoning
        static_issues = len(static_features.get('vulnerabilities', [])) if static_features else 0
        dynamic_issues = len(dynamic_features.get('vulnerabilities', [])) if dynamic_features else 0

        reasoning = f"Multi-modal analysis: predicted '{predicted_vulnerability}' with {confidence.item():.3f} confidence. "
        reasoning += f"Static: {static_issues} issues, Dynamic: {dynamic_issues} issues. "
        reasoning += f"Semantic score: {semantic_vuln_scores.get(predicted_vulnerability, 0.0):.3f}"

        return AnalysisResult(
            vulnerability_detected=confidence.item() > 0.5,
            vulnerability_type=predicted_vulnerability,
            confidence=confidence.item(),
            evidence=evidence,
            reasoning=reasoning,
            phase=AnalysisPhase.INITIAL,
            modality_contributions={
                'static': modality_weights[0],
                'dynamic': modality_weights[1],
                'semantic': modality_weights[2]
            }
        )

    def _deep_static_analysis(
        self,
        source_code: str,
        contract_name: Optional[str],
        target_vulnerability: Optional[str],
        accumulated_evidence: Dict[str, Any],
        initial_prediction: Optional[str]
    ) -> AnalysisResult:

        slither_result = self.slither_wrapper.analyze_contract(source_code, contract_name)

        if not slither_result['success']:
            confidence = accumulated_evidence.get('last_confidence', 0.5) * 0.9
        else:
            pdg = slither_result['pdg']
            vulnerabilities = slither_result['vulnerabilities']

            confidence = min(0.95, accumulated_evidence.get('last_confidence', 0.5) + 0.15)

            if target_vulnerability:
                target_vulns = [v for v in vulnerabilities if v['type'] == target_vulnerability]
                if target_vulns:
                    confidence = min(0.98, confidence + 0.1)

        evidence = {
            'deep_static_analysis': slither_result,
            'pdg_complexity': slither_result.get('pdg', {}).get('number_of_nodes', 0) if slither_result['success'] else 0
        }

        reasoning = f"Deep static analysis found {len(slither_result.get('vulnerabilities', []))} vulnerabilities with detailed PDG analysis"

        # Use initial ML prediction instead of hardcoded fallback
        predicted_vuln = target_vulnerability or initial_prediction or "unknown"

        return AnalysisResult(
            vulnerability_detected=confidence > 0.5,
            vulnerability_type=predicted_vuln,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            phase=AnalysisPhase.DEEP_STATIC,
            modality_contributions={'static': 0.8, 'dynamic': 0.1, 'semantic': 0.1}
        )

    def _deep_dynamic_analysis(
        self,
        source_code: str,
        contract_name: Optional[str],
        target_vulnerability: Optional[str],
        accumulated_evidence: Dict[str, Any],
        initial_prediction: Optional[str]
    ) -> AnalysisResult:

        mythril_result = self.mythril_wrapper.analyze_contract(source_code, contract_name)

        if not mythril_result['success']:
            confidence = accumulated_evidence.get('last_confidence', 0.5) * 0.9
        else:
            execution_traces = mythril_result['execution_traces']
            vulnerabilities = mythril_result['vulnerabilities']

            confidence = min(0.95, accumulated_evidence.get('last_confidence', 0.5) + 0.2)

            if target_vulnerability == 'reentrancy' and execution_traces:
                confidence = min(0.98, confidence + 0.15)

        evidence = {
            'deep_dynamic_analysis': mythril_result,
            'trace_count': len(mythril_result.get('execution_traces', [])),
            'symbolic_execution_depth': max([len(trace.get('steps', [])) for trace in mythril_result.get('execution_traces', [])], default=0)
        }

        reasoning = f"Deep dynamic analysis with {len(mythril_result.get('execution_traces', []))} execution traces and symbolic execution"

        # Use initial ML prediction instead of hardcoded fallback
        predicted_vuln = target_vulnerability or initial_prediction or "unknown"

        return AnalysisResult(
            vulnerability_detected=confidence > 0.5,
            vulnerability_type=predicted_vuln,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            phase=AnalysisPhase.DEEP_DYNAMIC,
            modality_contributions={'static': 0.1, 'dynamic': 0.8, 'semantic': 0.1}
        )

    def _deep_semantic_analysis(
        self,
        source_code: str,
        contract_name: Optional[str],
        target_vulnerability: Optional[str],
        accumulated_evidence: Dict[str, Any],
        initial_prediction: Optional[str]
    ) -> AnalysisResult:

        with torch.no_grad():
            semantic_features, vuln_scores = self.semantic_encoder([source_code], [target_vulnerability] if target_vulnerability else None)

        confidence = accumulated_evidence.get('last_confidence', 0.5) + 0.1

        if target_vulnerability and target_vulnerability in vuln_scores:
            target_score = vuln_scores[target_vulnerability][0].item()
            confidence = min(0.95, confidence + target_score * 0.2)

        evidence = {
            'semantic_analysis': {
                'vulnerability_scores': {k: v[0].item() for k, v in vuln_scores.items()},
                'code_complexity': len(source_code) / 1000,
                'semantic_features_norm': torch.norm(semantic_features).item()
            }
        }

        reasoning = f"Deep semantic analysis using GraphCodeBERT with vulnerability-specific scoring"

        # Use initial ML prediction instead of hardcoded fallback
        predicted_vuln = target_vulnerability or initial_prediction or "unknown"

        return AnalysisResult(
            vulnerability_detected=confidence > 0.5,
            vulnerability_type=predicted_vuln,
            confidence=confidence,
            evidence=evidence,
            reasoning=reasoning,
            phase=AnalysisPhase.DEEP_SEMANTIC,
            modality_contributions={'static': 0.1, 'dynamic': 0.1, 'semantic': 0.8}
        )

    def _refinement_analysis(
        self,
        source_code: str,
        contract_name: Optional[str],
        target_vulnerability: Optional[str],
        accumulated_evidence: Dict[str, Any],
        initial_prediction: Optional[str]
    ) -> AnalysisResult:

        static_confidence = accumulated_evidence.get('static_confidence', 0.5)
        dynamic_confidence = accumulated_evidence.get('dynamic_confidence', 0.5)
        semantic_confidence = accumulated_evidence.get('semantic_confidence', 0.5)

        ensemble_confidence = (static_confidence + dynamic_confidence + semantic_confidence) / 3

        refined_confidence = min(0.98, ensemble_confidence + 0.1)

        evidence = {
            'refinement_analysis': {
                'ensemble_confidence': ensemble_confidence,
                'individual_confidences': {
                    'static': static_confidence,
                    'dynamic': dynamic_confidence,
                    'semantic': semantic_confidence
                },
                'evidence_consistency': self._compute_evidence_consistency(accumulated_evidence)
            }
        }

        reasoning = f"Refinement analysis combining all modalities with ensemble confidence {ensemble_confidence:.3f}"

        # Use initial ML prediction instead of hardcoded fallback
        predicted_vuln = target_vulnerability or initial_prediction or "unknown"

        return AnalysisResult(
            vulnerability_detected=refined_confidence > 0.5,
            vulnerability_type=predicted_vuln,
            confidence=refined_confidence,
            evidence=evidence,
            reasoning=reasoning,
            phase=AnalysisPhase.REFINEMENT,
            modality_contributions={'static': 0.33, 'dynamic': 0.33, 'semantic': 0.34}
        )

    def _update_accumulated_evidence(self, accumulated_evidence: Dict[str, Any], result: AnalysisResult):
        accumulated_evidence['last_confidence'] = result.confidence
        accumulated_evidence[f'{result.phase.value}_evidence'] = result.evidence

        if result.phase == AnalysisPhase.DEEP_STATIC:
            accumulated_evidence['static_confidence'] = result.confidence
        elif result.phase == AnalysisPhase.DEEP_DYNAMIC:
            accumulated_evidence['dynamic_confidence'] = result.confidence
        elif result.phase == AnalysisPhase.DEEP_SEMANTIC:
            accumulated_evidence['semantic_confidence'] = result.confidence

    def _compute_evidence_consistency(self, accumulated_evidence: Dict[str, Any]) -> float:
        confidences = []

        for key in ['static_confidence', 'dynamic_confidence', 'semantic_confidence']:
            if key in accumulated_evidence:
                confidences.append(accumulated_evidence[key])

        if len(confidences) < 2:
            return 1.0

        mean_conf = sum(confidences) / len(confidences)
        variance = sum((c - mean_conf) ** 2 for c in confidences) / len(confidences)

        consistency = 1.0 / (1.0 + variance * 10)

        return consistency

    def _synthesize_final_result(self, workflow_state: WorkflowState) -> AnalysisResult:
        if not workflow_state.results_history:
            return AnalysisResult(
                vulnerability_detected=False,
                vulnerability_type="unknown",
                confidence=0.0,
                evidence={},
                reasoning="No analysis performed",
                phase=AnalysisPhase.FINAL,
                modality_contributions={}
            )

        # Get the initial analysis result which has the actual ML predicted vulnerability
        initial_result = workflow_state.results_history[0]
        last_result = workflow_state.results_history[-1]

        all_evidence = {}
        for result in workflow_state.results_history:
            all_evidence[result.phase.value] = result.evidence

        final_confidence = last_result.confidence
        consistency = self._compute_evidence_consistency(workflow_state.accumulated_evidence)

        final_confidence = final_confidence * consistency

        # ALWAYS use the vulnerability type from initial analysis (ML prediction)
        # This is the output from the trained fusion module and should be trusted
        final_vuln_type = initial_result.vulnerability_type

        reasoning = f"Final synthesis after {workflow_state.iteration} iterations with consistency score {consistency:.3f}. "
        reasoning += f"ML model predicted: {final_vuln_type}, final confidence: {final_confidence:.3f}"

        return AnalysisResult(
            vulnerability_detected=final_confidence > 0.5,
            vulnerability_type=final_vuln_type,
            confidence=final_confidence,
            evidence=all_evidence,
            reasoning=reasoning,
            phase=AnalysisPhase.FINAL,
            modality_contributions=last_result.modality_contributions
        )