import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VulnerabilityMetrics:
    def __init__(self, vulnerability_types: Optional[List[str]] = None):
        self.vulnerability_types = vulnerability_types or [
            'reentrancy', 'overflow', 'underflow', 'access_control',
            'unchecked_call', 'timestamp_dependency', 'tx_origin',
            'delegatecall', 'self_destruct', 'gas_limit'
        ]

        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []
        self.confidences = []
        self.vulnerability_predictions = {v: [] for v in self.vulnerability_types}
        self.vulnerability_labels = {v: [] for v in self.vulnerability_types}

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        confidences: Optional[torch.Tensor] = None,
        vuln_preds: Optional[Dict[str, torch.Tensor]] = None,
        vuln_labels: Optional[Dict[str, torch.Tensor]] = None
    ):
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())

        if confidences is not None:
            self.confidences.extend(confidences.cpu().numpy().tolist())

        if vuln_preds and vuln_labels:
            for vuln_type in self.vulnerability_types:
                if vuln_type in vuln_preds:
                    self.vulnerability_predictions[vuln_type].extend(
                        vuln_preds[vuln_type].cpu().numpy().tolist()
                    )
                    self.vulnerability_labels[vuln_type].extend(
                        vuln_labels[vuln_type].cpu().numpy().tolist()
                    )

    def compute_metrics(self) -> Dict:
        predictions = np.array(self.predictions)
        labels = np.array(self.labels)

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': false_positive_rate,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }

        if self.confidences:
            confidences = np.array(self.confidences)
            try:
                auc_score = roc_auc_score(labels, confidences)
                metrics['auc'] = auc_score
            except ValueError:
                logger.warning("Could not compute AUC score")

        vuln_metrics = {}
        for vuln_type in self.vulnerability_types:
            if (self.vulnerability_predictions[vuln_type] and
                self.vulnerability_labels[vuln_type]):

                vuln_preds = np.array(self.vulnerability_predictions[vuln_type])
                vuln_labels = np.array(self.vulnerability_labels[vuln_type])

                if len(np.unique(vuln_labels)) > 1:
                    vuln_tn, vuln_fp, vuln_fn, vuln_tp = confusion_matrix(
                        vuln_labels, vuln_preds
                    ).ravel()

                    vuln_metrics[vuln_type] = {
                        'precision': vuln_tp / (vuln_tp + vuln_fp) if (vuln_tp + vuln_fp) > 0 else 0,
                        'recall': vuln_tp / (vuln_tp + vuln_fn) if (vuln_tp + vuln_fn) > 0 else 0,
                        'fpr': vuln_fp / (vuln_fp + vuln_tn) if (vuln_fp + vuln_tn) > 0 else 0
                    }

        if vuln_metrics:
            metrics['vulnerability_metrics'] = vuln_metrics

        return metrics


class ConfidenceCalibrator:
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins

    def compute_calibration_error(
        self,
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Tuple[float, Dict]:

        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        bin_stats = {}

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.astype(float).mean()

            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).astype(float).mean()
                avg_confidence_in_bin = confidences[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_stats[f'bin_{i}'] = {
                    'range': (bin_lower, bin_upper),
                    'accuracy': accuracy_in_bin,
                    'avg_confidence': avg_confidence_in_bin,
                    'count': in_bin.sum()
                }

        return ece, bin_stats


def compute_adaptive_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.12
) -> float:

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    fprs = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fprs.append(fpr)

    fprs = np.array(fprs)

    valid_idx = np.where(fprs <= target_fpr)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(recalls[valid_idx])]
        return thresholds[best_idx]

    return thresholds[np.argmin(np.abs(fprs - target_fpr))]


def evaluate_multimodal_performance(
    static_scores: np.ndarray,
    dynamic_scores: np.ndarray,
    semantic_scores: np.ndarray,
    fusion_scores: np.ndarray,
    labels: np.ndarray,
    vulnerability_type: Optional[str] = None
) -> Dict:

    modality_weights = {
        'reentrancy': {'static': 0.2, 'dynamic': 0.6, 'semantic': 0.2},
        'access_control': {'static': 0.7, 'dynamic': 0.1, 'semantic': 0.2},
        'overflow': {'static': 0.3, 'dynamic': 0.4, 'semantic': 0.3},
        'default': {'static': 0.33, 'dynamic': 0.33, 'semantic': 0.34}
    }

    weights = modality_weights.get(vulnerability_type, modality_weights['default'])

    weighted_scores = (
        weights['static'] * static_scores +
        weights['dynamic'] * dynamic_scores +
        weights['semantic'] * semantic_scores
    )

    threshold = compute_adaptive_threshold(weighted_scores, labels, target_fpr=0.12)
    predictions = (weighted_scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        'modality_weights': weights,
        'threshold': threshold,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'fusion_improvement': {
            'vs_static': compute_improvement(static_scores, fusion_scores, labels),
            'vs_dynamic': compute_improvement(dynamic_scores, fusion_scores, labels),
            'vs_semantic': compute_improvement(semantic_scores, fusion_scores, labels)
        }
    }


def compute_improvement(
    baseline_scores: np.ndarray,
    fusion_scores: np.ndarray,
    labels: np.ndarray
) -> float:

    baseline_auc = roc_auc_score(labels, baseline_scores)
    fusion_auc = roc_auc_score(labels, fusion_scores)

    return ((fusion_auc - baseline_auc) / baseline_auc) * 100