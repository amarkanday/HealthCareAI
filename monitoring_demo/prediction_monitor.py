"""
Prediction monitoring class for model output drift detection
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from typing import Dict, List
import json


class PredictionMonitor:
    """Monitor model predictions for drift and performance degradation"""
    
    def __init__(self, baseline_predictions: pd.DataFrame):
        """
        Initialize with baseline predictions (e.g., first 2 weeks of production)
        
        Args:
            baseline_predictions: DataFrame with baseline predictions
                Must have columns: uplift_score, model_treated_prob, model_control_prob, recommendation
        """
        self.baseline_predictions = baseline_predictions.copy()
        self.baseline_stats = self._compute_baseline_stats()
    
    def _compute_baseline_stats(self) -> Dict:
        """Compute baseline statistics for predictions"""
        return {
            'mean_uplift': float(self.baseline_predictions['uplift_score'].mean()),
            'std_uplift': float(self.baseline_predictions['uplift_score'].std()),
            'quantiles': {
                str(k): float(v) for k, v in 
                self.baseline_predictions['uplift_score'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict().items()
            },
            'recommendation_rate': float(self.baseline_predictions['recommendation'].mean()),
            'extreme_low': float((self.baseline_predictions['uplift_score'] < 0).mean()),
            'extreme_high': float((self.baseline_predictions['uplift_score'] > 1).mean()),
            'mean_treated_prob': float(self.baseline_predictions['model_treated_prob'].mean()),
            'mean_control_prob': float(self.baseline_predictions['model_control_prob'].mean())
        }
    
    def monitor_daily_predictions(self, current_predictions: pd.DataFrame) -> Dict:
        """
        Monitor today's predictions for drift
        
        Args:
            current_predictions: DataFrame with columns:
                - patient_id
                - uplift_score
                - model_treated_prob (treated model probability)
                - model_control_prob (control model probability)
                - recommendation (binary)
        
        Returns:
            dict with monitoring results and alerts
        """
        current_stats = {
            'mean_uplift': float(current_predictions['uplift_score'].mean()),
            'std_uplift': float(current_predictions['uplift_score'].std()),
            'quantiles': {
                str(k): float(v) for k, v in 
                current_predictions['uplift_score'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_dict().items()
            },
            'recommendation_rate': float(current_predictions['recommendation'].mean()),
            'extreme_low': float((current_predictions['uplift_score'] < 0).mean()),
            'extreme_high': float((current_predictions['uplift_score'] > 1).mean()),
            'mean_treated_prob': float(current_predictions['model_treated_prob'].mean()),
            'mean_control_prob': float(current_predictions['model_control_prob'].mean())
        }
        
        alerts = []
        
        # Mean uplift shift
        baseline_mean = self.baseline_stats['mean_uplift']
        baseline_std = self.baseline_stats['std_uplift']
        
        if baseline_std > 0:
            z_score = abs(current_stats['mean_uplift'] - baseline_mean) / baseline_std
            
            if z_score > 2:
                alerts.append({
                    'severity': 'HIGH' if z_score > 3 else 'MEDIUM',
                    'metric': 'mean_uplift',
                    'message': f'Mean uplift score shifted by {z_score:.2f} std devs (baseline: {baseline_mean:.4f}, current: {current_stats["mean_uplift"]:.4f})'
                })
        
        # Recommendation rate shift
        rec_rate_change = abs(current_stats['recommendation_rate'] - self.baseline_stats['recommendation_rate'])
        if rec_rate_change > 0.1:  # 10pp change
            alerts.append({
                'severity': 'MEDIUM',
                'metric': 'recommendation_rate',
                'message': f'Recommendation rate shifted by {rec_rate_change:.1%} (baseline: {self.baseline_stats["recommendation_rate"]:.1%}, current: {current_stats["recommendation_rate"]:.1%})'
            })
        
        # Extreme predictions
        if current_stats['extreme_low'] > 0.01 or current_stats['extreme_high'] > 0.01:
            alerts.append({
                'severity': 'HIGH',
                'metric': 'extreme_predictions',
                'message': f'High rate of extreme predictions (low: {current_stats["extreme_low"]:.2%}, high: {current_stats["extreme_high"]:.2%})'
            })
        
        # Distribution shift (KS test on uplift scores)
        statistic, p_value = ks_2samp(
            self.baseline_predictions['uplift_score'],
            current_predictions['uplift_score']
        )
        
        if p_value < 0.01:
            alerts.append({
                'severity': 'MEDIUM',
                'metric': 'distribution_shift',
                'message': f'Significant shift in uplift score distribution (KS p-value: {p_value:.4f})'
            })
        
        # Model output stability check
        treated_prob_shift = abs(current_stats['mean_treated_prob'] - self.baseline_stats['mean_treated_prob'])
        control_prob_shift = abs(current_stats['mean_control_prob'] - self.baseline_stats['mean_control_prob'])
        
        if treated_prob_shift > 0.1 or control_prob_shift > 0.1:
            alerts.append({
                'severity': 'MEDIUM',
                'metric': 'model_output_shift',
                'message': f'Model output probabilities shifted (treated: {treated_prob_shift:.3f}, control: {control_prob_shift:.3f})'
            })
        
        return {
            'date': str(current_predictions['date'].iloc[0]) if 'date' in current_predictions.columns else None,
            'current_stats': current_stats,
            'baseline_stats': self.baseline_stats,
            'drift_test': {
                'ks_statistic': float(statistic),
                'ks_p_value': float(p_value),
                'drifted': p_value < 0.01
            },
            'alerts': alerts,
            'summary': {
                'total_predictions': len(current_predictions),
                'high_severity_alerts': sum(1 for a in alerts if a['severity'] == 'HIGH'),
                'total_alerts': len(alerts)
            }
        }
    
    def monitor_model_performance(
        self, 
        predictions_with_outcomes: pd.DataFrame
    ) -> Dict:
        """
        Monitor model performance once outcomes are available
        Run weekly
        
        Args:
            predictions_with_outcomes: DataFrame with:
                - patient_id
                - uplift_score
                - prediction_date
                - actual_outcome (0/1, discontinuation within 30d)
                - treatment_received (0/1, did FRM intervene)
        
        Returns:
            dict with performance metrics
        """
        results = {}
        
        treated = predictions_with_outcomes[predictions_with_outcomes['treatment_received'] == 1]
        control = predictions_with_outcomes[predictions_with_outcomes['treatment_received'] == 0]
        
        if len(treated) > 50 and len(control) > 50:
            # Observed treatment effect
            treated_outcome_rate = treated['actual_outcome'].mean()
            control_outcome_rate = control['actual_outcome'].mean()
            observed_effect = control_outcome_rate - treated_outcome_rate  # Reduction in discontinuation
            
            results['treated_outcome_rate'] = float(treated_outcome_rate)
            results['control_outcome_rate'] = float(control_outcome_rate)
            results['observed_treatment_effect'] = float(observed_effect)
            
            # Predicted vs actual for treated patients
            predicted_treated = treated['uplift_score'].mean()
            results['predicted_uplift_treated'] = float(predicted_treated)
            results['prediction_error'] = float(abs(observed_effect - predicted_treated))
            
            # Simple Qini coefficient approximation
            qini = self._compute_simple_qini(predictions_with_outcomes)
            results['qini_coefficient'] = float(qini)
            
            # Calibration check (for treated model)
            from sklearn.metrics import brier_score_loss
            treated_with_pred = treated.dropna(subset=['actual_outcome'])
            if len(treated_with_pred) > 0:
                brier = brier_score_loss(
                    treated_with_pred['actual_outcome'],
                    treated_with_pred['uplift_score'].clip(0, 1)
                )
                results['brier_score'] = float(brier)
            
            # Alerts
            alerts = []
            
            if observed_effect < 0:
                alerts.append({
                    'severity': 'HIGH',
                    'metric': 'treatment_effect',
                    'message': f'Negative treatment effect observed: {observed_effect:.3f} (treatment worse than control)'
                })
            elif observed_effect < 0.05:
                alerts.append({
                    'severity': 'MEDIUM',
                    'metric': 'treatment_effect',
                    'message': f'Low treatment effect: {observed_effect:.3f} (below 5pp threshold)'
                })
            
            if qini < 0.05:
                alerts.append({
                    'severity': 'MEDIUM',
                    'metric': 'qini_coefficient',
                    'message': f'Low Qini coefficient: {qini:.3f} (model not ranking well)'
                })
            
            if 'brier_score' in results and results['brier_score'] > 0.25:
                alerts.append({
                    'severity': 'MEDIUM',
                    'metric': 'calibration',
                    'message': f'Poor calibration (Brier score: {results["brier_score"]:.3f})'
                })
            
            results['alerts'] = alerts
            results['summary'] = {
                'n_treated': len(treated),
                'n_control': len(control),
                'high_severity_alerts': sum(1 for a in alerts if a['severity'] == 'HIGH'),
                'total_alerts': len(alerts)
            }
        else:
            results['message'] = 'Insufficient data for performance monitoring'
            results['n_treated'] = len(treated)
            results['n_control'] = len(control)
        
        return results
    
    def _compute_simple_qini(self, df: pd.DataFrame) -> float:
        """
        Simplified Qini coefficient computation
        Measures how well the model ranks patients by uplift
        """
        # Sort by uplift score descending
        df_sorted = df.sort_values('uplift_score', ascending=False).reset_index(drop=True)
        
        n = len(df_sorted)
        treated = df_sorted['treatment_received'].values
        outcome = df_sorted['actual_outcome'].values
        
        # Cumulative gains
        cumulative_gains = []
        
        for i in range(1, n+1):
            subset = df_sorted.iloc[:i]
            treated_subset = subset[subset['treatment_received'] == 1]
            control_subset = subset[subset['treatment_received'] == 0]
            
            if len(treated_subset) > 0 and len(control_subset) > 0:
                treated_rate = treated_subset['actual_outcome'].mean()
                control_rate = control_subset['actual_outcome'].mean()
                gain = control_rate - treated_rate  # Lower outcome is better
                cumulative_gains.append(gain * i / n)
            else:
                cumulative_gains.append(0)
        
        # Qini = area under curve
        qini = np.mean(cumulative_gains) if cumulative_gains else 0
        
        return qini
    
    def save_monitoring_results(self, results: Dict, filename: str):
        """Save monitoring results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_monitoring_results(self, filename: str) -> Dict:
        """Load monitoring results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

