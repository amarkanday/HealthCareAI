"""
Feature monitoring class for input data quality and drift detection
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from typing import Dict, List, Tuple
import json


class FeatureMonitor:
    """Monitor input features for quality and drift"""
    
    def __init__(self, baseline_df: pd.DataFrame):
        """
        Initialize with baseline data (e.g., first 2 weeks of production)
        
        Args:
            baseline_df: DataFrame with baseline features
        """
        self.baseline_df = baseline_df.copy()
        self.baseline_stats = self._compute_baseline_stats()
        self.feature_columns = [col for col in baseline_df.columns 
                               if col not in ['patient_id', 'date']]
    
    def _compute_baseline_stats(self) -> Dict:
        """Compute reference statistics for each feature"""
        stats = {}
        
        for col in self.baseline_df.columns:
            if col in ['patient_id', 'date']:
                continue
            
            if self.baseline_df[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    'type': 'continuous',
                    'mean': float(self.baseline_df[col].mean()),
                    'std': float(self.baseline_df[col].std()),
                    'min': float(self.baseline_df[col].min()),
                    'max': float(self.baseline_df[col].max()),
                    'quantiles': {
                        k: float(v) for k, v in 
                        self.baseline_df[col].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_dict().items()
                    },
                    'null_rate': float(self.baseline_df[col].isnull().mean())
                }
            else:
                stats[col] = {
                    'type': 'categorical',
                    'value_counts': {
                        k: float(v) for k, v in 
                        self.baseline_df[col].value_counts(normalize=True).to_dict().items()
                    },
                    'cardinality': int(self.baseline_df[col].nunique()),
                    'null_rate': float(self.baseline_df[col].isnull().mean())
                }
        
        return stats
    
    def monitor_daily_data(self, current_df: pd.DataFrame) -> Dict:
        """
        Run all monitoring checks on current data
        
        Args:
            current_df: DataFrame with current day's features
            
        Returns:
            dict with feature_metrics, drift_tests, and alerts
        """
        feature_metrics = {}
        drift_tests = {}
        alerts = []
        
        for col in self.feature_columns:
            if col not in current_df.columns:
                alerts.append({
                    'severity': 'HIGH',
                    'feature': col,
                    'message': f'Missing feature: {col}'
                })
                continue
            
            baseline = self.baseline_stats[col]
            
            if baseline['type'] == 'continuous':
                metrics, col_alerts = self._check_continuous_feature(
                    col, current_df[col], baseline
                )
                feature_metrics[col] = metrics
                
                drift_test = self._drift_test_continuous(
                    self.baseline_df[col], current_df[col]
                )
                drift_tests[col] = drift_test
                
                # Add drift alert if detected
                if drift_test.get('drifted', False):
                    alerts.append({
                        'severity': drift_test['severity'],
                        'feature': col,
                        'message': f"Distribution drift detected (PSI: {drift_test['psi']:.3f}, KS p-value: {drift_test['p_value']:.4f})"
                    })
                
                alerts.extend(col_alerts)
                
            else:  # categorical
                metrics, col_alerts = self._check_categorical_feature(
                    col, current_df[col], baseline
                )
                feature_metrics[col] = metrics
                
                drift_test = self._drift_test_categorical(
                    self.baseline_df[col], current_df[col]
                )
                drift_tests[col] = drift_test
                
                if drift_test.get('drifted', False):
                    alerts.append({
                        'severity': 'MEDIUM',
                        'feature': col,
                        'message': f"Distribution drift detected (Chi-square p-value: {drift_test['p_value']:.4f})"
                    })
                
                alerts.extend(col_alerts)
        
        return {
            'date': str(current_df['date'].iloc[0]) if 'date' in current_df.columns else None,
            'feature_metrics': feature_metrics,
            'drift_tests': drift_tests,
            'alerts': alerts,
            'summary': {
                'total_features': len(self.feature_columns),
                'features_with_drift': sum(1 for t in drift_tests.values() if t.get('drifted', False)),
                'high_severity_alerts': sum(1 for a in alerts if a['severity'] == 'HIGH'),
                'total_alerts': len(alerts)
            }
        }
    
    def _check_continuous_feature(self, name: str, series: pd.Series, baseline: Dict) -> Tuple[Dict, List]:
        """Check continuous feature for anomalies"""
        current_stats = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'null_rate': float(series.isnull().mean())
        }
        
        alerts = []
        
        # Null rate check
        null_increase = current_stats['null_rate'] - baseline['null_rate']
        if null_increase > 0.05:  # 5pp increase
            alerts.append({
                'severity': 'HIGH',
                'feature': name,
                'message': f'Null rate increased by {null_increase:.1%} (baseline: {baseline["null_rate"]:.1%}, current: {current_stats["null_rate"]:.1%})'
            })
        
        # Mean shift check (z-score)
        if baseline['std'] > 0 and not np.isnan(current_stats['mean']):
            z_score = abs(current_stats['mean'] - baseline['mean']) / baseline['std']
            if z_score > 3:
                alerts.append({
                    'severity': 'MEDIUM',
                    'feature': name,
                    'message': f'Mean shifted by {z_score:.2f} std devs (baseline: {baseline["mean"]:.2f}, current: {current_stats["mean"]:.2f})'
                })
        
        return current_stats, alerts
    
    def _check_categorical_feature(self, name: str, series: pd.Series, baseline: Dict) -> Tuple[Dict, List]:
        """Check categorical feature for anomalies"""
        current_value_counts = series.value_counts(normalize=True).to_dict()
        current_stats = {
            'value_counts': {k: float(v) for k, v in current_value_counts.items()},
            'cardinality': int(series.nunique()),
            'null_rate': float(series.isnull().mean())
        }
        
        alerts = []
        
        # New categories
        baseline_categories = set(baseline['value_counts'].keys())
        current_categories = set(current_value_counts.keys())
        new_categories = current_categories - baseline_categories
        
        if new_categories:
            alerts.append({
                'severity': 'MEDIUM',
                'feature': name,
                'message': f'New categories detected: {new_categories}'
            })
        
        return current_stats, alerts
    
    def _drift_test_continuous(self, baseline_series: pd.Series, current_series: pd.Series) -> Dict:
        """Kolmogorov-Smirnov test + PSI for continuous features"""
        baseline_clean = baseline_series.dropna()
        current_clean = current_series.dropna()
        
        if len(baseline_clean) == 0 or len(current_clean) == 0:
            return {'test': 'KS', 'drifted': False, 'reason': 'insufficient_data'}
        
        statistic, p_value = ks_2samp(baseline_clean, current_clean)
        psi = self._compute_psi(baseline_clean, current_clean)
        
        drifted = p_value < 0.01 or psi > 0.25
        
        return {
            'test': 'KS',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'psi': float(psi),
            'drifted': drifted,
            'severity': 'HIGH' if psi > 0.25 else 'MEDIUM' if psi > 0.1 else 'LOW'
        }
    
    def _drift_test_categorical(self, baseline_series: pd.Series, current_series: pd.Series) -> Dict:
        """Chi-square test for categorical features"""
        baseline_counts = baseline_series.value_counts()
        current_counts = current_series.value_counts()
        
        all_categories = set(baseline_counts.index) | set(current_counts.index)
        baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
        current_aligned = [current_counts.get(cat, 0) for cat in all_categories]
        
        if sum(baseline_aligned) == 0 or sum(current_aligned) == 0:
            return {'test': 'ChiSquare', 'drifted': False, 'reason': 'insufficient_data'}
        
        contingency_table = np.array([baseline_aligned, current_aligned])
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        drifted = p_value < 0.01
        
        return {
            'test': 'ChiSquare',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'dof': int(dof),
            'drifted': drifted
        }
    
    def _compute_psi(self, baseline: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """
        Compute Population Stability Index
        PSI = sum((current% - baseline%) * ln(current% / baseline%))
        """
        try:
            _, bin_edges = np.histogram(baseline, bins=bins)
            
            baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
            current_hist, _ = np.histogram(current, bins=bin_edges)
            
            # Convert to percentages (add small epsilon to avoid log(0))
            baseline_pct = (baseline_hist + 1e-6) / (baseline_hist.sum() + bins * 1e-6)
            current_pct = (current_hist + 1e-6) / (current_hist.sum() + bins * 1e-6)
            
            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            
            return float(psi)
        except Exception as e:
            return 0.0
    
    def save_monitoring_results(self, results: Dict, filename: str):
        """Save monitoring results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
    
    def load_monitoring_results(self, filename: str) -> Dict:
        """Load monitoring results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)

