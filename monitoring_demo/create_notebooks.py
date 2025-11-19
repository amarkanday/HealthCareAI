"""
Create monitoring demonstration notebooks
"""

import json

def create_feature_monitoring_notebook():
    """Create the feature monitoring notebook"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Feature Monitoring Demo\n",
                    "## Input Data Quality and Drift Detection\n",
                    "\n",
                    "This notebook demonstrates comprehensive feature monitoring including:\n",
                    "- Data quality checks (null rates, ranges, types)\n",
                    "- Distribution drift detection (KS test, PSI, Chi-square)\n",
                    "- Statistical alerting\n",
                    "- Visualization of drift\n",
                    "\n",
                    "**Use Case:** Monitor daily patient features for the Intervention Recommendation Engine"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from datetime import datetime, timedelta\n",
                    "from feature_monitor import FeatureMonitor\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_style('whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                    "\n",
                    "print(\"✓ Libraries imported successfully\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Baseline and Daily Data\n",
                    "\n",
                    "- Days 1-14: Baseline (stable distribution)\n",
                    "- Days 15-21: Drift period (mean shifts, distribution changes)\n",
                    "- Days 22-30: Data quality issues (increased null rates)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load baseline data (first 14 days)\n",
                    "baseline_df = pd.read_parquet('./data/baseline_features.parquet')\n",
                    "\n",
                    "print(f\"Baseline dataset: {len(baseline_df):,} rows, {len(baseline_df.columns)} columns\")\n",
                    "print(f\"\\nFeatures:\")\n",
                    "for col in baseline_df.columns:\n",
                    "    if col not in ['patient_id', 'date']:\n",
                    "        print(f\"  - {col}: {baseline_df[col].dtype}\")\n",
                    "\n",
                    "print(f\"\\nBaseline summary:\")\n",
                    "baseline_df.describe()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load all 30 days of data\n",
                    "import glob\n",
                    "feature_files = sorted(glob.glob('./data/patient_features_*.parquet'))\n",
                    "\n",
                    "daily_data = []\n",
                    "for file in feature_files:\n",
                    "    df = pd.read_parquet(file)\n",
                    "    daily_data.append(df)\n",
                    "\n",
                    "print(f\"✓ Loaded {len(daily_data)} days of data\")\n",
                    "print(f\"  - Each day: ~{len(daily_data[0]):,} patients\")\n",
                    "print(f\"\\nSample from Day 1:\")\n",
                    "daily_data[0].head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Initialize Feature Monitor"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize monitor with baseline\n",
                    "monitor = FeatureMonitor(baseline_df)\n",
                    "\n",
                    "print(f\"✓ Monitor initialized with baseline data\")\n",
                    "print(f\"  Features tracked: {len(monitor.feature_columns)}\")\n",
                    "print(f\"  Features: {monitor.feature_columns}\")\n",
                    "print(f\"\\nBaseline statistics sample (age):\")\n",
                    "print(json.dumps(monitor.baseline_stats['age'], indent=2))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Run Daily Monitoring for 30 Days"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run monitoring for all 30 days\n",
                    "monitoring_results = []\n",
                    "\n",
                    "for day_idx, df in enumerate(daily_data):\n",
                    "    result = monitor.monitor_daily_data(df)\n",
                    "    result['day'] = day_idx + 1\n",
                    "    monitoring_results.append(result)\n",
                    "\n",
                    "print(f\"✓ Monitoring completed for {len(monitoring_results)} days\")\n",
                    "print(f\"\\nDaily Summary:\")\n",
                    "print(\"Day | Status | Drift | Alerts\")\n",
                    "print(\"-\" * 40)\n",
                    "for result in monitoring_results:\n",
                    "    summary = result['summary']\n",
                    "    status = '🔴' if summary['high_severity_alerts'] > 0 else '🟡' if summary['total_alerts'] > 0 else '🟢'\n",
                    "    print(f\" {result['day']:2d} |   {status}    | {summary['features_with_drift']:2d}/{summary['total_features']}  |   {summary['total_alerts']:2d}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visualize Monitoring Metrics Over Time"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract time series of key metrics\n",
                    "days = [r['day'] for r in monitoring_results]\n",
                    "alerts_by_day = [r['summary']['total_alerts'] for r in monitoring_results]\n",
                    "drift_by_day = [r['summary']['features_with_drift'] for r in monitoring_results]\n",
                    "\n",
                    "# Create comprehensive monitoring dashboard\n",
                    "fig, axes = plt.subplots(3, 2, figsize=(16, 12))\n",
                    "fig.suptitle('Feature Monitoring Dashboard - 30 Days', fontsize=16, fontweight='bold')\n",
                    "\n",
                    "# Plot 1: Alerts over time\n",
                    "axes[0, 0].plot(days, alerts_by_day, marker='o', color='red', linewidth=2)\n",
                    "axes[0, 0].axvline(x=15, color='orange', linestyle='--', alpha=0.5, label='Drift introduced')\n",
                    "axes[0, 0].axvline(x=22, color='purple', linestyle='--', alpha=0.5, label='Quality issues')\n",
                    "axes[0, 0].set_xlabel('Day')\n",
                    "axes[0, 0].set_ylabel('Total Alerts')\n",
                    "axes[0, 0].set_title('Daily Alert Count')\n",
                    "axes[0, 0].legend()\n",
                    "axes[0, 0].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 2: Features with drift\n",
                    "axes[0, 1].plot(days, drift_by_day, marker='s', color='orange', linewidth=2)\n",
                    "axes[0, 1].axvline(x=15, color='orange', linestyle='--', alpha=0.5)\n",
                    "axes[0, 1].axvline(x=22, color='purple', linestyle='--', alpha=0.5)\n",
                    "axes[0, 1].set_xlabel('Day')\n",
                    "axes[0, 1].set_ylabel('Features with Drift')\n",
                    "axes[0, 1].set_title('Features Experiencing Drift')\n",
                    "axes[0, 1].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 3: PSI for discontinuation_risk_score over time\n",
                    "psi_risk = [r['drift_tests']['discontinuation_risk_score']['psi'] for r in monitoring_results]\n",
                    "axes[1, 0].plot(days, psi_risk, marker='o', color='blue', linewidth=2)\n",
                    "axes[1, 0].axhline(y=0.1, color='yellow', linestyle='--', label='Slight drift (PSI=0.1)')\n",
                    "axes[1, 0].axhline(y=0.25, color='red', linestyle='--', label='Major drift (PSI=0.25)')\n",
                    "axes[1, 0].axvline(x=15, color='orange', linestyle='--', alpha=0.5)\n",
                    "axes[1, 0].set_xlabel('Day')\n",
                    "axes[1, 0].set_ylabel('PSI')\n",
                    "axes[1, 0].set_title('Discontinuation Risk Score - PSI Over Time')\n",
                    "axes[1, 0].legend()\n",
                    "axes[1, 0].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 4: Mean age over time\n",
                    "mean_age = [r['feature_metrics']['age']['mean'] for r in monitoring_results]\n",
                    "baseline_mean_age = monitor.baseline_stats['age']['mean']\n",
                    "axes[1, 1].plot(days, mean_age, marker='o', color='green', linewidth=2, label='Current')\n",
                    "axes[1, 1].axhline(y=baseline_mean_age, color='blue', linestyle='--', label='Baseline')\n",
                    "axes[1, 1].axvline(x=15, color='orange', linestyle='--', alpha=0.5)\n",
                    "axes[1, 1].set_xlabel('Day')\n",
                    "axes[1, 1].set_ylabel('Mean Age')\n",
                    "axes[1, 1].set_title('Patient Age - Mean Over Time')\n",
                    "axes[1, 1].legend()\n",
                    "axes[1, 1].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 5: Null rate for age over time\n",
                    "null_rate_age = [r['feature_metrics']['age']['null_rate'] for r in monitoring_results]\n",
                    "axes[2, 0].plot(days, [x * 100 for x in null_rate_age], marker='o', color='red', linewidth=2)\n",
                    "axes[2, 0].axhline(y=5, color='yellow', linestyle='--', label='Baseline (5%)')\n",
                    "axes[2, 0].axhline(y=15, color='red', linestyle='--', label='Alert threshold (15%)')\n",
                    "axes[2, 0].axvline(x=22, color='purple', linestyle='--', alpha=0.5)\n",
                    "axes[2, 0].set_xlabel('Day')\n",
                    "axes[2, 0].set_ylabel('Null Rate (%)')\n",
                    "axes[2, 0].set_title('Age Feature - Null Rate Over Time')\n",
                    "axes[2, 0].legend()\n",
                    "axes[2, 0].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 6: Chi-square p-value for region\n",
                    "p_values_region = [r['drift_tests']['region']['p_value'] for r in monitoring_results]\n",
                    "axes[2, 1].plot(days, p_values_region, marker='s', color='purple', linewidth=2)\n",
                    "axes[2, 1].axhline(y=0.01, color='red', linestyle='--', label='Significance threshold (p=0.01)')\n",
                    "axes[2, 1].axvline(x=15, color='orange', linestyle='--', alpha=0.5)\n",
                    "axes[2, 1].set_xlabel('Day')\n",
                    "axes[2, 1].set_ylabel('Chi-square p-value')\n",
                    "axes[2, 1].set_title('Region (Categorical) - Distribution Change')\n",
                    "axes[2, 1].set_yscale('log')\n",
                    "axes[2, 1].legend()\n",
                    "axes[2, 1].grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('./feature_monitoring_dashboard.png', dpi=150, bbox_inches='tight')\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"\\n✓ Monitoring visualizations complete\")\n",
                    "print(\"  Saved: ./feature_monitoring_dashboard.png\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Detailed Alert Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract all alerts\n",
                    "all_alerts = []\n",
                    "for result in monitoring_results:\n",
                    "    for alert in result['alerts']:\n",
                    "        all_alerts.append({\n",
                    "            'day': result['day'],\n",
                    "            'severity': alert['severity'],\n",
                    "            'feature': alert['feature'],\n",
                    "            'message': alert['message']\n",
                    "        })\n",
                    "\n",
                    "alerts_df = pd.DataFrame(all_alerts)\n",
                    "\n",
                    "print(f\"Total alerts: {len(alerts_df)}\")\n",
                    "print(f\"\\nAlerts by severity:\")\n",
                    "print(alerts_df['severity'].value_counts())\n",
                    "print(f\"\\nAlerts by feature:\")\n",
                    "print(alerts_df['feature'].value_counts())\n",
                    "\n",
                    "print(f\"\\n\" + \"=\"*80)\n",
                    "print(\"HIGH SEVERITY ALERTS:\")\n",
                    "print(\"=\"*80)\n",
                    "high_alerts = alerts_df[alerts_df['severity'] == 'HIGH']\n",
                    "for idx, row in high_alerts.head(10).iterrows():\n",
                    "    print(f\"Day {row['day']:2d} | {row['feature']:30s} | {row['message']}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Distribution Comparison: Baseline vs Day 20"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compare distributions for key continuous features\n",
                    "day_20 = daily_data[19]  # Index 19 = Day 20\n",
                    "\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "fig.suptitle('Distribution Comparison: Baseline vs Day 20 (Drift Period)', fontsize=14, fontweight='bold')\n",
                    "\n",
                    "continuous_features = ['age', 'discontinuation_risk_score', 'adherence_mpr', 'days_on_therapy']\n",
                    "\n",
                    "for idx, feature in enumerate(continuous_features):\n",
                    "    ax = axes[idx // 2, idx % 2]\n",
                    "    \n",
                    "    # Plot histograms\n",
                    "    ax.hist(baseline_df[feature].dropna(), bins=30, alpha=0.5, label='Baseline', color='blue', density=True)\n",
                    "    ax.hist(day_20[feature].dropna(), bins=30, alpha=0.5, label='Day 20', color='red', density=True)\n",
                    "    \n",
                    "    # Add vertical lines for means\n",
                    "    ax.axvline(baseline_df[feature].mean(), color='blue', linestyle='--', linewidth=2)\n",
                    "    ax.axvline(day_20[feature].mean(), color='red', linestyle='--', linewidth=2)\n",
                    "    \n",
                    "    ax.set_xlabel(feature.replace('_', ' ').title())\n",
                    "    ax.set_ylabel('Density')\n",
                    "    ax.legend()\n",
                    "    ax.grid(True, alpha=0.3)\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('./distribution_comparison.png', dpi=150, bbox_inches='tight')\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"✓ Distribution comparison saved\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Summary & Key Findings\n",
                    "\n",
                    "### Detection Results:\n",
                    "\n",
                    "✅ **Successfully Detected:**\n",
                    "- Distribution drift in continuous features (KS test + PSI)\n",
                    "- Categorical distribution shifts (Chi-square test)\n",
                    "- Data quality degradation (null rate increases)\n",
                    "- Mean shifts beyond 3 standard deviations\n",
                    "\n",
                    "### Production Recommendations:\n",
                    "\n",
                    "1. **Daily Monitoring:** Run after feature ETL completes\n",
                    "2. **Alerting:** HIGH severity → PagerDuty, MEDIUM → Slack\n",
                    "3. **Model Retraining:** Trigger if drift persists >3 days\n",
                    "4. **Data Quality:** Investigate upstream if null rate >15%\n",
                    "5. **Baseline Refresh:** Update quarterly or after major changes"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Final summary\n",
                    "print(\"=\"*80)\n",
                    "print(\"FEATURE MONITORING SUMMARY\")\n",
                    "print(\"=\"*80)\n",
                    "print(f\"\\nMonitoring Period: 30 days\")\n",
                    "print(f\"Baseline: Days 1-14\")\n",
                    "print(f\"Features Monitored: {len(monitor.feature_columns)}\")\n",
                    "print(f\"\\nTotal Alerts: {len(alerts_df)}\")\n",
                    "print(f\"  - HIGH: {len(alerts_df[alerts_df['severity'] == 'HIGH'])}\")\n",
                    "print(f\"  - MEDIUM: {len(alerts_df[alerts_df['severity'] == 'MEDIUM'])}\")\n",
                    "if 'LOW' in alerts_df['severity'].values:\n",
                    "    print(f\"  - LOW: {len(alerts_df[alerts_df['severity'] == 'LOW'])}\")\n",
                    "print(f\"\\nDays with Alerts: {len([r for r in monitoring_results if r['summary']['total_alerts'] > 0])}\")\n",
                    "print(f\"Days with High Severity: {len([r for r in monitoring_results if r['summary']['high_severity_alerts'] > 0])}\")\n",
                    "print(f\"\\nMax Features with Drift (single day): {max(drift_by_day)}\")\n",
                    "print(f\"Avg Features with Drift: {np.mean(drift_by_day):.1f}\")\n",
                    "print(\"\\n\" + \"=\"*80)\n",
                    "print(\"✓ Feature monitoring demo complete!\")\n",
                    "print(\"=\"*80)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def create_prediction_monitoring_notebook():
    """Create the prediction monitoring notebook"""
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Model Output Monitoring Demo\n",
                    "## Prediction Drift and Performance Detection\n",
                    "\n",
                    "This notebook demonstrates model output monitoring including:\n",
                    "- Prediction distribution drift detection\n",
                    "- Recommendation rate stability\n",
                    "- Model output anomalies\n",
                    "- Performance monitoring (when outcomes available)\n",
                    "\n",
                    "**Use Case:** Monitor Random Forest uplift model predictions for the Intervention Recommendation Engine"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "from datetime import datetime, timedelta\n",
                    "from prediction_monitor import PredictionMonitor\n",
                    "import warnings\n",
                    "warnings.filterwarnings('ignore')\n",
                    "\n",
                    "# Set style\n",
                    "sns.set_style('whitegrid')\n",
                    "plt.rcParams['figure.figsize'] = (12, 6)\n",
                    "\n",
                    "print(\"✓ Libraries imported successfully\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Load Baseline and Daily Predictions\n",
                    "\n",
                    "Predictions from Random Forest T-Learner model (two models for uplift estimation)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load baseline predictions (first 14 days)\n",
                    "baseline_predictions = pd.read_parquet('./data/baseline_predictions.parquet')\n",
                    "\n",
                    "print(f\"Baseline predictions: {len(baseline_predictions):,} rows\")\n",
                    "print(f\"\\nColumns:\")\n",
                    "print(list(baseline_predictions.columns))\n",
                    "print(f\"\\nSummary statistics:\")\n",
                    "baseline_predictions[['model_treated_prob', 'model_control_prob', 'uplift_score', 'recommendation']].describe()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Load all 30 days of predictions\n",
                    "import glob\n",
                    "prediction_files = sorted(glob.glob('./data/predictions_*.parquet'))\n",
                    "\n",
                    "daily_predictions = []\n",
                    "for file in prediction_files:\n",
                    "    df = pd.read_parquet(file)\n",
                    "    daily_predictions.append(df)\n",
                    "\n",
                    "print(f\"✓ Loaded {len(daily_predictions)} days of predictions\")\n",
                    "print(f\"  - Each day: ~{len(daily_predictions[0]):,} predictions\")\n",
                    "print(f\"\\nSample from Day 1:\")\n",
                    "daily_predictions[0].head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Initialize Prediction Monitor"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Initialize monitor with baseline\n",
                    "pred_monitor = PredictionMonitor(baseline_predictions)\n",
                    "\n",
                    "print(f\"✓ Prediction monitor initialized\")\n",
                    "print(f\"\\nBaseline statistics:\")\n",
                    "print(json.dumps(pred_monitor.baseline_stats, indent=2))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Run Daily Prediction Monitoring"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Run monitoring for all 30 days\n",
                    "pred_monitoring_results = []\n",
                    "\n",
                    "for day_idx, df in enumerate(daily_predictions):\n",
                    "    result = pred_monitor.monitor_daily_predictions(df)\n",
                    "    result['day'] = day_idx + 1\n",
                    "    pred_monitoring_results.append(result)\n",
                    "\n",
                    "print(f\"✓ Prediction monitoring completed for {len(pred_monitoring_results)} days\")\n",
                    "print(f\"\\nDaily Summary:\")\n",
                    "print(\"Day | Status | Drift | Alerts | Rec Rate\")\n",
                    "print(\"-\" * 50)\n",
                    "for result in pred_monitoring_results:\n",
                    "    summary = result['summary']\n",
                    "    stats = result['current_stats']\n",
                    "    drift = result['drift_test']['drifted']\n",
                    "    status = '🔴' if summary['high_severity_alerts'] > 0 else '🟡' if summary['total_alerts'] > 0 else '🟢'\n",
                    "    drift_icon = '⚠️' if drift else '✓'\n",
                    "    print(f\" {result['day']:2d} |   {status}    |  {drift_icon}   |   {summary['total_alerts']:2d}   | {stats['recommendation_rate']*100:5.1f}%\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Visualize Prediction Monitoring Metrics"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract time series\n",
                    "days = [r['day'] for r in pred_monitoring_results]\n",
                    "mean_uplift = [r['current_stats']['mean_uplift'] for r in pred_monitoring_results]\n",
                    "rec_rate = [r['current_stats']['recommendation_rate'] * 100 for r in pred_monitoring_results]\n",
                    "ks_pvalues = [r['drift_test']['ks_p_value'] for r in pred_monitoring_results]\n",
                    "alerts = [r['summary']['total_alerts'] for r in pred_monitoring_results]\n",
                    "\n",
                    "# Create dashboard\n",
                    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                    "fig.suptitle('Prediction Monitoring Dashboard - 30 Days', fontsize=16, fontweight='bold')\n",
                    "\n",
                    "# Plot 1: Mean uplift score over time\n",
                    "baseline_mean = pred_monitor.baseline_stats['mean_uplift']\n",
                    "axes[0, 0].plot(days, mean_uplift, marker='o', color='blue', linewidth=2, label='Current')\n",
                    "axes[0, 0].axhline(y=baseline_mean, color='green', linestyle='--', linewidth=2, label='Baseline')\n",
                    "axes[0, 0].set_xlabel('Day')\n",
                    "axes[0, 0].set_ylabel('Mean Uplift Score')\n",
                    "axes[0, 0].set_title('Mean Uplift Score Over Time')\n",
                    "axes[0, 0].legend()\n",
                    "axes[0, 0].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 2: Recommendation rate over time\n",
                    "baseline_rec_rate = pred_monitor.baseline_stats['recommendation_rate'] * 100\n",
                    "axes[0, 1].plot(days, rec_rate, marker='s', color='orange', linewidth=2, label='Current')\n",
                    "axes[0, 1].axhline(y=baseline_rec_rate, color='green', linestyle='--', linewidth=2, label='Baseline')\n",
                    "axes[0, 1].axhline(y=baseline_rec_rate + 10, color='red', linestyle=':', alpha=0.5, label='Alert threshold')\n",
                    "axes[0, 1].axhline(y=baseline_rec_rate - 10, color='red', linestyle=':', alpha=0.5)\n",
                    "axes[0, 1].set_xlabel('Day')\n",
                    "axes[0, 1].set_ylabel('Recommendation Rate (%)')\n",
                    "axes[0, 1].set_title('Patients Recommended for Intervention')\n",
                    "axes[0, 1].legend()\n",
                    "axes[0, 1].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 3: Distribution drift (KS test p-value)\n",
                    "axes[1, 0].plot(days, ks_pvalues, marker='o', color='purple', linewidth=2)\n",
                    "axes[1, 0].axhline(y=0.01, color='red', linestyle='--', label='Significance threshold (p=0.01)')\n",
                    "axes[1, 0].set_xlabel('Day')\n",
                    "axes[1, 0].set_ylabel('KS Test p-value')\n",
                    "axes[1, 0].set_title('Uplift Score Distribution Drift')\n",
                    "axes[1, 0].set_yscale('log')\n",
                    "axes[1, 0].legend()\n",
                    "axes[1, 0].grid(True, alpha=0.3)\n",
                    "\n",
                    "# Plot 4: Alerts over time\n",
                    "axes[1, 1].bar(days, alerts, color='red', alpha=0.7)\n",
                    "axes[1, 1].set_xlabel('Day')\n",
                    "axes[1, 1].set_ylabel('Alert Count')\n",
                    "axes[1, 1].set_title('Daily Alerts')\n",
                    "axes[1, 1].grid(True, alpha=0.3, axis='y')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('./prediction_monitoring_dashboard.png', dpi=150, bbox_inches='tight')\n",
                    "plt.show()\n",
                    "\n",
                    "print(\"✓ Visualizations saved\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Uplift Score Distribution Comparison"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Compare uplift distributions: baseline vs current\n",
                    "day_1 = daily_predictions[0]\n",
                    "day_15 = daily_predictions[14]\n",
                    "day_30 = daily_predictions[29]\n",
                    "\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
                    "fig.suptitle('Uplift Score Distribution Over Time', fontsize=14, fontweight='bold')\n",
                    "\n",
                    "# Day 1\n",
                    "axes[0].hist(day_1['uplift_score'], bins=50, color='blue', alpha=0.7, edgecolor='black')\n",
                    "axes[0].axvline(day_1['uplift_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {day_1[\"uplift_score\"].mean():.3f}')\n",
                    "axes[0].set_xlabel('Uplift Score')\n",
                    "axes[0].set_ylabel('Frequency')\n",
                    "axes[0].set_title('Day 1 (Baseline)')\n",
                    "axes[0].legend()\n",
                    "axes[0].grid(True, alpha=0.3, axis='y')\n",
                    "\n",
                    "# Day 15\n",
                    "axes[1].hist(day_15['uplift_score'], bins=50, color='orange', alpha=0.7, edgecolor='black')\n",
                    "axes[1].axvline(day_15['uplift_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {day_15[\"uplift_score\"].mean():.3f}')\n",
                    "axes[1].set_xlabel('Uplift Score')\n",
                    "axes[1].set_ylabel('Frequency')\n",
                    "axes[1].set_title('Day 15')\n",
                    "axes[1].legend()\n",
                    "axes[1].grid(True, alpha=0.3, axis='y')\n",
                    "\n",
                    "# Day 30\n",
                    "axes[2].hist(day_30['uplift_score'], bins=50, color='green', alpha=0.7, edgecolor='black')\n",
                    "axes[2].axvline(day_30['uplift_score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {day_30[\"uplift_score\"].mean():.3f}')\n",
                    "axes[2].set_xlabel('Uplift Score')\n",
                    "axes[2].set_ylabel('Frequency')\n",
                    "axes[2].set_title('Day 30')\n",
                    "axes[2].legend()\n",
                    "axes[2].grid(True, alpha=0.3, axis='y')\n",
                    "\n",
                    "plt.tight_layout()\n",
                    "plt.savefig('./uplift_distribution_comparison.png', dpi=150, bbox_inches='tight')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Alert Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Extract all prediction alerts\n",
                    "pred_alerts = []\n",
                    "for result in pred_monitoring_results:\n",
                    "    for alert in result['alerts']:\n",
                    "        pred_alerts.append({\n",
                    "            'day': result['day'],\n",
                    "            'severity': alert['severity'],\n",
                    "            'metric': alert['metric'],\n",
                    "            'message': alert['message']\n",
                    "        })\n",
                    "\n",
                    "if len(pred_alerts) > 0:\n",
                    "    pred_alerts_df = pd.DataFrame(pred_alerts)\n",
                    "    \n",
                    "    print(f\"Total prediction alerts: {len(pred_alerts_df)}\")\n",
                    "    print(f\"\\nAlerts by severity:\")\n",
                    "    print(pred_alerts_df['severity'].value_counts())\n",
                    "    print(f\"\\nAlerts by metric:\")\n",
                    "    print(pred_alerts_df['metric'].value_counts())\n",
                    "    \n",
                    "    if len(pred_alerts_df[pred_alerts_df['severity'] == 'HIGH']) > 0:\n",
                    "        print(f\"\\n\" + \"=\"*80)\n",
                    "        print(\"HIGH SEVERITY ALERTS:\")\n",
                    "        print(\"=\"*80)\n",
                    "        high_alerts = pred_alerts_df[pred_alerts_df['severity'] == 'HIGH']\n",
                    "        for idx, row in high_alerts.iterrows():\n",
                    "            print(f\"Day {row['day']:2d} | {row['metric']:25s} | {row['message']}\")\n",
                    "else:\n",
                    "    print(\"✓ No alerts detected - predictions are stable!\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Summary & Key Findings\n",
                    "\n",
                    "### Detection Capabilities:\n",
                    "\n",
                    "✅ **Successfully Monitors:**\n",
                    "- Uplift score distribution stability (KS test)\n",
                    "- Mean uplift score shifts (z-score)\n",
                    "- Recommendation rate changes\n",
                    "- Extreme predictions (scores <0 or >1)\n",
                    "- Model component drift (treated/control model outputs)\n",
                    "\n",
                    "### Production Recommendations:\n",
                    "\n",
                    "1. **Daily Monitoring:** Run after daily scoring completes\n",
                    "2. **Alerting:** \n",
                    "   - HIGH: Mean shift >3 std devs OR extreme predictions >1%\n",
                    "   - MEDIUM: Distribution drift OR rec rate change >10pp\n",
                    "3. **Model Refresh:** Consider retraining if alerts persist >5 days\n",
                    "4. **Root Cause:** Check input data drift first (likely upstream cause)\n",
                    "5. **Model Health:** Random Forest models are typically stable - alerts indicate data issues"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Final summary\n",
                    "print(\"=\"*80)\n",
                    "print(\"PREDICTION MONITORING SUMMARY\")\n",
                    "print(\"=\"*80)\n",
                    "print(f\"\\nMonitoring Period: 30 days\")\n",
                    "print(f\"Baseline: Days 1-14\")\n",
                    "print(f\"Model: Random Forest T-Learner (Uplift)\")\n",
                    "print(f\"\\nPrediction Statistics:\")\n",
                    "print(f\"  - Total predictions: {sum(len(d) for d in daily_predictions):,}\")\n",
                    "print(f\"  - Avg daily predictions: {np.mean([len(d) for d in daily_predictions]):.0f}\")\n",
                    "print(f\"  - Avg recommendation rate: {np.mean(rec_rate):.1f}%\")\n",
                    "print(f\"\\nMonitoring Results:\")\n",
                    "if len(pred_alerts) > 0:\n",
                    "    pred_alerts_df = pd.DataFrame(pred_alerts)\n",
                    "    print(f\"  - Total alerts: {len(pred_alerts_df)}\")\n",
                    "    print(f\"  - HIGH severity: {len(pred_alerts_df[pred_alerts_df['severity'] == 'HIGH'])}\")\n",
                    "    print(f\"  - MEDIUM severity: {len(pred_alerts_df[pred_alerts_df['severity'] == 'MEDIUM'])}\")\n",
                    "else:\n",
                    "    print(f\"  - Total alerts: 0\")\n",
                    "print(f\"  - Days with drift: {sum(1 for r in pred_monitoring_results if r['drift_test']['drifted'])}\")\n",
                    "print(f\"\\nModel Stability: {'🟢 STABLE' if len(pred_alerts) == 0 else '🟡 MONITOR' if all(a['severity'] != 'HIGH' for a in pred_alerts) else '🔴 ACTION NEEDED'}\")\n",
                    "print(\"\\n\" + \"=\"*80)\n",
                    "print(\"✓ Prediction monitoring demo complete!\")\n",
                    "print(\"=\"*80)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


if __name__ == "__main__":
    print("Creating monitoring notebooks...")
    
    # Create feature monitoring notebook
    feature_nb = create_feature_monitoring_notebook()
    with open('feature_monitoring_demo.ipynb', 'w') as f:
        json.dump(feature_nb, f, indent=2)
    print("✓ Created: feature_monitoring_demo.ipynb")
    
    # Create prediction monitoring notebook
    prediction_nb = create_prediction_monitoring_notebook()
    with open('prediction_monitoring_demo.ipynb', 'w') as f:
        json.dump(prediction_nb, f, indent=2)
    print("✓ Created: prediction_monitoring_demo.ipynb")
    
    print("\n" + "="*80)
    print("✓ ALL NOTEBOOKS CREATED SUCCESSFULLY!")
    print("="*80)
    print("\nYou can now open and run:")
    print("  1. feature_monitoring_demo.ipynb")
    print("  2. prediction_monitoring_demo.ipynb")

