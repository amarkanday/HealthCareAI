# Data & Model Monitoring Demo

Comprehensive demonstration of feature monitoring and model output monitoring for ML systems in production.

## Overview

This folder contains a complete end-to-end demonstration of monitoring systems for the **Intervention Recommendation Engine** including:

1. **Feature Monitoring** - Input data quality and drift detection
2. **Model Output Monitoring** - Prediction drift and performance tracking
3. **Sample Data Generation** - 30 days of synthetic patient data with intentional drift
4. **Random Forest Model** - T-Learner uplift model for demonstration

## 📁 Contents

```
monitoring_demo/
├── README.md                           # This file
├── generate_sample_data.py            # Generate 30 days of patient data + model
├── feature_monitor.py                 # Feature monitoring class
├── prediction_monitor.py              # Prediction monitoring class
├── create_notebooks.py                # Helper to create demo notebooks
├── feature_monitoring_demo.ipynb      # 🎯 Feature monitoring notebook
├── prediction_monitoring_demo.ipynb   # 🎯 Prediction monitoring notebook
├── data/                              # Generated data (60+ files)
│   ├── patient_features_*.parquet    # Daily patient features (30 files)
│   ├── predictions_*.parquet         # Daily model predictions (30 files)
│   ├── baseline_features.parquet     # Reference distribution
│   └── baseline_predictions.parquet  # Reference predictions
└── models/                            # Trained models
    ├── model_treated.pkl              # Random Forest (treated patients)
    └── model_control.pkl              # Random Forest (control patients)
```

## 🚀 Quick Start

### 1. Generate Sample Data & Models

```bash
cd monitoring_demo
python generate_sample_data.py
```

**Output:**
- 30 days of patient feature data (75,000 patients total)
- 30 days of model predictions
- 2 trained Random Forest models (T-Learner approach)
- Baseline references for monitoring

**Data Characteristics:**
- **Days 1-14:** Stable baseline (no drift)
- **Days 15-21:** Distribution drift introduced
  - Age mean shifts +5 years
  - Risk scores increase
  - Regional distribution changes
- **Days 22-30:** Data quality issues
  - Null rate increases to 20%
  - Missing value spikes

### 2. Run Feature Monitoring Demo

Open and run: **`feature_monitoring_demo.ipynb`**

**What it demonstrates:**
- ✅ Null rate monitoring
- ✅ Mean/std deviation shift detection
- ✅ Distribution drift (KS test, PSI)
- ✅ Categorical feature drift (Chi-square)
- ✅ Alerting based on thresholds
- ✅ Visualization dashboards

**Key Metrics:**
- **PSI** (Population Stability Index): <0.1 stable, >0.25 major drift
- **KS Test**: p<0.01 indicates significant distribution change
- **Z-Score**: Mean shifts >3 std devs trigger alerts
- **Null Rate**: >5pp increase triggers HIGH severity alert

### 3. Run Model Output Monitoring Demo

Open and run: **`prediction_monitoring_demo.ipynb`**

**What it demonstrates:**
- ✅ Uplift score distribution monitoring
- ✅ Recommendation rate stability
- ✅ Extreme prediction detection
- ✅ Model component drift (treated vs control models)
- ✅ Performance metrics (when outcomes available)

**Key Metrics:**
- **Mean Uplift Shift**: >2 std devs triggers alert
- **Recommendation Rate**: >10pp change triggers alert
- **Distribution Drift**: KS test on predictions
- **Extreme Predictions**: Scores <0 or >1

## 📊 Expected Results

### Feature Monitoring

**Alert Timeline:**
- **Days 1-14:** ✅ No alerts (stable baseline)
- **Day 15:** 🟡 First drift alerts appear
- **Days 16-21:** 🟡 Distribution drift persists
- **Day 22+:** 🔴 HIGH severity - null rate spike

**Total Alerts:** ~40-60 alerts across 30 days
- HIGH: ~10-15 (data quality issues)
- MEDIUM: ~30-45 (drift detection)

### Prediction Monitoring

**Stability:**
- Random Forest models are inherently stable
- Prediction drift correlates with input feature drift
- Recommendation rate remains ~20% (stable)

**Alert Triggers:**
- Most prediction alerts caused by upstream feature drift
- Model outputs remain within acceptable bounds

## 🔧 Using Monitoring Classes Standalone

### Feature Monitor

```python
from feature_monitor import FeatureMonitor
import pandas as pd

# Load baseline data
baseline_df = pd.read_parquet('./data/baseline_features.parquet')

# Initialize monitor
monitor = FeatureMonitor(baseline_df)

# Monitor today's data
today_df = pd.read_parquet('./data/patient_features_20250101.parquet')
results = monitor.monitor_daily_data(today_df)

# Check alerts
if results['summary']['high_severity_alerts'] > 0:
    print("🔴 HIGH SEVERITY ALERTS!")
    for alert in results['alerts']:
        if alert['severity'] == 'HIGH':
            print(f"  - {alert['feature']}: {alert['message']}")

# Save results
monitor.save_monitoring_results(results, './monitoring_results.json')
```

### Prediction Monitor

```python
from prediction_monitor import PredictionMonitor
import pandas as pd

# Load baseline predictions
baseline_preds = pd.read_parquet('./data/baseline_predictions.parquet')

# Initialize monitor
pred_monitor = PredictionMonitor(baseline_preds)

# Monitor today's predictions
today_preds = pd.read_parquet('./data/predictions_20250101.parquet')
results = pred_monitor.monitor_daily_predictions(today_preds)

# Check for drift
if results['drift_test']['drifted']:
    print("⚠️ Prediction drift detected!")
    print(f"  KS p-value: {results['drift_test']['ks_p_value']:.4f}")

# Save results
pred_monitor.save_monitoring_results(results, './pred_monitoring_results.json')
```

## 📈 Visualization Outputs

Both notebooks generate publication-quality visualizations:

**Feature Monitoring Dashboard** (`feature_monitoring_dashboard.png`):
- 6 panels showing drift metrics over time
- PSI, null rates, mean shifts, categorical drift

**Prediction Monitoring Dashboard** (`prediction_monitoring_dashboard.png`):
- 4 panels showing prediction stability
- Mean uplift, recommendation rate, KS test p-values

**Distribution Comparisons**:
- Baseline vs drift period histograms
- Side-by-side categorical distributions

## 🎯 Use Cases

### 1. AWS Lambda Production Deployment

```python
# lambda_function.py
from feature_monitor import FeatureMonitor
import boto3
import pandas as pd

def lambda_handler(event, context):
    """Daily feature monitoring Lambda"""
    
    # Load baseline from S3
    s3 = boto3.client('s3')
    baseline_df = pd.read_parquet('s3://monitoring/baseline_features.parquet')
    
    # Load today's features
    today_df = pd.read_parquet('s3://features/daily/2025-01-15/features.parquet')
    
    # Monitor
    monitor = FeatureMonitor(baseline_df)
    results = monitor.monitor_daily_data(today_df)
    
    # Alert if needed
    if results['summary']['high_severity_alerts'] > 0:
        sns = boto3.client('sns')
        sns.publish(
            TopicArn='arn:aws:sns:us-east-1:123456789:monitoring-alerts',
            Subject='🔴 HIGH SEVERITY: Feature Monitoring Alert',
            Message=json.dumps(results['alerts'], indent=2)
        )
    
    return {'statusCode': 200, 'body': json.dumps(results['summary'])}
```

### 2. Weekly Model Performance Review

```python
# weekly_review.py
from prediction_monitor import PredictionMonitor
import pandas as pd

# Load predictions from last 7 days
predictions = []
for day in range(7):
    date = datetime.now() - timedelta(days=day)
    df = pd.read_parquet(f'./data/predictions_{date.strftime("%Y%m%d")}.parquet')
    predictions.append(df)

weekly_preds = pd.concat(predictions)

# Generate performance report
monitor = PredictionMonitor(baseline_preds)
summary_stats = {
    'mean_uplift': weekly_preds['uplift_score'].mean(),
    'recommendation_rate': weekly_preds['recommendation'].mean(),
    'total_predictions': len(weekly_preds)
}

print(f"Weekly Performance Summary")
print(f"  Mean Uplift: {summary_stats['mean_uplift']:.4f}")
print(f"  Recommendation Rate: {summary_stats['recommendation_rate']:.1%}")
```

### 3. Automated Retraining Trigger

```python
# check_drift_for_retraining.py

# Run monitoring
results = monitor.monitor_daily_data(today_df)

# Trigger retraining if multiple features drifting for >3 days
if results['summary']['features_with_drift'] >= 3:
    # Check historical results (last 3 days)
    drift_days = count_consecutive_drift_days()
    
    if drift_days >= 3:
        print("🔄 Triggering model retraining")
        trigger_sagemaker_training_job()
```

## 🔬 Technical Details

### Monitoring Algorithms

**Population Stability Index (PSI):**
```
PSI = Σ (current% - baseline%) × ln(current% / baseline%)

Interpretation:
- PSI < 0.1:  No significant change
- PSI 0.1-0.25: Slight distribution change
- PSI > 0.25: Major distribution shift (investigate)
```

**Kolmogorov-Smirnov Test:**
- Non-parametric test for continuous distributions
- Compares CDFs of baseline vs current
- p < 0.01 indicates significant drift

**Chi-Square Test:**
- Tests categorical feature distributions
- Compares frequency tables
- p < 0.01 indicates significant change

**Z-Score Alerting:**
```
z = (current_mean - baseline_mean) / baseline_std

Alert if |z| > 3 (99.7% confidence)
```

### Model Architecture

**T-Learner (Two-Model Uplift):**

```
Model_Treated:  P(success | intervene, X)
Model_Control:  P(success | no_intervene, X)

Uplift = Model_Treated(X) - Model_Control(X)

Recommendation = 1 if Uplift > threshold, 0 otherwise
```

**Random Forest Configuration:**
- 100 trees
- Max depth: 10
- Min samples split: 50
- Features: age, days_on_therapy, risk_score, adherence, comorbidities, interactions

## 📝 Production Recommendations

### Daily Operations

1. **Feature Monitoring**:
   - Run after feature ETL completes (1:30 AM)
   - HIGH severity → page on-call engineer
   - MEDIUM severity → Slack alert to data team

2. **Prediction Monitoring**:
   - Run after daily scoring (2:30 AM)
   - Investigate if drift persists >2 days
   - Check input features first (usually root cause)

3. **Performance Monitoring**:
   - Run weekly (once outcomes available)
   - Compare observed vs predicted treatment effect
   - Update stakeholder dashboard

### Alerting Thresholds

| Severity | Feature Monitoring | Prediction Monitoring | Action |
|----------|-------------------|----------------------|--------|
| **HIGH** | Null rate >15%, Missing feature | Extreme predictions >1%, Mean shift >3σ | Page on-call, investigate immediately |
| **MEDIUM** | PSI >0.1, Mean shift >2σ | Distribution drift, Rec rate >10pp change | Slack alert, review within 4 hours |
| **LOW** | New categories, Minor shifts | Model component drift | Log for review, no immediate action |

### Model Retraining Policy

**Trigger retraining if:**
- HIGH severity alerts for >2 consecutive days
- ≥4 features with PSI >0.25
- Model performance drops >10% (AUC, Qini)
- Manual trigger from stakeholder

**Retraining process:**
1. Pull last 90 days of data
2. Train new T-Learner models
3. Validate on holdout set
4. A/B test new model (20% traffic)
5. Gradual rollout if metrics improve

## 🧪 Extending the Demo

### Add New Features

```python
# In generate_sample_data.py
def generate_patient_features(...):
    data = {
        # ... existing features ...
        'new_feature': np.random.normal(100, 20, n_patients)
    }
```

### Customize Thresholds

```python
# In feature_monitor.py
def _check_continuous_feature(...):
    # Change null rate threshold
    if null_increase > 0.10:  # Changed from 0.05
        alerts.append(...)
```

### Add Custom Metrics

```python
# In prediction_monitor.py
def monitor_daily_predictions(...):
    # Add custom business metric
    high_risk_rec_rate = current_predictions[
        current_predictions['risk_score'] > 0.5
    ]['recommendation'].mean()
    
    current_stats['high_risk_rec_rate'] = high_risk_rec_rate
```

## 📚 Related Documentation

- [Project Plan](../specialty_pharmacy/intervention_recommendation_engine_project_plan.md) - Full production implementation plan
- [Incrementality Analysis](../specialty_pharmacy/incrementality_analysis.py) - Causal inference framework
- [AWS Architecture](../specialty_pharmacy/intervention_recommendation_engine_project_plan.md#6-production-implementation-plan) - Production deployment on AWS

## 🤝 Contributing

To add new monitoring capabilities:

1. Add test cases to `generate_sample_data.py`
2. Implement monitoring logic in `feature_monitor.py` or `prediction_monitor.py`
3. Update notebooks to demonstrate new capabilities
4. Update this README with usage examples

## 📞 Support

For questions or issues:
- Review notebooks for usage examples
- Check monitoring class docstrings
- Refer to project plan for production architecture

## 🎓 Learning Objectives

After completing this demo, you should understand:

- ✅ How to detect input data drift in production
- ✅ How to monitor model predictions for anomalies
- ✅ Statistical tests for distribution comparison (KS, Chi-square, PSI)
- ✅ Setting appropriate alerting thresholds
- ✅ Visualizing monitoring metrics over time
- ✅ Designing end-to-end monitoring infrastructure
- ✅ AWS deployment patterns for ML monitoring

---

**Version:** 1.0  
**Last Updated:** November 19, 2025  
**Author:** HealthCareAI Team

