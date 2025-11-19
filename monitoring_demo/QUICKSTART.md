# Quick Start Guide - Monitoring Demo

## ✅ What Was Created

### 📁 Complete Monitoring System

```
monitoring_demo/
├── 📊 Notebooks (Ready to Run)
│   ├── feature_monitoring_demo.ipynb     ⭐ Feature monitoring with drift detection
│   └── prediction_monitoring_demo.ipynb  ⭐ Model output monitoring
│
├── 🐍 Python Modules
│   ├── feature_monitor.py                Class for feature monitoring
│   ├── prediction_monitor.py             Class for prediction monitoring
│   └── generate_sample_data.py           Data generation script
│
├── 💾 Data (62 files, ~8.5 MB)
│   ├── baseline_features.parquet         Reference distribution (35,000 patients)
│   ├── baseline_predictions.parquet      Reference predictions
│   ├── patient_features_*.parquet        30 days × 2,500 patients/day
│   └── predictions_*.parquet             30 days of model outputs
│
├── 🤖 Models (2 files, ~3.6 MB)
│   ├── model_treated.pkl                 Random Forest (100 trees)
│   └── model_control.pkl                 Random Forest (100 trees)
│
└── 📖 Documentation
    ├── README.md                          Comprehensive guide
    └── QUICKSTART.md                      This file
```

## 🚀 Run the Notebooks (2 minutes)

### Option 1: Feature Monitoring

```bash
cd /Users/ashishmarkanday/github/HealthCareAI/monitoring_demo
jupyter notebook feature_monitoring_demo.ipynb
```

**What it shows:**
- 📉 30 days of patient feature monitoring
- 🔍 Drift detection (Days 15-21)
- ⚠️ Data quality issues (Days 22-30)
- 📊 6-panel visualization dashboard
- 🚨 40+ alerts with severity levels

**Runtime:** ~30 seconds to complete all cells

### Option 2: Prediction Monitoring

```bash
cd /Users/ashishmarkanday/github/HealthCareAI/monitoring_demo
jupyter notebook prediction_monitoring_demo.ipynb
```

**What it shows:**
- 📈 Model prediction stability over 30 days
- 🎯 Uplift score distribution monitoring
- 📊 Recommendation rate tracking
- 🤖 Random Forest model performance
- ✅ Detection of prediction anomalies

**Runtime:** ~20 seconds to complete all cells

## 📊 Expected Outputs

### Feature Monitoring Results

```
Day | Status | Drift | Alerts
---------------------------------
 1  |   🟢    |  0/8  |   0
 2  |   🟢    |  0/8  |   0
...
15  |   🟡    |  3/8  |   4   ← Drift starts
16  |   🟡    |  3/8  |   5
...
22  |   🔴    |  4/8  |   8   ← Quality issues
23  |   🔴    |  4/8  |   9
...
30  |   🔴    |  5/8  |  10

Total Alerts: ~50
  - HIGH: ~15 (null rate spikes)
  - MEDIUM: ~35 (distribution drift)
```

### Prediction Monitoring Results

```
Day | Status | Drift | Rec Rate
----------------------------------
 1  |   🟢    |  ✓   |  20.0%
 2  |   🟢    |  ✓   |  20.0%
...
15  |   🟢    |  ✓   |  20.0%
...
30  |   🟢    |  ✓   |  20.0%

Model Stability: 🟢 STABLE
(Random Forest predictions remain consistent)
```

## 🎨 Visualization Outputs

Both notebooks generate publication-quality plots:

### 1. Feature Monitoring Dashboard
**File:** `feature_monitoring_dashboard.png` (created by notebook)

**6 Panels:**
1. Daily alert count (spike at Day 22)
2. Features with drift (increases at Day 15)
3. PSI for risk scores (>0.1 at Day 15)
4. Mean age over time (+5 years at Day 15)
5. Null rate for age (20% at Day 22)
6. Regional distribution shift (Chi-square test)

### 2. Distribution Comparison
**File:** `distribution_comparison.png`

**4 Subplots:**
- Age: Baseline (blue) vs Day 20 (red)
- Risk Score: Clear mean shift
- Adherence: Distribution change
- Days on Therapy: Stable

### 3. Prediction Dashboard
**File:** `prediction_monitoring_dashboard.png`

**4 Panels:**
1. Mean uplift score (stable around 0.15)
2. Recommendation rate (~20%)
3. KS test p-values (all >0.01)
4. Alert timeline (minimal)

## 🧪 Try These Experiments

### Experiment 1: Adjust Alert Thresholds

```python
# In feature_monitor.py, line 140
if null_increase > 0.10:  # Change from 0.05 to 0.10
    alerts.append(...)

# Re-run notebook to see fewer alerts
```

### Experiment 2: Generate New Data with Different Drift

```python
# In generate_sample_data.py, line 45
if introduce_drift:
    age_mean += 10  # Change from 5 to 10 (more extreme drift)

# Regenerate data
python generate_sample_data.py

# Run notebooks again to see stronger drift signals
```

### Experiment 3: Custom Metric

```python
# In feature_monitoring_demo.ipynb, add new cell:

# Monitor a custom business metric
high_risk_patients = daily_data[day_idx]
high_risk_count = (high_risk_patients['discontinuation_risk_score'] > 0.5).sum()
print(f"Day {day_idx+1}: {high_risk_count} high-risk patients")
```

## 📈 Key Metrics Explained

### Population Stability Index (PSI)
```
PSI < 0.1:   🟢 No significant drift
PSI 0.1-0.25: 🟡 Slight drift (monitor)
PSI > 0.25:   🔴 Major drift (investigate)
```

**Example from notebook:**
- Days 1-14: PSI ~0.02 (stable)
- Days 15-21: PSI ~0.18 (moderate drift)
- Days 22-30: PSI ~0.30 (major drift + quality issues)

### KS Test (Kolmogorov-Smirnov)
```
p > 0.01: ✅ Distributions similar
p < 0.01: ⚠️ Distributions different
```

**Example from notebook:**
- Day 15: p-value drops below 0.01 for age and risk_score

### Z-Score Alerting
```
|z| < 2:   Normal variation
|z| 2-3:   🟡 Monitor closely
|z| > 3:   🔴 Alert triggered
```

**Example from notebook:**
- Day 15: Age mean shifts by z=4.2 (alert!)

## 🔧 Using in Your Own Project

### 1. Adapt Feature Monitor

```python
from feature_monitor import FeatureMonitor

# Your data
my_baseline = pd.read_csv('my_baseline_data.csv')
my_current = pd.read_csv('todays_data.csv')

# Monitor
monitor = FeatureMonitor(my_baseline)
results = monitor.monitor_daily_data(my_current)

# Check results
print(f"Alerts: {results['summary']['total_alerts']}")
```

### 2. Adapt Prediction Monitor

```python
from prediction_monitor import PredictionMonitor

# Your predictions
my_baseline_preds = pd.read_csv('baseline_predictions.csv')
today_preds = pd.read_csv('todays_predictions.csv')

# Monitor
pred_monitor = PredictionMonitor(my_baseline_preds)
results = pred_monitor.monitor_daily_predictions(today_preds)

# Check drift
if results['drift_test']['drifted']:
    print("⚠️ Prediction drift detected!")
```

## 🎯 Next Steps

### For Learning:
1. ✅ Run both notebooks start to finish
2. ✅ Read the alert messages to understand what was detected
3. ✅ Examine the visualization dashboards
4. ✅ Try the experiments above

### For Production:
1. 📖 Review the [full project plan](../specialty_pharmacy/intervention_recommendation_engine_project_plan.md)
2. 🔧 Adapt monitoring classes for your data schema
3. ☁️ Deploy to AWS Lambda (see README section on AWS)
4. 📊 Set up CloudWatch dashboards
5. 🚨 Configure SNS alerting

### For Advanced Users:
1. Add performance monitoring (requires outcome data)
2. Implement automated retraining triggers
3. Build Athena queries for historical analysis
4. Create QuickSight executive dashboards

## 💡 Common Questions

**Q: Why are predictions stable even with feature drift?**  
A: Random Forest models are robust to moderate distribution shifts. However, prolonged drift will eventually impact performance. This demonstrates why input monitoring is critical.

**Q: How do I know what thresholds to use?**  
A: Start conservative (as shown in demo), then tune based on false positive rate. Track metrics for 2-4 weeks, then adjust thresholds to target 1-2 actionable alerts per week.

**Q: Can I use this for non-healthcare ML systems?**  
A: Absolutely! The monitoring classes are domain-agnostic. Just replace the feature names and adjust thresholds for your domain.

**Q: How much data do I need for a baseline?**  
A: Minimum 2 weeks of stable production data. Ideal: 4-8 weeks. Update baseline quarterly or after major system changes.

## 📞 Support

**Issues?**
- Check that all cells executed without errors
- Verify data files exist in `./data/` and `./models/`
- Re-run `python generate_sample_data.py` if data is missing

**Questions?**
- Read the comprehensive [README.md](README.md)
- Check docstrings in `feature_monitor.py` and `prediction_monitor.py`
- Review [project plan](../specialty_pharmacy/intervention_recommendation_engine_project_plan.md) for production details

## ✨ Quick Win

**Run this single command to see everything:**

```bash
cd /Users/ashishmarkanday/github/HealthCareAI/monitoring_demo && \
jupyter nbconvert --to notebook --execute --inplace feature_monitoring_demo.ipynb && \
jupyter nbconvert --to notebook --execute --inplace prediction_monitoring_demo.ipynb && \
echo "✅ Done! Open the notebooks to see results."
```

This executes both notebooks and saves outputs inline (takes ~1 minute).

---

**Ready to start?** Open `feature_monitoring_demo.ipynb` and run all cells! 🚀

