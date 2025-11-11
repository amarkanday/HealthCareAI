# Intervention Recommendation Engine - Project Plan

**Document Version:** 1.0  
**Date:** November 11, 2025  
**Project Status:** Planning Phase  
**Estimated Timeline:** 6 months to full production

---

## Executive Summary

This document outlines a comprehensive plan create an intelligent **Intervention Recommendation Engine** that guides Field Reimbursement Managers (FRMs) on which patients to engage and when, maximizing therapy adherence while minimizing intervention fatigue. The solution leverages causal inference methodologies to measure true intervention incrementality and uses uplift modeling to personalize recommendations at scale.

**Key Benefits:**
- **15-25% reduction** in preventable medication discontinuations
- **30-40% improvement** in FRM productivity through prioritized patient lists
- **Data-driven** intervention strategies with measurable ROI
- **Continuous learning** through integrated A/B testing framework

---

## 1. Business Problem

### 1.1 Current State

**Problem Statement:**  
Field Reimbursement Managers (FRMs) currently receive alerts from predictive models (e.g., discontinuation risk scores) but lack actionable guidance on:
- **Which patients to prioritize** when multiple patients are at high risk
- **When to intervene** to maximize impact without causing intervention fatigue
- **What type of intervention** is most likely to succeed for each patient
- **Whether interventions are truly incremental** or if patients would have succeeded anyway

This results in:
- Suboptimal resource allocation across 2,000-3,000 patients per brand
- Intervention fatigue from over-contacting certain patients
- Missed opportunities for high-impact interventions
- Inability to measure true causal effect of FRM interventions
- No systematic learning from past intervention outcomes

### 1.2 Desired Future State

**Vision:**  
FRMs receive a daily prioritized list of patients with personalized intervention recommendations, powered by causal machine learning that predicts the **incremental benefit** of intervention for each patient.

**Success Metrics:**
- **Clinical Impact:** 15-25% reduction in preventable discontinuations
- **Operational Efficiency:** 40% reduction in time spent on patient prioritization
- **ROI:** $150K-$300K annual value per brand (based on avoided discontinuations)
- **FRM Satisfaction:** >80% report improved decision-making confidence
- **Model Performance:** AUC >0.75 for uplift predictions, SMD <0.1 after weighting

---

## 2. Target Users & Workflows

### 2.1 Primary Users

**Field Reimbursement Managers (FRMs)**
- **Count:** Varies by brand, typically 5-15 FRMs per specialty drug
- **Responsibilities:** 
  - Monitor 150-400 patients per FRM
  - Conduct phone outreach for adherence support
  - Coordinate with healthcare providers and pharmacies
  - Resolve prior authorization and reimbursement issues
- **Current Tools:** Claritas RX Patient Watch dashboard
- **Daily Time Budget:** ~4 hours for proactive outreach

**Secondary Users:**
- **Clinical Program Managers:** Monitor aggregate metrics and FRM performance
- **Data Science Team:** Model monitoring, retraining, and experimentation
- **Pharma Brand Teams:** Strategic insights on intervention effectiveness

### 2.2 Workflow Integration

#### Current Workflow (As-Is)
```
1. FRM logs into Patient Watch (8:00 AM)
2. Reviews risk score dashboard (all high-risk patients)
3. Manually prioritizes based on experience
4. Reaches out to 10-15 patients per day
5. Documents interaction in CRM
6. Repeats daily
```

#### Enhanced Workflow (To-Be)
```
1. FRM logs into Patient Watch (8:00 AM)
2. **NEW: Intervention Recommendations Tab appears**
   └─> Top 10-15 patients ranked by uplift score
   └─> Recommended action for each (call, text, wait)
   └─> Suggested talking points based on patient journey
   └─> Estimated impact if intervention succeeds
3. FRM reviews recommendations (5 min vs. 30 min)
4. Conducts outreach following guidance
5. Documents outcome (success, no answer, refused, etc.)
6. **NEW: System learns from outcomes for future recommendations**
```

**Key Integration Points:**
- **Input:** Patient Watch risk scores, journey data, past interactions
- **Output:** Daily recommendation list in existing dashboard
- **Feedback Loop:** Outcome tracking via CRM integration
- **Transparency:** Explainable scores (why this patient, why now)

---

## 3. Data Requirements

### 3.1 Required Data Sources

| Data Source | Availability | Refresh Frequency | Critical Fields |
|------------|--------------|-------------------|-----------------|
| **Patient Watch Core** | ✓ Existing | Daily | patient_id, brand, enrollment_date, status |
| **Risk Scores** | ✓ Existing | Daily | discontinuation_risk_score, adherence_risk_score |
| **Journey Events** | ✓ Existing | Real-time | event_type, event_date, event_source |
| **Interaction History** | ✓ CRM | Near real-time | interaction_date, frm_id, type, outcome |
| **Clinical Data** | ✓ Claims/EMR | Weekly | diagnosis, comorbidities, prior_medications |
| **Pharmacy Data** | ✓ Claims | Daily | fill_date, days_supply, refill_due_date |
| **Outcomes** | ⚠️ Need Enhancement | Weekly | discontinuation_date, reason, re-initiation |

### 3.2 Feature Engineering Pipeline

**Static Features (Patient Level):**
- Demographics: age, gender, geographic region
- Clinical: diagnosis, comorbidity_count, disease_severity
- Socioeconomic: estimated_income_bracket, insurance_type

**Dynamic Features (Time-Varying):**
- Risk trajectory: risk_score_7d_change, risk_score_30d_max
- Adherence patterns: MPR_30d, MPR_90d, days_since_last_fill
- Engagement history: total_interactions, days_since_last_contact
- Response patterns: past_response_rate, preferred_contact_method
- Journey stage: days_on_therapy, refill_number, switches_count

**Intervention Features:**
- Intervention fatigue: contacts_last_30d, contacts_last_90d
- Intervention timing: days_since_last_intervention, time_of_year
- FRM relationship: interactions_with_current_frm, frm_tenure

**Outcome Variables:**
- **Primary:** discontinuation_within_30d (binary)
- **Secondary:** discontinuation_within_90d, adherence_improvement_30d
- **Treatment:** intervention_received (binary), derived from CRM logs

### 3.3 Data Quality Requirements

**Minimum Viable Dataset for Model Training:**
- **Sample Size:** ≥1,500 patients with complete feature and outcome data
- **Observation Period:** ≥6 months of historical data
- **Treatment/Control Split:** ≥30% of patients with documented interventions
- **Outcome Completeness:** ≥90% of patients with known outcomes
- **Feature Completeness:** ≥85% non-missing for critical features

**Data Quality Checks (Daily):**
- Null rate monitoring per feature (<15% threshold)
- Duplicate patient record detection
- Temporal consistency checks (no future dates)
- Range validation for continuous features
- Referential integrity between tables

---

## 4. Assumptions & Constraints

### 4.1 Key Assumptions

**Business Assumptions:**
1. FRMs have capacity to conduct 10-15 prioritized interventions per day
2. Phone/text outreach can influence patient adherence behavior
3. Past intervention outcomes are reasonably documented in CRM (≥70% capture rate)
4. Patients tolerate 1-2 interventions per month without significant fatigue
5. Intervention effects manifest within 30-90 days

**Technical Assumptions:**
6. Historical data contains sufficient variation in intervention patterns for causal inference
7. Patient Watch data pipeline can support daily feature generation
8. AWS infrastructure is available for model deployment
9. CRM system can provide near-real-time feedback on intervention outcomes
10. Model can be retrained weekly without significant operational burden

**Statistical Assumptions:**
11. Unconfoundedness: All major confounders are observed (conditional ignorability)
12. Overlap: All patient types have non-zero probability of intervention
13. SUTVA: Patient outcomes are independent (no spillover effects)
14. Effect homogeneity: Treatment effects vary smoothly with covariates

### 4.2 Known Constraints

**Sample Size Constraints:**
- 2,000-3,000 patients per brand limits model complexity
- Small effect sizes may be hard to detect without sufficient power
- Rare subgroups may have unstable estimates

**Operational Constraints:**
- FRMs work across multiple time zones (scoring must complete by 7 AM EST)
- Limited bandwidth for model retraining (prefer weekly, not daily)
- Dashboard must load within 3 seconds for good UX

**Regulatory Constraints:**
- PHI data requires HIPAA-compliant infrastructure
- Model decisions must be explainable for compliance
- Cannot use protected characteristics (race, ethnicity) directly in scoring

**Technical Constraints:**
- AWS Lambda 15-minute timeout for batch processing
- Model file size <250 MB for Lambda deployment
- Must integrate with existing Patient Watch tech stack

---

## 5. Modeling Approach

### 5.1 Problem Framing

**Core ML Task:** Binary uplift modeling  
**Target:** Predict *P(success | intervene) - P(success | no intervene)* for each patient

This is a **causal inference problem**, not just a prediction problem, because:
- We don't just want to find high-risk patients (existing risk scores do this)
- We want to find patients whose outcomes **improve due to intervention**
- Some high-risk patients may succeed without intervention (opportunity cost)
- Some low-risk patients may benefit greatly from timely intervention

### 5.2 Modeling Options Considered

#### Option 1: Rule-Based Scoring (Baseline)
**Approach:** Simple weighted score based on risk thresholds
```
Intervention_Score = 0.4 * Risk_Score + 0.3 * (1 - Fatigue) + 0.3 * Time_Since_Contact
```

| Pros | Cons |
|------|------|
| ✓ Easy to implement (2 weeks) | ✗ No personalization |
| ✓ Fully transparent | ✗ Cannot learn from data |
| ✓ No training data needed | ✗ Ignores causal effects |
| ✓ Fast inference (<1ms) | ✗ Suboptimal ROI |

**Verdict:** Good for Phase 0 MVP (Month 1), but insufficient for final solution

---

#### Option 2: Single Outcome Model (Risk Prediction Only)
**Approach:** Train one model to predict discontinuation risk
```
Model: P(discontinuation | patient_features)
```

| Pros | Cons |
|------|------|
| ✓ Simple training pipeline | ✗ Doesn't account for intervention effect |
| ✓ Interpretable with SHAP | ✗ May prioritize patients who would succeed anyway |
| ✓ Works with 2K-3K samples | ✗ Cannot measure incrementality |

**Verdict:** Already available via existing risk scores; doesn't solve the problem

---

#### Option 3: Two-Model Uplift (T-Learner) ⭐ **RECOMMENDED**
**Approach:** Train two models, one on intervened patients and one on non-intervened
```
Model_T: P(success | intervene, X)
Model_C: P(success | no intervene, X)
Uplift = Model_T(X) - Model_C(X)
```

| Pros | Cons |
|------|------|
| ✓ Directly estimates heterogeneous treatment effects | ⚠️ Requires treatment variation in historical data |
| ✓ Works well with 2K-3K samples | ⚠️ Sensitive to propensity score model quality |
| ✓ Can use simple models (logistic regression) | ⚠️ Assumes correct functional form |
| ✓ Fast inference (2 predictions per patient) | |
| ✓ Interpretable with domain knowledge | |
| ✓ Proven in similar healthcare applications | |

**Implementation Details:**
- **Algorithm:** Logistic Regression or Gradient Boosting for each model
- **Training:** Use propensity score weighting (IPTW) to reduce confounding
- **Validation:** Stratified 5-fold CV with temporal holdout
- **Inference Time:** <1ms per patient on Lambda

**Verdict:** Best balance of statistical rigor, interpretability, and operationalization for your constraints

---

#### Option 4: Causal Forest / Meta-Learners (X-Learner, DR-Learner)
**Approach:** Doubly robust estimation with ensemble methods
```
DR-Learner: Combines outcome regression + propensity weighting
Causal Forest: Tree-based heterogeneous effect estimation
```

| Pros | Cons |
|------|------|
| ✓ More robust to model misspecification | ✗ Requires larger samples (5K+) |
| ✓ Better captures complex interactions | ✗ Harder to interpret |
| ✓ State-of-the-art in research | ✗ Longer inference time |

**Verdict:** Consider for Phase 2 (Month 9+) after accumulating more data

---

#### Option 5: Reinforcement Learning / Contextual Bandits
**Approach:** Online learning with exploration-exploitation
```
Thompson Sampling or LinUCB for dynamic recommendation
```

| Pros | Cons |
|------|------|
| ✓ Adapts in real-time to feedback | ✗ Complex infrastructure (real-time serving) |
| ✓ Natural exploration mechanism | ✗ Requires significant engineering |
| ✓ Optimal in the long run | ✗ Slower to converge with small samples |

**Verdict:** Not recommended for initial deployment; revisit in Year 2

---

### 5.3 Final Model Recommendation

**Phase 1 (Months 1-3): Hybrid Rule-Based + ML**
- Start with rule-based scoring for quick wins
- Add simple propensity score model to adjust for confounding
- Collect ground truth data for uplift modeling

**Phase 2 (Months 4-6): T-Learner Uplift Model** ⭐
- Train two-model approach on 6 months of Phase 1 data
- Implement IPTW weighting to handle confounding
- Deploy to production with A/B testing

**Phase 3 (Months 7-12): Enhanced Uplift + Business Logic**
- Add time-series features (trajectory modeling)
- Implement heterogeneous effect estimation by subgroup
- Optimize for business constraints (intervention capacity, timing)

**Long-Term (Year 2+):**
- Consider causal forests if sample size grows to 5K+
- Explore multi-armed bandit for intervention type selection
- Add LLM-powered talking point generation

---

## 6. Production Implementation Plan

### 6.1 Phased Rollout Strategy

#### **Phase 0: Foundation (Weeks 1-4)**
**Objective:** Set up infrastructure and baseline

**Deliverables:**
- ✓ AWS infrastructure (S3, Lambda, RDS)
- ✓ Data pipeline for feature generation
- ✓ Rule-based scoring model (v0.1)
- ✓ Basic dashboard integration
- ✓ Logging and monitoring setup

**Success Criteria:**
- Daily scoring runs without failures
- FRMs can access prioritized list in Patient Watch
- <10% null rate for critical features

**Rollout:** Single brand pilot (1 FRM team)

---

#### **Phase 1: Uplift Model Training (Weeks 5-12)**
**Objective:** Build and validate causal inference model

**Deliverables:**
- ✓ Historical data analysis with causal inference
- ✓ Propensity score model trained and validated
- ✓ T-Learner uplift model (v1.0)
- ✓ Offline validation with held-out data
- ✓ Model explainability dashboard (SHAP values)

**Success Criteria:**
- Propensity score overlap: SMD <0.1 after IPTW
- Uplift model AUC >0.70 (using qini curve or uplift AUC)
- Model passes sensitivity analysis (E-values)
- Stakeholder review and approval

**Rollout:** Not yet deployed; offline validation only

---

#### **Phase 2: A/B Test Deployment (Weeks 13-24)**
**Objective:** Prove causal impact in production

**Deliverables:**
- ✓ Deploy uplift model to production (Lambda)
- ✓ A/B testing infrastructure (20% control, 80% treatment)
- ✓ Weekly model retraining pipeline
- ✓ Real-time monitoring dashboard
- ✓ Automated alerting for anomalies

**A/B Test Design:**
```
Randomization Unit: Patient
Treatment Arm (80%): Show uplift-based recommendations to FRMs
Control Arm (20%): Show rule-based recommendations (current state)
Stratification: By risk quintile to ensure balance
Primary Metric: 30-day discontinuation rate
Secondary Metrics: 90-day adherence, FRM productivity, intervention count
Duration: 8 weeks (to capture full refill cycle)
Sample Size: ~2,400 patients (80% power to detect 5% absolute difference)
```

**Success Criteria:**
- Treatment arm shows ≥5% relative reduction in discontinuations (p<0.05)
- No increase in intervention fatigue (contacts per patient stable)
- FRM satisfaction score >80%
- System uptime >99.5%

**Rollout:** 2-3 brands in parallel

---

#### **Phase 3: Full Production (Weeks 25-26)**
**Objective:** Scale to all brands

**Deliverables:**
- ✓ Deploy to all specialty brands
- ✓ MLOps automation (retraining, monitoring, alerting)
- ✓ Model governance documentation
- ✓ Training materials for FRMs and stakeholders
- ✓ Ongoing A/B test framework (continuous learning)

**Success Criteria:**
- All brands migrated successfully
- <1% regression incidents per quarter
- Model drift alerts <5% false positive rate

**Rollout:** Gradual rollout (1 new brand per week)

---

### 6.2 AWS MLOps Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       PRODUCTION ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│  Data Sources    │
├──────────────────┤
│ Patient Watch DB │
│ CRM System       │──┐
│ Claims Feed      │  │
└──────────────────┘  │
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ AWS Glue ETL / Lambda                               │    │
│  │ - Extract from Patient Watch, CRM                   │    │
│  │ - Transform: Calculate risk trajectories, features  │    │
│  │ - Load to S3 Feature Store (Parquet)                │    │
│  └────────────────────────────────────────────────────┘    │
│  Trigger: Daily at 1:00 AM EST (EventBridge)               │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    FEATURE STORE (S3)                        │
│  s3://claritas-ml-features/                                  │
│  ├── daily/{date}/patient_features.parquet                   │
│  ├── daily/{date}/risk_scores.parquet                        │
│  └── daily/{date}/interaction_history.parquet                │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              DAILY SCORING PIPELINE (Lambda)                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ intervention-recommender-scoring                    │    │
│  │ - Load model from S3                                │    │
│  │ - Load features from Feature Store                  │    │
│  │ - Score all active patients (~3000)                 │    │
│  │ - Apply A/B test assignment logic                   │    │
│  │ - Generate recommendations                          │    │
│  │ - Write to RDS + S3                                 │    │
│  └────────────────────────────────────────────────────┘    │
│  Trigger: Daily at 2:00 AM EST (EventBridge)               │
│  Runtime: ~30-60 seconds                                    │
└─────────────────────────────────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
┌──────────────────────┐ ┌──────────────────────┐
│  Aurora RDS          │ │  S3 Predictions      │
│  (API Layer)         │ │  (Analytics Layer)   │
├──────────────────────┤ ├──────────────────────┤
│ recommendations      │ │ s3://predictions/    │
│ ├─ patient_id        │ │ ├─ daily/{date}/     │
│ ├─ uplift_score      │ │ └─ batch_results.csv │
│ ├─ recommendation    │ └──────────────────────┘
│ ├─ experiment_group  │           │
│ └─ show_to_frm       │           │
└──────────────────────┘           │
          │                        │
          ▼                        ▼
┌──────────────────────┐ ┌──────────────────────┐
│  Patient Watch UI    │ │  Athena / QuickSight │
│  (FRM Interface)     │ │  (Analytics)         │
└──────────────────────┘ └──────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              WEEKLY MODEL RETRAINING (SageMaker)             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ 1. Extract training data (S3 + RDS outcomes)        │    │
│  │ 2. Train propensity score model                     │    │
│  │ 3. Apply IPTW weighting                             │    │
│  │ 4. Train T-Learner (Model_T, Model_C)               │    │
│  │ 5. Validate on holdout set                          │    │
│  │ 6. Run sensitivity analysis                         │    │
│  │ 7. If metrics pass: Upload to S3 staging            │    │
│  │ 8. Notify team for review via SNS                   │    │
│  └────────────────────────────────────────────────────┘    │
│  Trigger: Weekly on Sunday at 3:00 AM (EventBridge)        │
│  Compute: SageMaker Training Job (ml.m5.xlarge, spot)      │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY (S3)                       │
│  s3://claritas-ml-models/intervention-recommender/           │
│  ├── production/                                             │
│  │   ├── model_v1.2.3.pkl (current)                         │
│  │   ├── propensity_model_v1.2.3.pkl                        │
│  │   └── metadata.json (metrics, timestamp, approval)       │
│  ├── staging/                                                │
│  │   └── model_v1.2.4_candidate.pkl (awaiting review)       │
│  └── archive/                                                │
│      └── model_v1.2.2.pkl (previous version)                 │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                MONITORING & ALERTING (CloudWatch)            │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Real-time Metrics:                                  │    │
│  │ - Lambda execution time, errors, throttles          │    │
│  │ - Number of patients scored                         │    │
│  │ - Average uplift score (drift detection)            │    │
│  │ - % patients recommended for intervention           │    │
│  │ - Feature distribution shifts (KL divergence)       │    │
│  │ - Prediction distribution shifts                    │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Alarms → SNS → PagerDuty / Email:                   │    │
│  │ - Lambda failure rate >1%                           │    │
│  │ - Scoring duration >60 seconds                      │    │
│  │ - Feature null rate >15%                            │    │
│  │ - Uplift score drift (>2 std dev)                   │    │
│  │ - Weekly training job failure                       │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│          A/B TESTING & EXPERIMENTATION PLATFORM              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Config Store: s3://config/ab_test.json              │    │
│  │ - Active experiments                                │    │
│  │ - Treatment/control split rules                     │    │
│  │ - Stratification logic                              │    │
│  └────────────────────────────────────────────────────┘    │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Analysis Pipeline (Weekly):                         │    │
│  │ - Query outcomes from RDS/S3 via Athena             │    │
│  │ - Calculate metrics by experiment arm               │    │
│  │ - Statistical testing (t-tests, bootstrap CIs)      │    │
│  │ - Generate report for stakeholders                  │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

### 6.3 Key AWS Services & Cost Estimate

| Service | Purpose | Configuration | Monthly Cost |
|---------|---------|---------------|--------------|
| **S3** | Feature store, model registry, predictions | Standard + Intelligent-Tiering | $30-50 |
| **Lambda** | Daily scoring (2 AM), feature generation | 1024 MB, 5 min timeout, 1/day | $10-20 |
| **RDS Aurora Serverless v2** | Recommendation results, outcomes | 0.5-2 ACU auto-scaling | $50-100 |
| **SageMaker Training** | Weekly model retraining | ml.m5.xlarge spot (2 hrs/week) | $15-25 |
| **EventBridge** | Scheduling (daily scoring, weekly training) | 10 rules | $1 |
| **CloudWatch** | Logs, metrics, alarms | 10 GB logs, 20 alarms | $20-30 |
| **Athena** | Ad-hoc analysis, A/B test queries | 50 GB scanned/month | $5-10 |
| **Glue** | ETL for feature engineering (optional) | 5 DPU-hours/day | $40-60 |
| **Secrets Manager** | Database credentials | 5 secrets | $2 |
| **SNS** | Alerting | 1000 notifications/month | $1 |
| **VPC** | Network isolation (if required) | 1 NAT gateway | $30-40 |

**Total Estimated Cost: $200-350/month per environment**  
(Multiply by 2-3 for dev/staging/prod)

**Cost Optimization:**
- Use S3 Intelligent-Tiering for automatic archiving
- Lambda reserved concurrency = 1 (only need one concurrent execution)
- SageMaker spot instances (70% savings on training)
- Aurora serverless auto-scales to near-zero during off-peak
- Athena: Partition S3 data by date to reduce scan costs

---

## 7. A/B Testing Framework

### 7.1 Experimental Design

**Primary Hypothesis:**  
Uplift-based intervention recommendations reduce 30-day discontinuation rates by ≥5 percentage points compared to rule-based recommendations, while maintaining or improving FRM productivity.

**Randomization Strategy:**
```python
# Deterministic assignment based on patient_id hash
def assign_experiment_group(patient_id, control_percentage=0.20):
    """
    Ensures same patient always in same group across days
    Allows for stratification by risk quintile
    """
    hash_value = hash(str(patient_id) + "experiment_seed_12345")
    return "control" if (hash_value % 100) < (control_percentage * 100) else "treatment"
```

**Arms:**
- **Control (20%):** Rule-based recommendations (baseline)
- **Treatment (80%):** Uplift model recommendations

**Stratification:** By risk quintile to ensure balance across risk levels

**Duration:** 8-12 weeks minimum (to capture 1-2 full refill cycles)

**Sample Size Calculation:**
```
Assumptions:
- Baseline discontinuation rate: 20%
- Minimum detectable effect: 5 percentage points (25% relative reduction)
- Power: 80%
- Alpha: 0.05 (two-tailed)
- Expected sample: 2,400 patients

Power analysis:
- Control: 480 patients
- Treatment: 1,920 patients
- Adequate power to detect 5pp difference
```

### 7.2 Metrics & Analysis

**Primary Outcome:**
- 30-day discontinuation rate (binary)

**Secondary Outcomes:**
- 90-day discontinuation rate
- 30-day adherence (MPR)
- Time to discontinuation (survival analysis)
- Re-initiation rate (if discontinued)

**Operational Metrics:**
- Interventions per patient per month
- FRM time spent on prioritization
- Contact success rate (reached patient)
- Intervention acceptance rate

**Guardrail Metrics:**
- Patient complaints about over-contacting
- FRM satisfaction score
- System latency and uptime

**Statistical Analysis Plan:**
```python
# Primary Analysis (Intent-to-Treat)
from scipy.stats import ttest_ind, chi2_contingency

# 1. Baseline balance check
balance_check(control_df, treatment_df, covariates)

# 2. Primary outcome
control_disc_rate = control_df['discontinued_30d'].mean()
treatment_disc_rate = treatment_df['discontinued_30d'].mean()
effect_size = treatment_disc_rate - control_disc_rate
p_value = chi2_contingency(contingency_table)

# 3. Confidence interval (bootstrap)
ci_lower, ci_upper = bootstrap_ci(effect_size, n_bootstrap=10000)

# 4. Subgroup analysis (CATE)
heterogeneous_effects = estimate_cate_by_subgroup(df, subgroups=['risk_quintile', 'age_group'])

# 5. Sensitivity analysis
sensitivity_to_unmeasured_confounding(effect_size, se)
```

**Reporting Cadence:**
- **Weekly:** Operational metrics (intervention counts, system health)
- **Bi-weekly:** Interim analysis (early stopping if harm detected)
- **End of experiment:** Full statistical analysis and recommendation

### 7.3 Decision Criteria

**Success Criteria (Proceed to Full Rollout):**
✓ Treatment arm reduces 30-day discontinuation by ≥5pp (p<0.05)  
✓ No increase in intervention fatigue (contacts per patient stable)  
✓ FRM satisfaction ≥80%  
✓ No guardrail metric violations  
✓ Cost per avoided discontinuation <$500  

**Iteration Criteria (Refine Model):**
⚠️ Effect size 2-5pp (marginal improvement)  
⚠️ Significant heterogeneity (works for some subgroups, not others)  
⚠️ FRM adoption <70%  

**Stop Criteria (Rollback):**
✗ No significant effect (p>0.10) after 8 weeks  
✗ Increase in discontinuation rate (harm)  
✗ Guardrail violations (patient complaints spike)  
✗ Technical failures >1% of scoring runs  

---

## 8. Model Governance & Risk Management

### 8.1 Model Governance Framework

**Ownership & Accountability:**
| Role | Responsibility |
|------|----------------|
| **Model Owner** | Data Science Lead - overall model performance and business outcomes |
| **Model Developer** | ML Engineers - development, training, deployment |
| **Model Validator** | Senior Data Scientist (independent) - validation, bias testing |
| **Business Owner** | Clinical Program Director - business impact, FRM feedback |
| **Compliance Officer** | Legal/Compliance - HIPAA, bias, ethical review |

**Model Documentation (Required Artifacts):**
1. **Model Card:** Algorithm, training data, features, performance metrics
2. **Validation Report:** Holdout performance, subgroup analysis, bias testing
3. **Causal Inference Report:** Propensity score diagnostics, sensitivity analysis
4. **A/B Test Report:** Experimental design, results, statistical tests
5. **Deployment Checklist:** Infrastructure, monitoring, rollback plan
6. **Change Log:** Version history with justification for updates

**Review & Approval Process:**
```
Model Training → Validation → Staging Deployment → Review Meeting → Production
     (DS)          (DS)            (DevOps)         (Committee)       (DevOps)
     
Review Committee:
- Data Science Lead (chair)
- Clinical Program Director
- Compliance Officer
- Engineering Manager
```

**Approval Criteria:**
- Validation AUC ≥0.70 (uplift AUC or qini coefficient)
- No significant bias across protected subgroups
- Propensity score balance (SMD <0.1)
- Sensitivity analysis passes (E-values >2.0)
- A/B test shows positive impact (if available)
- Business owner sign-off

### 8.2 Bias & Fairness Monitoring

**Protected Subgroups:**
- Age: <30, 30-50, 50-65, >65
- Gender: Male, Female, Other
- Geographic region: Urban, Suburban, Rural
- Insurance type: Commercial, Medicare, Medicaid, Patient Pay

**Fairness Metrics (Quarterly Review):**
1. **Demographic Parity:** P(recommended | group A) ≈ P(recommended | group B)
2. **Equalized Odds:** FPR and TPR similar across groups
3. **Calibration:** Predicted probabilities match observed rates per group
4. **Outcome Fairness:** Treatment effect similar across groups (CATE analysis)

**Mitigation Strategies:**
- Post-processing: Adjust thresholds per group if disparity >10%
- Reweighting: Oversample underrepresented groups in training
- Fairness constraints: Add demographic parity constraint to optimization
- Feature audit: Remove features with high correlation to protected attributes

### 8.3 Monitoring & Maintenance

**Daily Monitoring (Automated):**
- Lambda execution success rate (>99%)
- Feature null rates (<15%)
- Prediction distribution (flag if >2 std dev shift)
- Number of patients scored (should match active patient count ±5%)

**Weekly Monitoring (Manual Review):**
- Average uplift score (drift detection)
- % patients recommended for intervention (typical range: 10-20%)
- Intervention outcome rates (success, no answer, refused)
- FRM feedback scores

**Monthly Monitoring (Data Science Review):**
- Feature importance stability (SHAP values)
- Subgroup performance (CATE by risk/age/region)
- Prediction calibration (Brier score)
- A/B test interim results

**Quarterly Governance Review:**
- Full model validation on recent data
- Bias and fairness audit
- Business impact assessment (ROI, clinical outcomes)
- Compliance review
- Model refresh decision (retrain vs. rebuild)

**Retraining Triggers:**
- Scheduled: Weekly retraining on latest data
- Performance degradation: AUC drops >5 points
- Data drift: Feature distributions shift significantly
- Concept drift: Outcome patterns change (e.g., new drug formulation)

**Model Retirement Criteria:**
- Consistently underperforms baseline (3 months)
- Better alternative model available
- Business process changes (e.g., automated interventions replace FRMs)
- Regulatory changes require new approach

### 8.4 Incident Response Plan

**Severity Levels:**
| Level | Definition | Example | Response Time |
|-------|------------|---------|---------------|
| **P0 - Critical** | Patient safety risk or complete system outage | Model recommends harmful interventions | <1 hour |
| **P1 - High** | Significant performance degradation | Scoring fails for >50% of patients | <4 hours |
| **P2 - Medium** | Partial degradation | Latency spike, minor accuracy drop | <24 hours |
| **P3 - Low** | Minor issues, no user impact | Logging failure, cosmetic UI issue | <1 week |

**Incident Response Workflow:**
1. **Detection:** Automated alert or manual report
2. **Triage:** On-call engineer assesses severity
3. **Escalation:** P0/P1 → page model owner + business owner
4. **Mitigation:** Rollback to previous model version if needed
5. **Root Cause Analysis:** Investigate within 48 hours
6. **Remediation:** Fix issue, test, deploy
7. **Post-Mortem:** Document learnings, update playbook

**Rollback Plan:**
- Previous model version stored in S3 (versioned)
- Lambda environment variable points to current model S3 path
- To rollback: Update environment variable, redeploy (5 min)
- Fallback: Revert to rule-based scoring (zero-downtime)

### 8.5 Ethical Considerations

**Transparency:**
- FRMs can view explanation for each recommendation (SHAP values)
- Patients can request information on how system uses their data (HIPAA right)

**Autonomy:**
- FRMs retain full discretion to override recommendations
- No penalties for ignoring model suggestions (at least in pilot phase)

**Beneficence:**
- Model optimizes for patient health outcomes (adherence, not cost)
- A/B testing ensures we "do no harm" before full rollout

**Justice:**
- Fairness monitoring ensures equitable access to interventions
- Subgroup analysis prevents systematic underserving of any population

**Privacy:**
- HIPAA-compliant infrastructure (BAA with AWS)
- PHI encrypted at rest (S3 SSE-KMS) and in transit (TLS)
- Minimum necessary data principle (only use features required for prediction)
- De-identification for any external sharing or publications

---

## 9. Success Metrics & KPIs

### 9.1 North Star Metrics (6-Month Goals)

**Clinical Impact:**
- **Primary:** 15-25% reduction in preventable discontinuations  
  *Baseline: 20% discontinuation rate → Target: 15-17%*
- **Secondary:** 10-15% improvement in 90-day adherence (MPR)  
  *Baseline: 75% MPR → Target: 82-86%*

**Operational Efficiency:**
- **FRM Productivity:** 40% reduction in time spent on patient prioritization  
  *Baseline: 30 min/day → Target: 18 min/day*
- **Intervention Quality:** 25% increase in successful patient contacts  
  *Baseline: 40% contact rate → Target: 50%*

**Business Value:**
- **ROI:** $150K-$300K annual value per brand  
  *Based on: Avoided discontinuations × Lifetime value per patient*
- **Cost per Avoided Discontinuation:** <$500  
  *Total program cost / Number of discontinuations prevented*

**Model Performance:**
- **Uplift AUC:** >0.75 on holdout data
- **Calibration:** Brier score <0.20
- **Fairness:** Demographic parity within ±10% across subgroups

### 9.2 Leading Indicators (Weekly/Monthly)

**Model Health:**
- Feature null rate <15%
- Prediction drift (KL divergence) <0.05
- Uplift score distribution stable (mean ±1 std dev)

**System Reliability:**
- Uptime >99.5%
- Scoring latency <60 seconds for 3000 patients
- Zero P0 incidents per month

**User Adoption:**
- FRM login rate >90% of days
- Recommendation acceptance rate >70%
- FRM satisfaction score >80% (quarterly survey)

### 9.3 Measurement & Reporting

**Data Collection:**
- **Outcomes:** Automatically captured from Patient Watch (discontinuation events)
- **Interventions:** CRM logs (date, type, outcome)
- **Model predictions:** Stored in RDS + S3 for every scoring run
- **User behavior:** Dashboard analytics (views, actions taken)

**Reporting Cadence:**
| Audience | Frequency | Format | Key Metrics |
|----------|-----------|--------|-------------|
| **FRMs** | Daily | Dashboard | Top 10-15 recommended patients, uplift scores |
| **Clinical Program Managers** | Weekly | Email digest | Intervention counts, success rates, model health |
| **Data Science Team** | Weekly | Automated report | Model performance, drift, errors |
| **Executive Leadership** | Monthly | Slide deck | ROI, clinical impact, A/B test results |
| **Governance Committee** | Quarterly | Full review | Compliance, bias audit, strategic recommendations |

**Dashboards:**
1. **FRM Dashboard (Patient Watch UI):**
   - Top recommended patients with uplift scores
   - Patient details, risk trajectory, intervention history
   - Action buttons (call, text, snooze)

2. **Clinical Operations Dashboard (Tableau/QuickSight):**
   - Aggregate metrics by brand, region, FRM
   - Discontinuation trends over time
   - Intervention volume and outcomes

3. **Model Monitoring Dashboard (CloudWatch):**
   - System health (uptime, latency, errors)
   - Feature distributions (drift detection)
   - Prediction distributions (concept drift)
   - A/B test interim results

4. **Executive Dashboard (QuickSight):**
   - North Star metrics progress
   - ROI calculation
   - Program status (on track / at risk)

---

## 10. Risks & Mitigation Strategies

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Insufficient historical data for causal inference** | High | Medium | - Start with 6 months of observational data<br>- Use propensity score methods<br>- Phase 1 uses simpler models |
| **Model performance degrades over time** | High | Medium | - Weekly retraining<br>- Automated drift detection<br>- Quarterly full validation |
| **AWS Lambda timeout during scoring** | Medium | Low | - Optimize feature loading (parquet)<br>- Use lightweight models<br>- Increase timeout to 15 min if needed |
| **Feature pipeline failures** | High | Low | - Alerting on ETL failures<br>- Fallback to previous day's features<br>- Data quality checks |
| **Integration issues with Patient Watch** | Medium | Low | - Early API testing<br>- Gradual rollout with single brand pilot |

### 10.2 Business & Operational Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **FRMs don't trust/adopt model recommendations** | High | Medium | - Transparent explanations (SHAP)<br>- Training and change management<br>- FRM feedback loop |
| **Intervention capacity constraints** | Medium | Medium | - Recommend top N based on FRM capacity<br>- Business rules to cap daily recommendations |
| **Patient backlash from over-contacting** | High | Low | - Intervention fatigue features<br>- Respect patient preferences<br>- Guardrail metrics |
| **No measurable impact in A/B test** | High | Low | - Power analysis ensures adequate sample<br>- Pilot shows promising signals<br>- Fallback to enhanced rule-based |
| **Budget cuts / deprioritization** | Medium | Low | - Show early wins with Phase 0<br>- Quantify ROI continuously |

### 10.3 Regulatory & Compliance Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **HIPAA violation (PHI exposure)** | Critical | Low | - AWS BAA in place<br>- Encryption at rest and in transit<br>- Access controls and logging |
| **Algorithmic bias (unfair treatment)** | High | Medium | - Quarterly bias audits<br>- Fairness constraints in model<br>- Subgroup monitoring |
| **Lack of model transparency** | Medium | Low | - Model documentation (model cards)<br>- Explainability (SHAP values)<br>- Governance review process |
| **Off-label promotion concerns** | Medium | Low | - Model recommends engagement, not specific advice<br>- Legal review of UI language |

### 10.4 Data Quality Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Incomplete outcome data (discontinuations not captured)** | High | Medium | - Cross-reference multiple data sources<br>- Phone verification by FRMs<br>- 30-day lag for outcome certainty |
| **Intervention data not logged consistently** | High | Medium | - CRM integration improvements<br>- FRM training on data entry<br>- Imputation for missing data |
| **Selection bias in training data** | High | Medium | - Propensity score weighting (IPTW)<br>- Sensitivity analysis<br>- A/B test validates in production |

---

## 11. Change Management & Training

### 11.1 Stakeholder Communication Plan

**Pre-Launch (Weeks 1-4):**
- **Executive briefing:** Vision, business case, ROI projections
- **Clinical program managers:** Detailed walkthrough, timeline, success metrics
- **FRM town hall:** Introduction to concept, address concerns, gather input
- **IT/DevOps:** Technical architecture review, integration planning

**Launch (Week 5):**
- **FRM training sessions:** 2-hour workshop on how to use recommendations
- **Quick reference guide:** One-pager for daily workflow
- **Pilot kickoff:** Weekly check-ins with pilot FRMs

**Post-Launch (Ongoing):**
- **Weekly office hours:** Open Q&A for FRMs
- **Monthly performance reviews:** Share results with all stakeholders
- **Quarterly lunch & learn:** Advanced tips, success stories, model updates

### 11.2 FRM Training Program

**Learning Objectives:**
1. Understand what uplift score means (probability of incremental benefit)
2. Navigate the recommendation dashboard
3. Interpret patient-level explanations
4. Know when to override recommendations
5. Provide feedback on model quality

**Training Modules:**
1. **Intro to Uplift Modeling (20 min):**
   - Why not all high-risk patients need intervention
   - How the model identifies incremental benefit
   - Examples of high uplift vs. low uplift patients

2. **Dashboard Walkthrough (30 min):**
   - Accessing daily recommendations
   - Sorting, filtering, and prioritizing patients
   - Viewing patient details and intervention history
   - Marking actions taken

3. **Interpretation & Overrides (30 min):**
   - Reading SHAP explanations
   - Scenarios to override model (e.g., known patient preferences)
   - Using clinical judgment with model guidance

4. **Data Quality & Feedback (20 min):**
   - Importance of logging outcomes in CRM
   - How to report model errors or surprising recommendations
   - Continuous improvement through feedback

5. **Hands-On Practice (20 min):**
   - Simulated patient scenarios
   - Practice making decisions with model support

**Certification:** Pass quiz (80% required) + shadow experienced FRM for 1 week

### 11.3 Change Management Strategy

**Key Challenges:**
- FRMs may resist algorithmic guidance ("AI replacing my judgment")
- Concern about reduced autonomy
- Fear of being evaluated by model adoption metrics

**Strategies:**
1. **Positioning:** "AI augments, not replaces" - FRMs retain full decision authority
2. **Co-creation:** Involve FRMs in pilot design and feedback loops
3. **Quick wins:** Show time savings and success stories early
4. **Transparency:** Explain model limitations and when to trust/override
5. **No penalties:** FRM performance not tied to model adherence in first 6 months

---

## 12. Next Steps & Timeline

### 12.1 Immediate Actions (Week 1-2)

- [ ] **Kick-off meeting** with Data Science, Engineering, Clinical Operations
- [ ] **Data audit:** Assess quality and completeness of historical intervention data
- [ ] **AWS setup:** Provision S3, Lambda, RDS infrastructure (dev environment)
- [ ] **Stakeholder alignment:** Present this plan to executive sponsor for approval
- [ ] **Hire/assign resources:** 1 ML Engineer, 1 Data Scientist, 0.5 DevOps Engineer

### 12.2 6-Month Roadmap

| Phase | Timeline | Key Milestones |
|-------|----------|----------------|
| **Phase 0: Foundation** | Weeks 1-4 | ✓ Infrastructure setup<br>✓ Data pipeline<br>✓ Rule-based model v0.1<br>✓ Pilot with 1 brand |
| **Phase 1: Model Development** | Weeks 5-12 | ✓ Causal inference analysis<br>✓ Uplift model trained<br>✓ Offline validation<br>✓ Governance approval |
| **Phase 2: A/B Test** | Weeks 13-24 | ✓ Production deployment<br>✓ A/B test (8 weeks)<br>✓ Statistical analysis<br>✓ Go/no-go decision |
| **Phase 3: Full Rollout** | Weeks 25-26 | ✓ Scale to all brands<br>✓ MLOps automation<br>✓ Training all FRMs |

### 12.3 Resource Requirements

**Team:**
- **ML Engineer (1 FTE):** Model development, deployment, monitoring
- **Data Scientist (1 FTE):** Causal inference, A/B test analysis, validation
- **DevOps Engineer (0.5 FTE):** AWS infrastructure, CI/CD, monitoring
- **Product Manager (0.5 FTE):** Requirements, stakeholder management, roadmap
- **Clinical SME (0.25 FTE):** Domain expertise, FRM liaison, training

**Budget:**
- **AWS Infrastructure:** $600-1,000/month (dev + staging + prod)
- **Third-party tools:** $500/month (e.g., DataDog for monitoring, optional)
- **Training & change management:** $10K (one-time)
- **Contingency:** 20% buffer

**Total 6-Month Budget:** ~$50-70K

---

## 13. Appendices

### Appendix A: Technical Glossary

- **Uplift Modeling:** Causal ML technique to predict treatment effect heterogeneity
- **T-Learner:** Two-model approach (one for treated, one for control) to estimate uplift
- **IPTW:** Inverse Propensity of Treatment Weighting, adjusts for confounding
- **AIPW:** Augmented Inverse Propensity Weighting, doubly robust estimator
- **CATE:** Conditional Average Treatment Effect, treatment effect per subgroup
- **SMD:** Standardized Mean Difference, balance metric for covariates (<0.1 is good)
- **Qini Curve:** Uplift model evaluation metric (like ROC for uplift)

### Appendix B: References

1. **Causal Inference Methods:**
   - Hernán MA, Robins JM. *Causal Inference: What If.* 2020.
   - Künzel SR, et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." *PNAS*, 2019.

2. **Uplift Modeling in Healthcare:**
   - Radcliffe NJ. "Using control groups to target on predicted lift." *Direct Marketing Association*, 2007.
   - Athey S, Wager S. "Estimating treatment effects with causal forests." *Annals of Statistics*, 2019.

3. **MLOps Best Practices:**
   - Sculley D, et al. "Hidden technical debt in machine learning systems." *NeurIPS*, 2015.
   - AWS Well-Architected Framework for Machine Learning. 2023.

4. **Healthcare AI Governance:**
   - FDA. "Clinical Decision Support Software: Guidance for Industry." 2022.
   - HIMSS. "AI in Healthcare: Trust, Adoption, and Ethics." 2023.

### Appendix C: Contact Information

**Project Leadership:**
- **Project Sponsor:** [Name], VP Clinical Operations
- **Technical Lead:** [Name], SR Director of Data Science
- **Business Owner:** [Name], CPO, Specialty Pharmacy

**Support:**
- **Weekly Sync:** Tuesdays 10:00 AM EST

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| **Executive Sponsor** | | | |
| **Data Science Lead** | | | |
| **Clinical Operations Director** | | | |
| **Compliance Officer** | | | |
| **Engineering Manager** | | | |

---

**Document Control:**
- **Version:** 1.0
- **Last Updated:** November 11, 2025
- **Next Review:** December 11, 2025 (monthly during planning phase)
- **Document Owner:** Data Science Team
- **Storage Location:** SharePoint / Confluence

---


