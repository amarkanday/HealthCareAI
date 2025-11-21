# Selection Bias in Patient Watch Tower Impact Measurement
## A Deep Dive for Claritas Rx Analytics Team

**Version:** 1.0  
**Date:** November 2025  
**Author:** Claritas Rx Analytics Team

---

## Executive Summary

**The Problem:**  
Claritas Rx currently measures Field Reimbursement Manager (FRM) impact by comparing outcomes between patients who received interventions and those who didn't. This "treated vs untreated" approach suffers from **severe selection bias** because FRMs systematically choose which patients to help based on risk scores, perceived "savability," and other factors. As a result, our current impact estimates are **likely biased by 30-50% or more**—either over-estimating or under-estimating the true incremental value we deliver.

**Why It Matters:**  
Biased estimates affect pricing decisions, product expansion strategies, resource allocation, and credibility with sophisticated pharma clients. As the market evolves, clients will increasingly scrutinize our methodology.

**The Solution:**  
We need to adopt causal inference methods that account for selection bias:
- **Immediate:** Implement propensity score adjustments and regression-based methods
- **Near-term:** Build uplift modeling capabilities to guide FRM prioritization
- **Long-term:** Run randomized controlled trials (RCTs) where feasible

**Bottom Line:**  
This is not just a technical issue—it's a strategic imperative for defensible impact measurement and competitive differentiation.

---

## 1. The Patient Watch Tower Workflow

### 1.1 System Overview

```
┌────────────────────────────────────────────────────────────────┐
│  DATA AGGREGATION LAYER                                         │
│  • Hub data (patient eligibility, benefits checks)              │
│  • Specialty pharmacy data (fills, claims, adherence)           │
│  • Provider data (prescriptions, treatment plans)               │
│  • Claims data (medical/pharmacy utilization)                   │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  PREDICTIVE MODELS (Patient Watch Tower)                        │
│  • Abandonment Risk: P(patient never starts therapy)            │
│  • Discontinuation Risk: P(patient stops within 90/180 days)    │
│  • PA Denial Risk: P(prior authorization denied)                │
│                                                                  │
│  Inputs: Demographics, clinical, payer, site, prior history     │
│  Outputs: Risk scores (0-1), risk bands (Low/Medium/High)       │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  FRM DASHBOARD                                                  │
│  • View at-risk patients sorted by risk score                  │
│  • Filter by site, payer, indication, enrollment window         │
│  • See patient details: script date, payer, last contact        │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  FRM DECISION PROCESS                                           │
│  • Review risk list                                             │
│  • Apply judgment: "Is this patient savable?"                   │
│  • Consider: Time constraints, past responsiveness, complexity  │
│  • DECIDE: Intervene or Not                                     │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                   ┌───────┴────────┐
                   │                │
                   ▼                ▼
          ┌───────────────┐  ┌────────────────┐
          │  INTERVENE    │  │  DO NOT        │
          │               │  │  INTERVENE     │
          └───────┬───────┘  └────────────────┘
                  │
                  ▼
┌────────────────────────────────────────────────────────────────┐
│  FRM INTERVENTIONS                                              │
│  • Phone outreach (educate, troubleshoot)                       │
│  • Benefits investigation (copay assistance, PAP)               │
│  • PA support (documentation, appeals)                          │
│  • Coordination (pharmacy, prescriber, payer)                   │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│  OUTCOMES                                                        │
│  • Successfully started therapy? (Yes/No)                        │
│  • PA approved? (Yes/No)                                         │
│  • Still on therapy at 90/180 days? (Yes/No)                    │
└────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Definitions

**Treatment:**  
FRM intervention—any proactive outreach, benefits investigation, PA support, or coordination conducted by an FRM for a specific patient. Logged in CRM with date, type, and outcome.

**Outcome:**  
Binary success indicator depending on context:
- For abandonment risk: Did patient successfully start therapy? (1 = Yes, 0 = No)
- For discontinuation risk: Is patient still on therapy at 90/180 days? (1 = Yes, 0 = No)
- For PA risk: Was PA approved? (1 = Yes, 0 = No)

**Observed Covariates:**  
Variables measured before treatment that might affect both treatment assignment and outcomes:
- `risk_score`: Model-predicted probability of negative outcome (0-1)
- `risk_band`: Categorical (Low < 0.33, Medium 0.33-0.67, High > 0.67)
- `payer_type`: Commercial, Medicare, Medicaid, Patient Pay
- `site_type`: Academic Medical Center, Community Hospital, Specialty Clinic
- `channel`: Hub referral vs non-Hub
- `age`, `disease_severity`, `comorbidities`
- `days_since_script`: How long since prescription written
- `prior_attempts`: Number of previous FRM contact attempts

### 1.3 Current Impact Measurement Approach

**What we do now:**

```python
# Simplified pseudocode
treated = patients[patients['frm_intervention'] == 1]
untreated = patients[patients['frm_intervention'] == 0]

success_rate_treated = treated['outcome'].mean()
success_rate_untreated = untreated['outcome'].mean()

estimated_impact = success_rate_treated - success_rate_untreated
```

**Example result:**
- Treated: 75% success rate (300 out of 400 patients)
- Untreated: 60% success rate (360 out of 600 patients)
- **Reported impact: +15 percentage points**

**This seems straightforward. What's the problem?**

---

## 2. Why Treatment Is Not Randomly Assigned

### 2.1 How FRMs Actually Select Patients

FRMs are experienced professionals with limited time (~150-400 patients to monitor, ~4 hours/day for outreach). They develop heuristics for prioritization:

**They tend to INTERVENE on:**
- ✅ Medium-risk patients (0.35-0.65 risk score) who "just need a nudge"
- ✅ Patients with **solvable** barriers (missing PA documentation, unclear benefits)
- ✅ Patients who **answer calls** (demonstrated engagement)
- ✅ Commercial insurance (simpler navigation, higher success probability)
- ✅ Patients early in the journey (before abandonment/discontinuation is irreversible)
- ✅ Patients at sites with good FRM relationships

**They tend to SKIP:**
- ❌ Very low-risk patients (<0.20) who will probably succeed anyway
- ❌ Very high-risk patients (>0.80) who seem "lost causes"
- ❌ Patients who don't answer after 3 attempts (low engagement signal)
- ❌ Complex Medicaid cases (time-consuming, lower success rates historically)
- ❌ Patients far into abandonment/discontinuation process (too late)
- ❌ Patients at sites with poor responsiveness

**Result:**  
The "treated" group is systematically different from the "untreated" group on both **observed** factors (risk score, payer type, etc.) and potentially **unobserved** factors (patient motivation, family support, transportation access).

### 2.2 Visual Representation of Selection

```
Population of At-Risk Patients (N=1,000)

Risk Score Distribution:
0.0 ├───────────────────────────────────────────────────┤ 1.0
    Low Risk          Medium Risk           High Risk
    (N=300)           (N=500)               (N=200)

FRM Intervention Probability by Risk Band:
    Low: ▓░░░░░░░░░  10% get treated (30 patients)
 Medium: ▓▓▓▓▓▓▓▓░░  80% get treated (400 patients)
   High: ▓▓▓░░░░░░░  30% get treated (60 patients)

Resulting Groups:
┌────────────────────────────────────────────────────────┐
│ TREATED (N=490)                                         │
│ • 30 Low-risk  (6%)                                     │
│ • 400 Medium-risk  (82%)   ← Concentrated here!        │
│ • 60 High-risk  (12%)                                   │
│ Mean risk score: 0.48                                   │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│ UNTREATED (N=510)                                       │
│ • 270 Low-risk  (53%)   ← Many low-risk                │
│ • 100 Medium-risk  (20%)                                │
│ • 140 High-risk  (27%)  ← Many high-risk               │
│ Mean risk score: 0.42 (but bimodal!)                   │
└────────────────────────────────────────────────────────┘
```

**Key insight:** The treated and untreated groups have **different risk distributions** even though their mean risk scores might be similar. The untreated group is **bimodal**—it includes many patients who are very low risk (will succeed anyway) and many who are very high risk (will fail anyway). The treated group is concentrated in the **middle** where outcomes are uncertain.

---

## 3. Why This Creates Bias: A Toy Example

### 3.1 Setup

Let's use a simple scenario with concrete numbers to see how selection bias emerges.

**Assumptions:**
- 1,000 patients total
- 3 risk bands: Low (300 patients), Medium (500 patients), High (200 patients)
- **True treatment effect** (the causal impact of FRM intervention):
  - Low risk: +2pp (baseline 90% → 92% with treatment)
  - Medium risk: +15pp (baseline 60% → 75% with treatment)
  - High risk: +10pp (baseline 30% → 40% with treatment)
- FRM intervention rates:
  - Low risk: 10% (30 out of 300 get treated)
  - Medium risk: 80% (400 out of 500 get treated)
  - High risk: 30% (60 out of 200 get treated)

### 3.2 Calculate True Average Treatment Effect

**Weighted average (population ATE):**
```
ATE = (300/1000)*2pp + (500/1000)*15pp + (200/1000)*10pp
    = 0.6pp + 7.5pp + 2.0pp
    = 10.1pp
```

So the **true average treatment effect** if we treated everyone is **+10.1 percentage points**.

### 3.3 Calculate Outcomes for Each Group

**UNTREATED GROUP (510 patients):**

| Risk Band | N Untreated | Baseline Success Rate | Successes |
|-----------|-------------|-----------------------|-----------|
| Low | 270 | 90% | 243 |
| Medium | 100 | 60% | 60 |
| High | 140 | 30% | 42 |
| **Total** | **510** | — | **345** |

**Untreated success rate:** 345 / 510 = **67.6%**

**TREATED GROUP (490 patients):**

| Risk Band | N Treated | Success Rate With Treatment | Successes |
|-----------|-----------|------------------------------|-----------|
| Low | 30 | 92% | 28 |
| Medium | 400 | 75% | 300 |
| High | 60 | 40% | 24 |
| **Total** | **490** | — | **352** |

**Treated success rate:** 352 / 490 = **71.8%**

### 3.4 Naive Estimate vs Truth

**Naive "treated vs untreated" comparison:**
```
71.8% - 67.6% = +4.2pp
```

**True average treatment effect (from 3.2):**
```
+10.1pp
```

**BIAS:**
```
4.2pp - 10.1pp = -5.9pp
```

We are **UNDER-estimating** the true impact by **58%**!

### 3.5 Why Is the Naive Estimate So Far Off?

The problem is **compositional bias**:

- The **untreated** group has many low-risk patients (270 out of 510 = 53%) who have high baseline success rates (90%) even without treatment.
- The **treated** group is dominated by medium-risk patients (400 out of 490 = 82%) who have lower baseline success rates (60% without treatment, 75% with).

So even though treatment is helping the medium-risk patients substantially (+15pp), the naive comparison mixes this with the fact that untreated patients include many "sure things" (low-risk) that boost the untreated group's average.

**Intuition:**  
It's like comparing the test scores of students who got tutoring (mostly C students) to those who didn't (mix of A students and F students). The untutored group might have a decent average because of the A students, but that doesn't mean tutoring doesn't work—it just means you're not comparing apples to apples.

---

## 4. Formal Definition of Selection Bias

### 4.1 Causal Inference Framework

Let's formalize the problem using the **potential outcomes framework**:

**Notation:**
- $Y_i(1)$ = outcome for patient $i$ IF they receive treatment
- $Y_i(0)$ = outcome for patient $i$ IF they do NOT receive treatment
- $T_i$ = 1 if patient $i$ actually receives treatment, 0 otherwise
- $Y_i^{obs}$ = observed outcome = $T_i \cdot Y_i(1) + (1-T_i) \cdot Y_i(0)$
- $X_i$ = vector of observed covariates for patient $i$

**Individual treatment effect (ITE):**
```
τ_i = Y_i(1) - Y_i(0)
```

This is the causal effect of treatment for patient $i$. **Problem:** We never observe both $Y_i(1)$ and $Y_i(0)$ for the same patient ("fundamental problem of causal inference").

**Average treatment effect (ATE):**
```
ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]
```

This is what we want to estimate—the average causal effect across the population.

### 4.2 What We Actually Estimate (Naive Approach)

**Naive estimate:**
```
E[Y | T=1] - E[Y | T=0]
```

This is the difference in **observed** outcomes between treated and untreated groups.

**Decomposition:**
```
E[Y | T=1] - E[Y | T=0]
  = E[Y(1) | T=1] - E[Y(0) | T=0]
  = E[Y(1) | T=1] - E[Y(0) | T=1]  ← ATE for the treated
    + E[Y(0) | T=1] - E[Y(0) | T=0]  ← SELECTION BIAS!
```

The second term is **selection bias**: the difference in **baseline** outcomes (without treatment) between those who got treated and those who didn't.

**When is selection bias zero?**
- When $E[Y(0) | T=1] = E[Y(0) | T=0]$
- This happens if treatment assignment is **random** (independent of potential outcomes)
- In our case, treatment is NOT random → selection bias ≠ 0

### 4.3 Visualizing Selection Bias

```
Observed Difference = E[Y | T=1] - E[Y | T=0]
                    = Causal Effect + Selection Bias

       ┌────────────────────────────────┐
       │ What We Observe (Naive)        │
       │                                 │
 Treated│     ╔═══════════════╗         │
 Success│     ║    71.8%      ║         │
   Rate │     ╚═══════════════╝         │
       │                                 │
Untreated     ╔═══════════════╗         │
 Success│     ║    67.6%      ║         │
   Rate │     ╚═══════════════╝         │
       │                                 │
       │ Difference = 4.2pp              │
       └────────────────────────────────┘

       ┌────────────────────────────────┐
       │ What We Want (Causal)          │
       │                                 │
  Treated    ╔═══════════════╗         │
 Potential│  ║    71.8%      ║         │
  Outcome │  ╚═══════════════╝         │
       │                                 │
Untreated    ╔════════╗                │
 Potential│  ║ 61.7% ║ ← Counterfactual│
  Outcome │  ╚════════╝   (unobserved) │
       │                                 │
       │ True Effect = 10.1pp            │
       └────────────────────────────────┘

Selection Bias = 67.6% - 61.7% = 5.9pp
(Untreated group has better baseline than treated group)
```

---

## 5. Analytic Strategies to Address Selection Bias

### 5.1 The Gold Standard: Randomized Controlled Trial (RCT)

**Design:**
1. Identify eligible at-risk patients (e.g., all Medium-risk patients)
2. **Randomly assign** 50% to Treatment (FRM allowed to intervene) and 50% to Control (no FRM intervention, or delayed intervention)
3. Collect outcomes for both groups
4. Compare: $E[Y | T=1] - E[Y | T=0]$ now **equals** ATE because randomization eliminates selection bias

**Why it works:**
- Randomization ensures $E[Y(0) | T=1] = E[Y(0) | T=0]$ (groups identical at baseline on average)
- The only difference between groups is the treatment assignment
- Causality is unambiguous

**Practical RCT designs for Claritas Rx:**

**Option A: Patient-Level Randomization**
```
Medium-Risk Patients (N=500)
           │
      Randomize (50/50)
           │
     ┌─────┴─────┐
     ▼           ▼
Treatment     Control
 (N=250)      (N=250)
FRM may      FRM cannot
intervene    intervene
     │           │
     └─────┬─────┘
           │
    Compare Outcomes
```

**Pros:** Cleanest, most powerful  
**Cons:** Hardest to enforce (FRMs may resist "hands off" rule), ethical concerns

**Option B: Site/Geography Randomization**
```
Sites/Hubs (N=20)
           │
      Randomize (50/50)
           │
     ┌─────┴─────┐
     ▼           ▼
Treatment     Control
  Sites         Sites
  (N=10)        (N=10)
     │           │
     └─────┬─────┘
           │
  Compare Aggregate Outcomes
```

**Pros:** Easier to enforce, natural units, less ethical concern  
**Cons:** Fewer randomization units (less power), potential cross-contamination

**Option C: Stepped Wedge Design**
```
Time →
Site 1: [Control] [Control] [Treatment] [Treatment] [Treatment]
Site 2: [Control] [Treatment] [Treatment] [Treatment] [Treatment]
Site 3: [Control] [Control] [Control] [Treatment] [Treatment]
...
```

**Pros:** Everyone eventually gets treatment (ethical appeal), can compare within sites over time  
**Cons:** Complex analysis, need to account for time trends, long timeline

**Recommendation:** Start with a **pilot RCT** on one willing client, one indication, medium-risk band only (where clinical equipoise is strongest). Use patient-level or site-level randomization. Duration: 3-6 months to accumulate sufficient outcomes.

### 5.2 Observational Methods When RCT Is Infeasible

When we can't randomize (e.g., due to ethical concerns, client resistance, or operational constraints), we can use statistical methods to **adjust** for observed confounders. These methods attempt to mimic randomization using data.

#### 5.2.1 Regression Adjustment

**Idea:** Control for confounding variables in a regression model.

**Model:**
```python
# Logistic regression for binary outcome
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)

# X includes: treatment indicator, risk_score, payer_type, site_type, etc.
# Coefficient on treatment indicator = adjusted treatment effect
```

**Assumptions:**
- Correct functional form (linearity, additivity)
- No unmeasured confounders (all important variables included)
- No model misspecification

**Pros:** Simple, interpretable, widely understood  
**Cons:** Relies heavily on correct model specification; extrapolates if groups don't overlap

#### 5.2.2 Propensity Score Methods

**Idea:** Instead of controlling for many covariates, summarize them into a single score—the probability of receiving treatment.

**Step 1: Estimate Propensity Score**
```python
# Predict probability of treatment given covariates
from sklearn.linear_model import LogisticRegression

ps_model = LogisticRegression()
ps_model.fit(X, treatment)

propensity_scores = ps_model.predict_proba(X)[:, 1]
```

**Step 2: Use Propensity Scores to Adjust**

**Option A: Matching**
- For each treated patient, find an untreated patient with similar propensity score
- Compare outcomes within matched pairs
- Discard patients without good matches (improves balance but reduces sample size)

**Option B: Stratification**
- Divide patients into propensity score bins (e.g., quintiles)
- Compare treated vs untreated within each bin
- Weighted average across bins

**Option C: Inverse Probability Weighting (IPW)**
- Weight treated patients by $1 / e(X)$
- Weight untreated patients by $1 / (1 - e(X))$
- Creates a "pseudo-population" where treatment is independent of $X$

**Assumptions:**
- **Unconfoundedness:** $Y(0), Y(1) \perp T | X$ (treatment assignment independent of potential outcomes, conditional on $X$)
- **Positivity:** $0 < P(T=1|X) < 1$ for all $X$ (every type of patient has some chance of being treated and untreated)

**Pros:** Intuitive, flexible, reduces extrapolation, handles high-dimensional $X$  
**Cons:** Requires correct propensity model; sensitive to extreme weights; still assumes no unmeasured confounding

#### 5.2.3 Doubly Robust Methods

**Idea:** Combine propensity scores (treatment model) and outcome regression (outcome model). You only need ONE to be correct.

**Augmented Inverse Probability Weighting (AIPW):**
```
τ_AIPW = (1/n) Σ [
  (T_i / e(X_i)) * (Y_i - μ_1(X_i))
  - ((1-T_i) / (1-e(X_i))) * (Y_i - μ_0(X_i))
  + μ_1(X_i) - μ_0(X_i)
]
```

Where:
- $e(X_i)$ = propensity score
- $μ_1(X_i)$ = predicted outcome if treated, given $X_i$
- $μ_0(X_i)$ = predicted outcome if untreated, given $X_i$

**Why "doubly robust":**
- Consistent if EITHER $e(X)$ or $μ(X)$ is correctly specified
- If both are correct, variance is reduced

**Pros:** More robust to model misspecification, state-of-the-art in epidemiology  
**Cons:** More complex to implement, still assumes unconfoundedness

#### 5.2.4 Uplift Modeling (Heterogeneous Treatment Effects)

**Idea:** Instead of estimating a single average treatment effect, model how treatment effects **vary** across patients.

**Approaches:**
- **T-Learner:** Train two models (one for treated, one for untreated), predict both, take difference
- **S-Learner:** Train one model with treatment as a feature, compare predictions with/without treatment
- **X-Learner:** More sophisticated version of T-Learner with cross-validation
- **Causal Forests:** Tree-based methods specifically designed for heterogeneous effects

**Application for Claritas Rx:**
- Model $τ(X) = E[Y(1) - Y(0) | X]$ (treatment effect as a function of patient characteristics)
- Prioritize patients with **highest predicted uplift**, not highest risk
- Shifts focus from "who will fail?" to "who will benefit most from our help?"

**Pros:** Actionable for prioritization, identifies subgroups with high/low benefit  
**Cons:** Requires larger sample sizes, more complex validation

### 5.3 Comparison of Methods

| Method | Randomization Needed? | Key Assumption | Sample Size | Complexity | Bias Reduction |
|--------|----------------------|----------------|-------------|------------|----------------|
| **RCT** | ✅ Yes | None (design-based) | Medium | Low | Eliminates |
| **Regression Adjustment** | ❌ No | Correct model, no unmeasured confounders | Small-Medium | Low | Moderate |
| **Propensity Score (IPW)** | ❌ No | Unconfoundedness, positivity | Medium | Medium | Good |
| **Doubly Robust (AIPW)** | ❌ No | Unconfoundedness, need 1 of 2 models right | Medium-Large | High | Best (observational) |
| **Uplift Modeling** | ❌ No | Same as above + heterogeneity structure | Large | High | Good + actionable |

---

## 6. Practical Recommendations for Claritas Rx

### 6.1 Immediately Implementable (This Quarter)

**Action 1: Enhance Data Logging**
- Add "Reason for Decision" dropdown in FRM tool:
  - "High priority: likely to benefit"
  - "Too high risk: unlikely to respond"
  - "Too low risk: will succeed anyway"
  - "No capacity: out of time"
  - "Patient unresponsive"
- Log **all** contact attempts (not just successful interventions)
- Capture FRM subjective judgment separate from model risk score

**Purpose:** Better data for propensity score modeling

**Action 2: Stop Reporting Naive Estimates**
- Immediately cease reporting "treated vs untreated" differences without adjustment
- Replace with stratified analyses (compare within risk bands)
- Add disclaimer to existing reports: "Estimates not adjusted for selection bias; interpret with caution"

**Action 3: Implement Basic Regression Adjustment**
- Fit logistic regression: `outcome ~ treatment + risk_score + payer_type + site_type + ...`
- Report adjusted treatment effect coefficient
- Show both naive and adjusted estimates side-by-side for transparency

**Effort:** 1-2 weeks for implementation

### 6.2 Near-Term Implementation (Next 6 Months)

**Action 4: Build Propensity Score Pipeline**
- Develop propensity score model: `P(treatment | X)`
- Implement IPW adjustment in standard reports
- Create diagnostic plots: propensity score distributions, covariate balance
- Validate: check that treated and untreated groups are balanced after weighting

**Effort:** 1 month for robust implementation

**Action 5: Retrospective Re-Analysis**
- Re-analyze past 1-2 years of data using propensity scores and doubly robust methods
- Compare naive vs adjusted estimates across products/indications
- Identify where bias was largest
- Present revised estimates to key clients with clear methodology explanation

**Effort:** 2 months for comprehensive re-analysis

**Action 6: Pilot RCT Design**
- Identify one willing client and indication for pilot experiment
- Work with FRM leadership and client to design feasible RCT
- Target: 500-1,000 patients, 3-6 month duration
- Use results to validate observational methods

**Effort:** 3-6 months from design to results

### 6.3 Long-Term Strategic Initiatives (12+ Months)

**Action 7: Shift to Uplift-Based Prioritization**
- Build uplift models (T-Learner or Causal Forest)
- Replace "risk score" with "uplift score" in FRM dashboard
- Guide FRMs to prioritize patients with **highest expected incremental benefit**
- Measure: Are we intervening on the right patients?

**Action 8: Systematic A/B Testing Culture**
- Embed experimentation into product development
- Randomize new features (e.g., different intervention types, timing, messaging)
- Build infrastructure for multi-arm bandits (adaptive experimentation)

**Action 9: Establish Causal Inference Center of Excellence**
- Hire or train specialists in causal inference
- Create internal playbooks, code libraries, and best practices
- Offer consulting to account teams on study design
- External visibility: publish methodology papers, speak at conferences

---

## 7. Common Questions & Objections

**Q: Can't we just use A/B testing like tech companies do?**  
A: Yes, that's exactly what we should do—but we need to actually randomize treatment, which requires operational changes. The current "treated vs untreated" comparison is NOT an A/B test because treatment wasn't randomly assigned.

**Q: Isn't it unethical to withhold treatment from a control group?**  
A: This is a valid concern, but:
- Control group still receives standard care (just no extra FRM intervention)
- We can randomize only within medium-risk band (where benefit is uncertain)
- Many patients currently don't get FRM intervention anyway due to capacity constraints
- Without experiments, we can't know if we're helping—that's arguably less ethical

**Q: What if our adjusted estimates show LOWER impact than we currently report?**  
A: Then we learn which products/interventions work and which don't. That's valuable! It allows us to optimize, reprice appropriately, and focus on areas where we add real value.

**Q: Won't clients be upset if we change our methodology and numbers change?**  
A: Better to proactively update methods than have clients discover issues themselves. Frame it as: "We're adopting gold-standard causal inference methods to give you more defensible estimates."

**Q: This sounds complicated and time-consuming.**  
A: Initial setup takes time, but once built, propensity score methods are automated and actually simpler than complex business logic. The alternative is continuing with biased estimates—that's more costly long-term.

**Q: Do our competitors do this?**  
A: Some might, but many likely don't. This is an opportunity for differentiation. As pharma analytics teams get more sophisticated, methodological rigor will become table stakes.

---

## 8. Next Steps

### For Data Science Team

1. **This week:**
   - Review simulation notebook (`01_selection_bias_simulation.ipynb`)
   - Replicate bias calculations for one recent product analysis
   
2. **Next sprint:**
   - Implement basic propensity score adjustment
   - Create template code for IPW and AIPW
   
3. **Next quarter:**
   - Build systematic causal inference pipeline
   - Train team on causal inference fundamentals

### For Product/Engineering Team

1. **This sprint:**
   - Add "Reason for Decision" field to FRM tool
   - Enhance logging of contact attempts and outcomes
   
2. **Next quarter:**
   - Design dashboard for uplift-based prioritization
   - Build experiment management infrastructure

### For FRM Leadership

1. **This month:**
   - Participate in causal inference training
   - Provide input on pilot RCT design
   - Help identify client partner for experiment
   
2. **Next quarter:**
   - Test uplift-based prioritization in pilot
   - Gather feedback from FRMs on new guidance

### For Executive Leadership

1. **This month:**
   - Review revised impact estimates for top 3 products
   - Decide on messaging strategy for methodology updates
   - Allocate budget for causal inference capabilities
   
2. **Next quarter:**
   - Approve pilot RCT
   - Determine competitive positioning around analytical rigor

---

## 9. Technical Appendix

### 9.1 Python Libraries for Causal Inference

**Recommended packages:**
- `econml` (Microsoft): Comprehensive library for heterogeneous treatment effects
- `causalml` (Uber): Uplift modeling focused, good visualizations
- `dowhy` (Microsoft): Causal graph-based inference, integrates with econml
- `statsmodels`: For basic regression, matching, weighting
- `scikit-learn`: For propensity score models and machine learning

**Installation:**
```bash
pip install econml causalml dowhy statsmodels scikit-learn
```

### 9.2 Sample Code: Propensity Score Weighting

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Step 1: Fit propensity score model
X = df[['risk_score', 'payer_commercial', 'payer_medicare', 
        'site_academic', 'days_since_script']]
treatment = df['frm_intervention']

ps_model = LogisticRegression(random_state=42)
ps_model.fit(X, treatment)
propensity_scores = ps_model.predict_proba(X)[:, 1]

# Step 2: Calculate IPW weights
weights = np.where(treatment == 1, 
                   1 / propensity_scores,
                   1 / (1 - propensity_scores))

# Optional: Trim extreme weights (stabilization)
weights = np.clip(weights, 0, np.percentile(weights, 99))

# Step 3: Estimate weighted treatment effect
outcome = df['success']
treated_outcome = np.average(outcome[treatment == 1], 
                             weights=weights[treatment == 1])
untreated_outcome = np.average(outcome[treatment == 0], 
                                weights=weights[treatment == 0])

ate_ipw = treated_outcome - untreated_outcome
print(f"IPW-adjusted ATE: {ate_ipw:.3f}")
```

### 9.3 Checking Covariate Balance

```python
from scipy import stats

def check_balance(df, treatment_col, covariates, weights=None):
    """
    Check covariate balance between treated and untreated groups
    """
    results = []
    for cov in covariates:
        treated = df[df[treatment_col] == 1][cov]
        untreated = df[df[treatment_col] == 0][cov]
        
        if weights is not None:
            w_treated = weights[df[treatment_col] == 1]
            w_untreated = weights[df[treatment_col] == 0]
            mean_treated = np.average(treated, weights=w_treated)
            mean_untreated = np.average(untreated, weights=w_untreated)
            std_treated = np.sqrt(np.average((treated - mean_treated)**2, weights=w_treated))
            std_untreated = np.sqrt(np.average((untreated - mean_untreated)**2, weights=w_untreated))
        else:
            mean_treated = treated.mean()
            mean_untreated = untreated.mean()
            std_treated = treated.std()
            std_untreated = untreated.std()
        
        # Standardized mean difference (SMD)
        pooled_std = np.sqrt((std_treated**2 + std_untreated**2) / 2)
        smd = (mean_treated - mean_untreated) / pooled_std if pooled_std > 0 else 0
        
        results.append({
            'covariate': cov,
            'mean_treated': mean_treated,
            'mean_untreated': mean_untreated,
            'smd': smd,
            'balanced': abs(smd) < 0.1  # Rule of thumb: SMD < 0.1 is balanced
        })
    
    return pd.DataFrame(results)

# Before weighting
balance_unweighted = check_balance(df, 'frm_intervention', 
                                    ['risk_score', 'payer_commercial', 'site_academic'])
print("Balance before weighting:")
print(balance_unweighted)

# After IPW
balance_weighted = check_balance(df, 'frm_intervention', 
                                  ['risk_score', 'payer_commercial', 'site_academic'],
                                  weights=weights)
print("\nBalance after IPW:")
print(balance_weighted)
```

Good balance: SMD < 0.1 for all covariates after weighting.

---

## 10. References & Further Reading

### Books
- **"Causal Inference: The Mixtape"** by Scott Cunningham (free online)
- **"Causal Inference for Statistics, Social, and Biomedical Sciences"** by Imbens & Rubin
- **"The Book of Why"** by Judea Pearl (accessible introduction)

### Papers
- Rosenbaum & Rubin (1983): "The Central Role of the Propensity Score in Observational Studies for Causal Effects"
- Lunceford & Davidian (2004): "Stratification and Weighting via the Propensity Score"
- Bang & Robins (2005): "Doubly Robust Estimation in Missing Data and Causal Inference Models"

### Online Courses
- Coursera: "A Crash Course in Causality" (University of Pennsylvania)
- edX: "Causal Inference" (MIT)
- YouTube: "Causal Inference Bootcamp" (Stanford)

### Software Documentation
- econml: https://econml.azurewebsites.net/
- causalml: https://causalml.readthedocs.io/
- dowhy: https://microsoft.github.io/dowhy/

---

## Appendix: Glossary

**ATE (Average Treatment Effect):** The expected difference in outcomes if everyone received treatment vs if no one did.

**ATT (Average Treatment Effect on the Treated):** The expected difference in outcomes for those who actually received treatment.

**Confounding:** When a variable affects both treatment assignment and outcome, creating spurious associations.

**Equipoise:** Clinical uncertainty about whether treatment is beneficial; ethical foundation for randomization.

**IPTW (Inverse Probability of Treatment Weighting):** Weighting method that creates a pseudo-population where treatment is independent of covariates.

**Positivity:** Assumption that every type of patient has some chance of being both treated and untreated.

**Propensity Score:** Probability of receiving treatment given observed covariates.

**Selection Bias:** Systematic difference in baseline characteristics between treated and untreated groups due to non-random treatment assignment.

**SMD (Standardized Mean Difference):** Measure of covariate balance; |SMD| < 0.1 typically considered balanced.

**Unconfoundedness (Ignorability):** Assumption that treatment assignment is independent of potential outcomes, conditional on observed covariates.

**Uplift:** The incremental effect of treatment for an individual or subgroup.

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Next Review:** After first RCT completion

**Feedback:** Please send comments or questions to the Analytics Team via #causal-inference-questions on Slack.
