# Causal Inference Method Selection Framework
## A Practical Decision Guide for Claritas Rx Impact Measurement

**Version:** 1.0  
**Date:** November 2025  
**Purpose:** Help teams choose the right causal inference method based on constraints and context

---

## Executive Summary

Not all causal inference methods are created equal. **Randomized Controlled Trials (RCTs)** are the gold standard, but they're not always feasible in specialty pharma. This document provides:

1. **Hierarchy of evidence**: From strongest (RCT) to weakest (naive comparison)
2. **Decision flowchart**: Navigate constraints to find the right method
3. **Practical guidance**: Sample size requirements, timelines, and trade-offs
4. **Implementation roadmap**: How to execute each method

**Quick Answer:**
- **Can you randomize?** → Do an RCT or stepped wedge design
- **Can't randomize but have good data?** → Use doubly robust methods (AIPW)
- **Small sample size?** → Combine methods or use sensitivity analyses
- **Need quick answer?** → Propensity score matching (but document limitations)

---

## 1. Hierarchy of Causal Inference Methods

### Evidence Pyramid

```
                    ╔═══════════════════════╗
                    ║   RANDOMIZED          ║
                    ║   CONTROLLED TRIAL    ║  ← GOLD STANDARD
                    ║   (RCT)               ║    Unbiased, Definitive
                    ╚═══════════════════════╝
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   STEPPED WEDGE       ║
                    ║   CLUSTER RCT         ║  ← Strong Alternative
                    ╚═══════════════════════╝    Everyone gets treatment
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   INTERRUPTED         ║
                    ║   TIME SERIES         ║  ← Quasi-Experimental
                    ╚═══════════════════════╝    Before/After + Control
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   DOUBLY ROBUST       ║
                    ║   (AIPW, TMLE)        ║  ← Best Observational
                    ╚═══════════════════════╝    Two models, one must work
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   PROPENSITY SCORE    ║
                    ║   METHODS (IPW)       ║  ← Standard Observational
                    ╚═══════════════════════╝    Adjust for confounding
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   REGRESSION          ║
                    ║   ADJUSTMENT          ║  ← Basic Adjustment
                    ╚═══════════════════════╝    Assumes correct model
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   INSTRUMENTAL        ║
                    ║   VARIABLES           ║  ← Special Case
                    ╚═══════════════════════╝    If you have an IV
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   DIFFERENCE-IN-      ║
                    ║   DIFFERENCES         ║  ← Panel Data
                    ╚═══════════════════════╝    Repeated measures
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   MATCHING ONLY       ║
                    ║   (No adjustment)     ║  ← Weak
                    ╚═══════════════════════╝    High bias risk
                            ▲
                            │
                    ╔═══════════════════════╗
                    ║   NAIVE COMPARISON    ║
                    ║   (Treated vs Control)║  ← ⚠️ BIASED
                    ╚═══════════════════════╝    DO NOT USE
```

### Quality Ratings

| Method | Strength of Evidence | Bias Risk | Sample Size Need | Complexity | Cost |
|--------|---------------------|-----------|-----------------|------------|------|
| **RCT** | ⭐⭐⭐⭐⭐ | Very Low | Medium-Large | Medium | High |
| **Stepped Wedge** | ⭐⭐⭐⭐⭐ | Very Low | Medium-Large | High | High |
| **Interrupted Time Series** | ⭐⭐⭐⭐ | Low | Medium | Medium | Medium |
| **Doubly Robust (AIPW)** | ⭐⭐⭐⭐ | Low-Medium | Medium-Large | High | Low |
| **Propensity Score (IPW)** | ⭐⭐⭐ | Medium | Medium | Medium | Low |
| **Regression Adjustment** | ⭐⭐⭐ | Medium | Small-Medium | Low | Low |
| **Instrumental Variables** | ⭐⭐⭐ | Medium | Large | High | Low |
| **Difference-in-Differences** | ⭐⭐⭐ | Medium | Medium | Medium | Low |
| **Matching Only** | ⭐⭐ | High | Medium | Low | Low |
| **Naive Comparison** | ⭐ | Very High | Any | Very Low | Low |

---

## 2. Decision Flowchart

```
START: Need to estimate FRM intervention impact
│
├─ Q1: Can you RANDOMIZE treatment assignment?
│  │
│  ├─ YES → Q2: Can patients/FRMs tolerate control group?
│  │  │
│  │  ├─ YES → Q3: Sample size adequate? (See Section 4)
│  │  │  │
│  │  │  ├─ YES (N > 500) → ✅ PATIENT-LEVEL RCT
│  │  │  │                    Best option!
│  │  │  │
│  │  │  └─ NO (N < 500) → Q4: Can you cluster by site/hub?
│  │  │     │
│  │  │     ├─ YES (≥8 clusters) → ✅ CLUSTER RCT or STEPPED WEDGE
│  │  │     │                       Everyone gets treatment eventually
│  │  │     │
│  │  │     └─ NO → ⚠️ Underpowered
│  │  │                Try: Observational with careful analysis
│  │  │
│  │  └─ NO (Ethical/Political concerns) → ✅ STEPPED WEDGE DESIGN
│  │                                        Everyone treated eventually
│  │
│  └─ NO → Q5: Why can't you randomize?
│     │
│     ├─ Reason: "Ethical concerns"
│     │  → Consider: Randomize within MEDIUM-RISK band only
│     │             (Clinical equipoise strongest there)
│     │
│     ├─ Reason: "FRMs will resist"
│     │  → Consider: STEPPED WEDGE or TIME-BASED randomization
│     │             (Easier to operationalize)
│     │
│     ├─ Reason: "Client won't allow"
│     │  → Continue to Q6 (Observational methods)
│     │
│     └─ Reason: "Too expensive/complex"
│        → Continue to Q6 (Observational methods)
│
├─ Q6: Do you have HISTORICAL data with treatment variation?
│  │
│  ├─ YES → Q7: Sample size?
│  │  │
│  │  ├─ LARGE (N > 2000) → Q8: Overlap in propensity scores?
│  │  │  │
│  │  │  ├─ GOOD (All types get both treatments) 
│  │  │  │  → ✅ DOUBLY ROBUST (AIPW)
│  │  │  │     Best observational method
│  │  │  │
│  │  │  └─ POOR (Some types only get one treatment)
│  │  │     → ✅ TRIM SAMPLE + AIPW
│  │  │        OR MATCHING
│  │  │
│  │  ├─ MEDIUM (500-2000) → ✅ PROPENSITY SCORE WEIGHTING (IPW)
│  │  │                        Simpler, adequate power
│  │  │
│  │  └─ SMALL (N < 500) → Q9: Repeated measures over time?
│  │     │
│  │     ├─ YES → ✅ DIFFERENCE-IN-DIFFERENCES
│  │     │         OR INTERRUPTED TIME SERIES
│  │     │
│  │     └─ NO → ⚠️ HIGH UNCERTAINTY
│  │               Use: REGRESSION + SENSITIVITY ANALYSIS
│  │                    Report wide confidence intervals
│  │
│  └─ NO (No historical data) → ⚠️ CANNOT ESTIMATE CAUSALLY
│                                 Options:
│                                 1. Collect data prospectively
│                                 2. Find external comparison group
│                                 3. Wait for more data
│
├─ Q10: Do you have a NATURAL EXPERIMENT or INSTRUMENT?
│  │
│  ├─ YES (e.g., policy change, capacity constraint, etc.)
│  │  → ✅ INSTRUMENTAL VARIABLES
│  │     OR REGRESSION DISCONTINUITY
│  │     OR DIFFERENCE-IN-DIFFERENCES
│  │
│  └─ NO → Return to Q6
│
└─ Q11: Under time/resource pressure?
   │
   ├─ YES, need quick answer → ✅ PROPENSITY SCORE MATCHING
   │                             Fast, interpretable
   │                             BUT: Document limitations!
   │
   └─ NO → ✅ Use DOUBLY ROBUST
               Most rigorous observational method
```

---

## 3. Detailed Method Selection Criteria

### 3.1 Randomized Controlled Trial (RCT)

**When to Use:**
- ✅ You can enforce random assignment
- ✅ Sufficient sample size (typically N > 500 total, N > 250 per arm)
- ✅ Stakeholders accept control group concept
- ✅ Timeline allows for prospective data collection (3-6 months)
- ✅ Ethical to withhold treatment from control (clinical equipoise)

**When NOT to Use:**
- ❌ Sample size too small (rare disease)
- ❌ Can't enforce randomization operationally
- ❌ Unethical to withhold treatment
- ❌ Client/political resistance
- ❌ Need immediate results (can't wait for new data)

**Specialty Pharma Considerations:**
- **Sample size challenge:** Rare diseases may have <500 patients total
  - *Solution:* Cluster at site/hub level or use stepped wedge
- **FRM resistance:** FRMs want autonomy to help patients they think need it
  - *Solution:* Randomize within medium-risk band only (where equipoise is clearest)
- **Client concerns:** "Don't experiment on our patients"
  - *Solution:* Frame as "validating our approach" not "withholding help"

**Sample Size Formula:**
```
n = 2 * [(Z_α/2 + Z_β)² * p*(1-p)] / δ²

Where:
- p = expected baseline success rate (e.g., 0.60)
- δ = minimum detectable effect (e.g., 0.10 = 10pp)
- Z_α/2 = 1.96 (for 95% confidence)
- Z_β = 0.84 (for 80% power)

Example:
- Baseline: 60% success
- Want to detect: 10pp improvement
- Result: n ≈ 385 per arm = 770 total
```

**Implementation Checklist:**
- [ ] Calculate required sample size for adequate power
- [ ] Design randomization scheme (simple, stratified, or cluster)
- [ ] Get stakeholder buy-in (FRMs, clients, leadership)
- [ ] Set up randomization system (automated, tamper-proof)
- [ ] Train FRMs on protocol adherence
- [ ] Monitor for contamination (control patients getting treatment)
- [ ] Pre-register analysis plan to avoid p-hacking

---

### 3.2 Stepped Wedge Cluster Randomized Trial

**When to Use:**
- ✅ RCT is best but control group is ethically/politically unacceptable
- ✅ Have multiple sites/hubs (≥8 clusters minimum)
- ✅ Can roll out intervention sequentially over time
- ✅ Longer timeline acceptable (6-12 months)

**Design:**
```
Time Period:   1    2    3    4    5    6
           ┌────┬────┬────┬────┬────┬────┐
Site/Hub 1 │ C  │ C  │ T  │ T  │ T  │ T  │
Site/Hub 2 │ C  │ C  │ C  │ T  │ T  │ T  │
Site/Hub 3 │ C  │ C  │ C  │ C  │ T  │ T  │
Site/Hub 4 │ C  │ C  │ C  │ C  │ C  │ T  │
           └────┴────┴────┴────┴────┴────┘
           C = Control, T = Treatment

Analysis: Mixed effects model comparing before vs after at each site
```

**Advantages:**
- Everyone eventually gets treatment (ethical appeal)
- Can compare within-site (before vs after) AND across-site (earlier vs later)
- Controls for secular trends
- Less resistance from FRMs and clients

**Disadvantages:**
- Requires ≥8 clusters for adequate power
- Long timeline (need multiple time periods)
- Complex analysis (mixed effects models, adjust for time trends)
- Can't undo treatment once rolled out

**Sample Size:**
- Requires more patients than standard RCT (cluster correlation reduces power)
- Rule of thumb: Multiply standard RCT sample size by 1.5-2x
- Minimum: 50-100 patients per cluster per time period

**Specialty Pharma Application:**
- Cluster by: Hub, geographic region, or disease site
- Time periods: Monthly or quarterly (depending on outcome measurement lag)
- Duration: 6-12 months to allow all clusters to cross over

---

### 3.3 Interrupted Time Series (ITS)

**When to Use:**
- ✅ Intervention rolled out at specific time point
- ✅ Have historical data before intervention
- ✅ Can collect data after intervention
- ✅ Have a comparison group (another product, site, etc.)

**Design:**
```
Outcome
  │
  │    ╱╲    ╱╲   │ ← Intervention starts
  │   ╱  ╲  ╱  ╲  │    ╱────
  │  ╱    ╲╱    ╲ │   ╱      (New level + trend)
  │ ╱            ╲│  ╱
  │╱              ╲│╱
  └────────────────┼────────────→ Time
   Pre-intervention│Post-intervention
```

**Analysis:**
- Segmented regression: `Y = β0 + β1*Time + β2*Intervention + β3*Time_After_Intervention`
- β2 = immediate level change
- β3 = slope change
- Compare to control group trend

**Advantages:**
- Uses existing data (no need to wait for RCT)
- Can detect both immediate effects and trend changes
- Natural comparison if you have multiple products/sites

**Disadvantages:**
- Requires long pre-intervention period (≥12 time points)
- Vulnerable to confounding from other changes at same time
- Can't isolate patient-level effects

**Specialty Pharma Application:**
- Useful when: New FRM team hired, new tool launched, process change
- Time unit: Weekly or monthly (depending on outcome lag)
- Control: Different product, different geography, or different channel

---

### 3.4 Doubly Robust Methods (AIPW, TMLE)

**When to Use:**
- ✅ Cannot randomize but have good historical data
- ✅ Sample size adequate (N > 1000)
- ✅ Have measured important confounders
- ✅ Want most robust observational estimate

**How It Works:**
1. Fit propensity score model: P(Treatment | Covariates)
2. Fit outcome models: E[Outcome | Treatment, Covariates]
3. Combine both using AIPW formula
4. Consistent if EITHER model is correct (not both!)

**Advantages:**
- **Doubly robust:** Two chances to get it right
- **Efficient:** Uses all data, lower variance than matching
- **Standard errors:** Can compute with bootstrapping
- **Flexible:** Can use machine learning for both models

**Disadvantages:**
- **Complex:** Requires understanding of two estimation stages
- **Assumes unconfoundedness:** No unmeasured confounders
- **Sample size:** Needs sufficient overlap in propensity scores
- **Software:** Requires specialized packages (econml, causalml)

**Sample Size Guidance:**
- Minimum: N > 1000 (prefer > 2000)
- Rule: Need 50+ patients in each treatment × covariate combination
- Check overlap: After trimming, should retain ≥80% of sample

**Implementation:**
```python
from econml.dr import DRLearner

# Fit doubly robust model
dr = DRLearner(model_propensity=LogisticRegression(),
               model_regression=RandomForestRegressor())
dr.fit(Y=outcome, T=treatment, X=covariates)

# Estimate ATE
ate = dr.ate(X=covariates)
ate_se = dr.ate_inference(X=covariates).std_err()
```

---

### 3.5 Propensity Score Methods (IPW, Matching, Stratification)

**When to Use:**
- ✅ Cannot randomize but have observational data
- ✅ Medium sample size (N = 500-2000)
- ✅ Have measured key confounders
- ✅ Want interpretable, standard method

**Three Approaches:**

**A. Inverse Probability Weighting (IPW):**
- Weight each patient by 1/P(treatment they received)
- Creates "pseudo-population" where treatment is independent of covariates
- Most efficient (uses all data)

**B. Matching:**
- For each treated patient, find similar untreated patient (by propensity score)
- Compare outcomes within matched pairs
- Most intuitive, but throws away unmatched data

**C. Stratification:**
- Divide into propensity score bins (quintiles)
- Compare treated vs untreated within each bin
- Average across bins
- Middle ground between IPW and matching

**When to Use Each:**
- **IPW:** Large sample, good overlap → most efficient
- **Matching:** Small-medium sample, want interpretability → easy to explain
- **Stratification:** Diagnostic tool, checking robustness → less common as primary analysis

**Sample Size:**
- IPW: Minimum N = 500, prefer N > 1000
- Matching: Minimum N = 300 (after matching), prefer N > 500
- All: Need adequate overlap (positivity)

**Checking Overlap:**
```
Good overlap:
Treated:    ┌─────────────────────────┐ 0.2 to 0.8
Untreated:  ┌─────────────────────────┐ 0.2 to 0.8
            0.0    0.2    0.4    0.6    0.8    1.0
                 Propensity Score

Poor overlap:
Treated:    ┌──────────┐                   0.6 to 0.9
Untreated:        ┌──────────┐             0.1 to 0.4
            0.0    0.2    0.4    0.6    0.8    1.0
                 Propensity Score
            ↑ Problem: No common support!
```

**Implementation Checklist:**
- [ ] Fit propensity score model (logistic regression)
- [ ] Check balance before adjustment (SMD > 0.1 indicates imbalance)
- [ ] Apply weighting/matching/stratification
- [ ] Check balance after adjustment (target: SMD < 0.1)
- [ ] Estimate treatment effect
- [ ] Bootstrap confidence intervals
- [ ] Sensitivity analysis (how strong would unmeasured confounder need to be?)

---

### 3.6 Regression Adjustment

**When to Use:**
- ✅ Quick, simple analysis needed
- ✅ Small-medium sample (N = 300-1000)
- ✅ Confident in model specification
- ✅ Treated and untreated groups overlap on covariates

**How It Works:**
```
Logistic Regression:
  Outcome ~ Treatment + Risk_Score + Payer + Site + Age + ...
  
Coefficient on Treatment = Adjusted Treatment Effect
```

**Advantages:**
- Simple, widely understood
- Fast to implement
- Works with smaller samples than propensity scores
- Can include interactions

**Disadvantages:**
- **Assumes correct functional form** (linearity, additivity)
- **Extrapolates** if groups don't overlap
- **Single model:** No robustness if misspecified

**When It Works Well:**
- Treated and untreated groups are similar on covariates
- Relationships are approximately linear
- Few interaction effects

**When It Fails:**
- Poor overlap (extrapolation)
- Complex non-linear relationships
- Many interactions

**Sample Size:**
- Minimum: N = 300
- Rule of thumb: ≥10 events per variable
- Example: If 40% success rate and 10 covariates, need N ≥ 250

---

### 3.7 Special Cases

**Instrumental Variables (IV)**

**When to Use:**
- ✅ Have an "instrument": variable that affects treatment but not outcome directly
- ✅ Large sample size (N > 2000, IV requires more power)
- ✅ Strong first-stage (instrument predicts treatment well, F > 10)

**Example Instruments for FRM Interventions:**
- **FRM capacity:** Sites with more FRM resources → more interventions, but doesn't directly affect outcomes except through interventions
- **Distance to hub:** Patients closer to hub → more interventions, but distance doesn't directly affect adherence
- **FRM turnover:** New FRM hired → different intervention rates for short period

**Advantage:** Can estimate causal effect even with unmeasured confounding (if instrument is valid)

**Disadvantage:** Hard to find valid instruments; requires large sample; estimates Local Average Treatment Effect (LATE) not ATE

---

**Difference-in-Differences (DID)**

**When to Use:**
- ✅ Have repeated measures (panel data)
- ✅ Intervention rolled out to some units but not others
- ✅ Parallel trends assumption plausible

**Design:**
```
         Control Group    Treatment Group
Before:      Y₀ᶜ              Y₀ᵗ
After:       Y₁ᶜ              Y₁ᵗ

DID = (Y₁ᵗ - Y₀ᵗ) - (Y₁ᶜ - Y₀ᶜ)
```

**Assumption:** Parallel trends - control and treatment would have followed same trend without intervention

**Specialty Pharma Application:**
- Treatment group: Sites that got new FRM team
- Control group: Sites that didn't (yet)
- Measure: Before and after FRM team hire

**Sample Size:** Minimum 4 time periods, 2 groups, 50+ patients per group-period

---

## 4. Sample Size Quick Reference Table

| Method | Minimum N | Recommended N | N per Variable | Special Considerations |
|--------|-----------|---------------|----------------|----------------------|
| **Patient-Level RCT** | 400 (200/arm) | 800 (400/arm) | — | Use power calculation for specific effect size |
| **Cluster RCT** | 800 (8 clusters × 100) | 1600 (16 clusters × 100) | — | Need ≥8 clusters; ICC typically 0.01-0.05 |
| **Stepped Wedge** | 1000 | 2000+ | — | More than standard RCT; depends on # time periods |
| **Doubly Robust (AIPW)** | 1000 | 2000+ | 50+ | Need overlap in propensity scores |
| **Propensity Score (IPW)** | 500 | 1000+ | 30+ | Trim extreme weights |
| **Regression Adjustment** | 300 | 500+ | 10 events | ≥10 events per variable rule |
| **IV / 2SLS** | 2000 | 5000+ | — | Need strong first stage (F > 10) |
| **Difference-in-Differences** | 400 (2 groups × 4 periods × 50) | 800+ | — | More periods = more power |
| **Matching** | 300 (after matching) | 500+ | — | Expect to lose 20-30% in matching |

**How to Use This Table:**

1. **Count your sample:** How many patients do you have with complete data?
2. **Find applicable methods:** Which methods meet the minimum N?
3. **Apply other criteria:** Can you randomize? Have repeated measures? etc.
4. **Choose highest in hierarchy** that meets all criteria

**Example:**
- You have N = 1,200 patients
- Cannot randomize (client won't allow)
- Have historical data with treatment variation
- Good covariate overlap
→ **Recommended: Doubly Robust (AIPW)** ✓ Meets minimum, high in hierarchy

---

## 5. Timeline and Resource Requirements

| Method | Timeline | Analyst Time | FRM Impact | Client Engagement | Cost |
|--------|----------|-------------|------------|------------------|------|
| **RCT** | 3-6 months | 40-80 hours | HIGH (constrained) | HIGH (approval needed) | $$$ |
| **Stepped Wedge** | 6-12 months | 60-100 hours | MEDIUM (phased) | HIGH (approval needed) | $$$ |
| **ITS** | 1-3 months (if data exists) | 20-40 hours | NONE (retrospective) | LOW | $ |
| **AIPW** | 2-4 weeks | 30-50 hours | NONE (retrospective) | LOW | $ |
| **IPW** | 1-2 weeks | 20-30 hours | NONE (retrospective) | LOW | $ |
| **Regression** | 1 week | 10-20 hours | NONE (retrospective) | LOW | $ |

**Quick vs Rigorous Trade-off:**
- **Need answer in 1 week?** → Regression or IPW (but document limitations)
- **Can wait 1 month?** → AIPW (doubly robust, more defensible)
- **Want gold standard?** → RCT (but 3-6 month wait)

---

## 6. Decision Rules for Common Scenarios

### Scenario 1: New Drug Launch

**Situation:**
- Just onboarded new specialty drug
- ~2,000 patients expected in first year
- FRM team being hired
- Client wants to prove value

**Recommended Approach:**
1. **Pilot RCT** (First 6 months, N = 500-800)
   - Randomize first patients (while building FRM capacity anyway)
   - Get gold-standard estimate for this product
   - Use to validate observational methods

2. **Switch to AIPW** (After pilot, ongoing)
   - Use pilot RCT to check observational estimates
   - Apply AIPW to larger population
   - Report with confidence: "Validated against RCT"

---

### Scenario 2: Rare Disease / Small Population

**Situation:**
- Only 300-500 patients total across all sites
- Can't afford to "waste" control group
- FRMs resist randomization
- Need results soon

**Recommended Approach:**
1. **Stepped Wedge Design**
   - Cluster by site (5-8 sites × 50-100 patients each)
   - Roll out over 6 months
   - Everyone gets treatment eventually (ethical + political win)
   - Compare early vs late sites

2. **Alternative: Propensity Score Matching**
   - If absolutely can't randomize
   - Match on key covariates
   - Report with wide confidence intervals
   - Plan RCT for future

---

### Scenario 3: Established Product, Historical Data

**Situation:**
- Have 3 years of data (N > 5,000)
- FRM intervention patterns varied historically
- Want to improve current estimates
- Cannot do prospective experiment

**Recommended Approach:**
1. **Doubly Robust (AIPW)**
   - Use all historical data
   - Two modeling chances (robust to misspecification)
   - Compare to regression and IPW for sensitivity

2. **Add Sensitivity Analysis**
   - E-values: How strong would unmeasured confounder need to be?
   - Rosenbaum bounds
   - Report range of estimates under different assumptions

---

### Scenario 4: Client Demands Immediate Answer

**Situation:**
- Client call tomorrow
- Need impact estimate today
- Only have observational data
- No time for sophisticated methods

**Recommended Approach:**
1. **Quick Propensity Score Matching** (4-6 hours)
   - Fit propensity model
   - Match treated to untreated
   - Report with large caveats

2. **Immediate Follow-Up Plan**
   - "This is preliminary, based on observational data"
   - "We're conducting more rigorous analysis (AIPW) this week"
   - "Long-term, we recommend pilot RCT"
   - Present updated estimates at next meeting

---

## 7. Common Pitfalls and How to Avoid Them

### Pitfall 1: "We don't have enough sample size for anything"

**What people think:** Need thousands of patients for any rigorous method

**Reality:** 
- Regression adjustment: Works with N = 300
- Propensity matching: Works with N = 400
- You CAN do better than naive comparison

**Solution:**
- Use what you have, but report **wide confidence intervals**
- Combine methods: Regression + sensitivity analysis
- Be transparent about uncertainty

---

### Pitfall 2: "RCT is impossible because [reason], so naive comparison is fine"

**What people think:** If gold standard is infeasible, any method is equally bad

**Reality:**
- Huge difference between naive (biased 30-50%) and AIPW (biased <5%)
- Observational methods can be very good if done right

**Solution:**
- Use decision flowchart to find best feasible method
- Don't settle for naive just because RCT is hard

---

### Pitfall 3: "We randomized, so we're done!"

**What people think:** Randomization guarantees correct answer

**Reality:**
- Need adequate sample size (power)
- Need to prevent contamination
- Need to handle attrition/non-compliance
- Still need correct analysis

**Solution:**
- Pre-register analysis plan
- Monitor for protocol violations
- Use intention-to-treat analysis

---

### Pitfall 4: "Propensity scores fix everything"

**What people think:** PS methods eliminate all bias

**Reality:**
- Only adjusts for **measured** confounders
- Still assumes unconfoundedness
- Can't fix poor overlap (positivity)

**Solution:**
- Check covariate balance (SMD < 0.1)
- Do sensitivity analysis
- Consider doubly robust methods

---

## 8. Checklist for Method Selection

Use this checklist to navigate the decision:

### Step 1: Assess Randomization Feasibility
- [ ] Can you enforce random assignment? (operationally)
- [ ] Is control group ethically acceptable?
- [ ] Will FRMs comply with protocol?
- [ ] Will client allow experiment?
- [ ] Timeline allows for prospective data (3-6 months)?

**If YES to all:** → RCT or Stepped Wedge  
**If NO to any:** → Continue to Step 2

---

### Step 2: Check Sample Size
- [ ] Total N available: ________
- [ ] Treatment group: ________
- [ ] Control group: ________
- [ ] After exclusions: ________

**Compare to Table in Section 4:**
- RCT feasible? (N > 400)
- AIPW feasible? (N > 1000)
- IPW feasible? (N > 500)
- Regression feasible? (N > 300)

---

### Step 3: Assess Data Quality
- [ ] Have historical treatment variation?
- [ ] Measured important confounders (risk, payer, site, etc.)?
- [ ] Overlap in covariate distributions?
- [ ] Repeated measures over time?
- [ ] Have a natural experiment or instrument?

**Based on answers:** → See Section 3 for method details

---

### Step 4: Consider Practical Constraints
- [ ] Timeline: Need answer in ___ days/weeks/months
- [ ] Analyst expertise: Comfortable with advanced methods?
- [ ] Software: Have econml, causalml, or similar?
- [ ] Stakeholder sophistication: Need to explain methodology?
- [ ] Precedent: What have we done for similar products?

---

### Step 5: Make Decision
Based on Steps 1-4, my recommended method is:

**Primary Method:** ____________________

**Justification:**
- Sample size: ___________________
- Can/cannot randomize because: ___________________
- Timeline: ___________________
- Data quality: ___________________

**Backup Methods (for sensitivity):**
1. ___________________
2. ___________________

**Known Limitations:**
- ___________________
- ___________________

**Mitigation Plan:**
- ___________________
- ___________________

---

## 9. Reporting Standards by Method

### For RCTs:
**Must Report:**
- CONSORT diagram (enrollment, allocation, follow-up, analysis)
- Baseline characteristics table (check for balance)
- Intention-to-treat analysis
- Per-protocol analysis (as sensitivity)
- Attrition rates and reasons

---

### For Observational Methods:
**Must Report:**
- Covariate balance before adjustment (SMD)
- Covariate balance after adjustment (SMD < 0.1 goal)
- Propensity score overlap plot
- Multiple methods for robustness (e.g., IPW + AIPW)
- Sensitivity analysis (E-values, bounds)
- Clear statement of assumptions (unconfoundedness, positivity)

---

### For All Methods:
- Confidence intervals (not just p-values)
- Sample size and power calculation
- Missing data handling
- Subgroup analyses (pre-specified)
- Limitations section

---

## 10. Summary Table: Quick Method Selector

| Your Situation | Sample Size | Recommended Method | Alternative | Time to Result |
|----------------|-------------|-------------------|-------------|----------------|
| Can randomize, large N | >800 | Patient-Level RCT | Stepped Wedge | 3-6 months |
| Can randomize, small N | 300-800 | Cluster RCT | Stepped Wedge | 6-12 months |
| Can't randomize, large N | >2000 | Doubly Robust (AIPW) | IPW | 2-4 weeks |
| Can't randomize, medium N | 500-2000 | IPW | Regression + IPW | 1-2 weeks |
| Can't randomize, small N | 300-500 | Regression + Sensitivity | Matching | 1 week |
| Rare disease, can randomize | <500 | Stepped Wedge (cluster) | Wait for more data | 6-12 months |
| Rare disease, can't randomize | <500 | Regression + Wide CIs | External comparison | 1-2 weeks |
| Need immediate answer | Any | Matching (quick) | Regression | 1 day |
| Have panel data | Any | Diff-in-Diff | ITS | 2-3 weeks |
| Have instrument | >2000 | IV / 2SLS | AIPW | 2-3 weeks |

---

## 11. Next Steps

### For Your Current Analysis

1. **Complete the checklist** (Section 8)
2. **Identify your scenario** (Section 6 or Table in Section 10)
3. **Follow method-specific guidance** (Section 3)
4. **Implement chosen method** (use code from selection_bias repository)
5. **Report following standards** (Section 9)

### For Future Analyses

1. **Build capacity:**
   - Train team on AIPW and IPW methods
   - Create templates and code libraries
   - Document lessons learned

2. **Improve data collection:**
   - Log FRM decision rationale
   - Capture all contact attempts
   - Measure outcomes consistently

3. **Plan strategic experiments:**
   - Identify 1-2 products for pilot RCT
   - Engage clients in experimentation discussion
   - Build experimental culture

### Resources

- **Simulation notebook:** `selection_bias/notebooks/01_selection_bias_simulation.ipynb`
- **Code examples:** `selection_bias/src/analysis_naive_vs_adjusted.py`
- **Technical guide:** `selection_bias/docs/02_markdown_explainer_selection_bias.md`
- **Presentation:** `selection_bias/docs/01_presentation_experimentation_selection_bias.md`

---

## Appendix: Definitions

**ATE (Average Treatment Effect):** Expected difference in outcomes if everyone received treatment vs if no one did

**AIPW (Augmented Inverse Probability Weighting):** Doubly robust estimator combining propensity scores and outcome regression

**Cluster RCT:** Randomization at group level (e.g., site, hub) rather than individual level

**Confounding:** Variable that affects both treatment and outcome, creating spurious association

**Doubly Robust:** Method that's consistent if EITHER the propensity model OR outcome model is correct

**ICC (Intraclass Correlation):** Proportion of variance due to clustering; reduces effective sample size

**IPW (Inverse Probability Weighting):** Weighting by inverse of treatment probability to balance groups

**ITS (Interrupted Time Series):** Compares trends before vs after intervention point

**Positivity (Overlap):** Assumption that every type of patient has some chance of both treatments

**SMD (Standardized Mean Difference):** Balance metric; |SMD| < 0.1 typically considered balanced

**Stepped Wedge:** Design where all clusters eventually receive treatment, but at staggered times

**Unconfoundedness:** Assumption that treatment assignment is independent of potential outcomes, conditional on measured covariates

---

**Document Version:** 1.0  
**Last Updated:** November 2025  
**Next Review:** Quarterly or after first major application

**Questions?** Contact Analytics Team via #causal-inference-questions on Slack

