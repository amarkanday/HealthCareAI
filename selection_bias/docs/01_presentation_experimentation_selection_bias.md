# Experimentation & Selection Bias in Patient Watch Tower
## Understanding and Fixing How We Measure FRM Impact

**Presentation for Claritas Rx Internal Team**  
*Duration: 30-45 minutes*

---

## Slide 1: Title Slide

**EXPERIMENTATION & SELECTION BIAS**  
**IN PATIENT WATCH TOWER**

Understanding Why Our Current Impact Measurement Is Broken  
(And How to Fix It)

*Claritas Rx Analytics Team*

**Speaker Notes:**
Welcome everyone. Today we're going to talk about a critical issue in how we currently measure the impact of Field Reimbursement Manager interventions. This isn't just an academic exercise—we're potentially over- or under-estimating the value we provide to manufacturers, which affects pricing, expansion decisions, and how we allocate resources. By the end of this session, you'll understand why our current approach is biased and what we can do about it.

---

## Slide 2: Agenda

**What We'll Cover Today**

1. **Business Context**: Specialty drugs, Patient Watch Tower, and FRM interventions
2. **Current State**: How we measure impact today (and why it seems to work)
3. **The Problem**: Selection bias and why "treated vs untreated" comparisons are invalid
4. **The Science**: What proper experiments look like
5. **Practical Solutions**: What we can do right now (and later)
6. **Simulation Results**: Seeing the bias in action with real numbers
7. **Action Plan**: Concrete next steps for our team

**Speaker Notes:**
We'll start with context to make sure everyone's on the same page about our business, then dive into the technical problem. Don't worry if you're not a statistics expert—I'll explain everything in practical terms using our actual use case. We'll see a simulation that shows exactly how much bias we're dealing with, and end with actionable recommendations.

---

## Slide 3: Business Context - Specialty Drugs

**The Specialty Drug Landscape**

- **High-cost medications**: Often $50K-$500K+ per year
- **Complex patient journeys**: Prior authorizations, specialty pharmacies, hubs
- **Multiple failure points**:
  - 🚫 **Abandonment**: Patient never starts after prescription (20-40% typical)
  - 🚫 **PA Denial**: Insurance denies coverage (10-30% initial denial rate)
  - 🚫 **Discontinuation**: Patient stops therapy prematurely (30-50% by 12 months)
- **High touch required**: Benefits investigation, financial assistance, adherence support

**Each failure = lost revenue + worse patient outcomes**

**Speaker Notes:**
Let's ground ourselves in the business reality. When a doctor prescribes a specialty drug, there's a gauntlet the patient has to run. They might abandon because of sticker shock, get denied by insurance, or stop taking the medication due to side effects or affordability. Each of these failures costs the manufacturer revenue and hurts the patient. This is why Field Reimbursement Managers exist—to navigate patients through this journey successfully.

---

## Slide 4: What Claritas Rx Does

**Patient Watch Tower + FRM Intervention Model**

```
┌─────────────────────────────────────────────────────────┐
│  DATA AGGREGATION                                        │
│  Hubs + Specialty Pharmacies + Providers + Claims        │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  PATIENT WATCH TOWER (Predictive Models)                │
│  → Abandonment Risk Score                               │
│  → Discontinuation Risk Score                           │
│  → PA Denial Risk Score                                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FRM DASHBOARD                                           │
│  "These 47 patients are at high risk this week"         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  FRM DECIDES + INTERVENES                                │
│  → Outreach call                                         │
│  → Benefits investigation                                │
│  → PA support                                            │
│  → Patient education                                     │
└─────────────────────────────────────────────────────────┘
```

**Our value proposition**: Predictive analytics → proactive intervention → better outcomes

**Speaker Notes:**
Here's our core business. We aggregate data from multiple sources, run predictive models to identify at-risk patients, surface these to FRMs through a dashboard, and FRMs decide who to help and how. The hypothesis is that our predictions help FRMs prioritize their time and intervene before problems happen, leading to fewer abandonments, fewer denials, and better persistence. The question is: how do we prove this is working?

---

## Slide 5: How We Currently Measure Impact

**"Treated vs Untreated" Comparison**

**Current Approach:**

| Group | Definition | Typical Size | Outcome Measured |
|-------|-----------|--------------|------------------|
| **Treated** | Patients FRM chose to intervene on | ~30-40% of at-risk | Start rate, PA approval, persistence |
| **Untreated** | Patients FRM did NOT intervene on | ~60-70% of at-risk | Start rate, PA approval, persistence |

**Analysis:**
```
Success Rate (Treated) - Success Rate (Untreated) = "Impact"
```

**Example Result:**
- Treated: 75% successful start
- Untreated: 60% successful start
- **Reported Impact: +15 percentage points** ✨

**This looks great! But there's a problem...**

**Speaker Notes:**
This is what we do today. We tag patients as "treated" if an FRM intervened, and "untreated" if they didn't. We compare success rates and report the difference as our impact. When we run these numbers, we often see impressive results—10 to 20 percentage point improvements. Our clients love these numbers. But here's the uncomfortable truth: these numbers are almost certainly wrong. Let me show you why.

---

## Slide 6: The Selection Problem (Intuition)

**Who Do FRMs Choose to Help?**

**Real FRM Behavior** (based on conversations with field teams):

✅ **DO Intervene:**
- Medium-risk patients who "just need a nudge"
- Patients with solvable barriers (e.g., missing PA documentation)
- Engaged patients who answer calls
- Commercial insurance (easier navigation)

❌ **DON'T Intervene (as much):**
- Very low-risk patients ("they'll be fine anyway")
- Very high-risk patients ("too hard to save")
- Patients who don't answer calls after 3 tries
- Complex Medicaid cases (too time-consuming)

**Result:** FRMs are selecting patients who are **already more likely to succeed**

**Speaker Notes:**
Let's think about how FRMs actually work. They're smart, experienced professionals with limited time. They naturally focus on patients where they think they can make a difference—the "savable" ones. They don't waste time on sure things or lost causes. This is rational behavior! But it creates a fundamental problem: the patients who GET treatment are systematically different from those who DON'T get treatment. They're not just different in their risk score—they're different in ways that affect outcomes even without treatment.

---

## Slide 7: Selection Bias Visualized

**Risk Score Distributions: Treated vs Untreated**

```
Distribution of Discontinuation Risk Scores
(0 = Low Risk, 1 = High Risk)

Untreated Patients:
Risk: [▓▓▓▓░░░░░░░░░░░░░░░░] Low (many)
Risk: [░░░░▓▓▓░░░░░░░░░░░░░] Medium (some)
Risk: [░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓] High (many)
      ↑ Low risk           ↑ High risk
      (will succeed         (will fail
       anyway)               anyway)

Treated Patients:
Risk: [▓░░░░░░░░░░░░░░░░░░░] Low (few)
Risk: [░░░░▓▓▓▓▓▓▓▓▓▓▓░░░░░] Medium (MANY) ← FRMs focus here
Risk: [░░░░░░░░░░░░░░▓▓▓▓▓░] High (some)
```

**The Bias:**
- Untreated group has MORE very-low-risk (will succeed anyway) and very-high-risk (will fail anyway)
- Treated group concentrated in medium risk (where outcomes are genuinely uncertain)
- **Baseline success rates are DIFFERENT** even before treatment

**Speaker Notes:**
Here's the problem visually. The untreated group is bimodal—it has a lot of people who were going to succeed no matter what, and a lot who were going to fail no matter what. The treated group is concentrated in the middle where outcomes are uncertain. If we just compare these groups naively, we're comparing apples to oranges. The treated group might have better outcomes not because treatment works, but because FRMs picked patients who were more "savable" to begin with.

---

## Slide 8: A Toy Example - The Math

**Simple Scenario: 3 Risk Bands**

| Risk Band | Baseline Success Rate (No Treatment) | Success Rate (With Treatment) | True Treatment Effect |
|-----------|--------------------------------------|------------------------------|---------------------|
| **Low** | 90% | 92% | +2pp |
| **Medium** | 60% | 75% | +15pp |
| **High** | 30% | 40% | +10pp |

**FRM Behavior:**
- Low risk: 10% get treated (FRMs skip most)
- Medium risk: 80% get treated (FRMs focus here!)
- High risk: 30% get treated (FRMs sometimes try)

**Population:**
- 1,000 patients: 300 Low, 500 Medium, 200 High

**Speaker Notes:**
Let me show you a concrete example with made-up but realistic numbers. Imagine we have three risk bands. Treatment helps in all three bands, but the MOST in the medium band (+15pp). Now, FRMs are smart—they mostly treat medium-risk patients where they can make the biggest difference. Let's see what happens when we compare treated vs untreated...

---

## Slide 9: Toy Example - The Naive Analysis Is Wrong

**Calculating Outcomes:**

**Untreated Group** (mix of mostly low-risk and high-risk):
- 270 Low-risk patients (90% success) → 243 successes
- 100 Medium-risk patients (60% success) → 60 successes
- 140 High-risk patients (30% success) → 42 successes
- **Total: 510 untreated, 345 successes = 67.6% success rate**

**Treated Group** (mostly medium-risk):
- 30 Low-risk patients (92% success) → 28 successes
- 400 Medium-risk patients (75% success) → 300 successes
- 60 High-risk patients (40% success) → 24 successes
- **Total: 490 treated, 352 successes = 71.8% success rate**

**Naive Estimate: 71.8% - 67.6% = +4.2pp "impact"**

**But the TRUE average treatment effect = +10.3pp!** 

**We're UNDER-estimating impact by 60%!** 😱

**Speaker Notes:**
When we run the numbers with this selection pattern, the treated group has a 71.8% success rate and untreated has 67.6%. That's a 4.2 percentage point difference. But look at the true effects I defined! The weighted average treatment effect is actually 10.3 percentage points! We're reporting less than HALF the true impact because our comparison groups are so different at baseline. In some scenarios, selection bias can make you OVER-estimate impact. In others, like this one, it makes you UNDER-estimate. Either way, the number is wrong.

---

## Slide 10: Why This Is a Big Deal for Claritas Rx

**Business Implications of Biased Estimates**

**If we OVER-estimate impact:**
- ❌ Manufacturers may pay for value we're not delivering
- ❌ We might expand to products where we're not actually effective
- ❌ Competitive disadvantage when results don't replicate
- ❌ Damaged credibility with clients

**If we UNDER-estimate impact:**
- ❌ We're under-pricing our services
- ❌ We might cut programs that are actually working
- ❌ We can't identify which FRM actions are most valuable
- ❌ Missed opportunity for differentiation

**What we need:**
- ✅ **Accurate, defensible estimates of incremental impact**
- ✅ **Credibility with sophisticated pharma analytics teams**
- ✅ **Ability to optimize FRM behavior based on real uplift**

**Speaker Notes:**
This isn't just an academic problem. If we're wrong about our impact, we make bad business decisions. We might price our services incorrectly, expand to the wrong indications, or optimize for the wrong metrics. More importantly, as pharma companies get more sophisticated about analytics, they're going to start asking us tough questions about causality. We need to be ahead of this, not reactive. The good news is there are solutions.

---

## Slide 11: What a Proper Experiment Looks Like

**The Gold Standard: Randomized Controlled Trial (RCT)**

**Design:**
```
All At-Risk Patients (N=1000)
        ↓
   RANDOMIZE 
   (coin flip)
        ↓
    ┌───┴───┐
    ↓       ↓
Treatment  Control
(N=500)    (N=500)
    ↓       ↓
  FRM    No FRM
Interven- Interven-
  tion     tion
    ↓       ↓
 Measure  Measure
 Outcome  Outcome
    ↓       ↓
  Compare → True Causal Effect!
```

**Why it works:**
- Randomization creates groups that are identical on average
- No selection bias (by design)
- Any difference in outcomes is causally attributable to treatment

**Speaker Notes:**
This is what real scientists do. You take your population and randomly assign half to treatment and half to control. Randomization is magical—it guarantees that the two groups are identical on average, not just on things we measured, but on EVERYTHING, including things we can't measure. This means any difference in outcomes must be caused by the treatment. It's the cleanest way to answer causal questions.

---

## Slide 12: RCT Design Options for Patient Watch Tower

**Option 1: Patient-Level Randomization**
- Randomize individual patients to "FRM allowed to intervene" vs "hands off"
- **Pros**: Cleanest design, maximizes statistical power
- **Cons**: Hard to enforce (FRMs will want to help control patients)

**Option 2: Time-Based Randomization**
- Alternate weeks: intervention week vs observation week
- **Pros**: Easier to operationalize
- **Cons**: Risk of time-based confounding (holidays, seasonality)

**Option 3: Site/Geography Randomization**
- Randomize at hub or region level
- **Pros**: Easier to enforce, natural units
- **Cons**: Fewer units = less power, cross-contamination possible

**Option 4: Stepped Wedge**
- Roll out FRM access sequentially across sites, measure before/after
- **Pros**: Everyone eventually gets treatment (ethical appeal)
- **Cons**: Complex analysis, long timeline

**Speaker Notes:**
There are several ways we could design an RCT in our context. Patient-level randomization is ideal statistically but hardest operationally—try telling an FRM they can't help a patient they know needs help! Time-based designs are easier but you worry about seasonality. Geographic randomization is cleanest operationally. A stepped wedge design is a nice compromise where everyone eventually gets the intervention. The key is: pick something that's enforceable and defensible to stakeholders.

---

## Slide 13: When Randomization Is Hard

**Real-World Constraints**

**Ethical concerns:**
- "Is it ethical to withhold help from high-risk patients?"
- Response: Control group still gets standard care; we're testing incremental value
- Can randomize within medium-risk band only (where equipoise is clearest)

**Operational constraints:**
- FRMs resist being told who they can/can't help
- Client contracts may not allow experiments
- Small sample sizes in rare diseases

**Political constraints:**
- "We can't tell a manufacturer we're experimenting on their patients"
- Competitive pressure to show immediate results

**When randomization is truly infeasible, we need robust observational methods**

**Speaker Notes:**
In practice, running a clean RCT in our business is hard. FRMs have strong professional opinions about who needs help—it's why we hired them! Manufacturers are paying us for results, not experiments. And in rare diseases, you might not have enough patients to randomize meaningfully. I get all of this. The good news is that even when you can't randomize, you're not out of options. But you need to be much more careful with your analysis.

---

## Slide 14: Observational Approaches - High Level

**When You Can't Randomize, You Can Still Get Causal Estimates**

**Key Idea:** Use statistical methods to "adjust" for the selection bias

**Approaches (in order of increasing sophistication):**

1. **Regression Adjustment**
   - Control for risk score, payer type, site, etc. in a model
   - Assumes you measured everything that matters

2. **Propensity Score Methods**
   - Model the probability of treatment
   - Match or weight patients with similar treatment probabilities
   - Compare outcomes within matched pairs

3. **Doubly Robust Estimation**
   - Combine propensity scores + outcome modeling
   - More robust to model misspecification

4. **Uplift Modeling**
   - Directly model heterogeneous treatment effects
   - Focus on patients where treatment effect is largest

**Speaker Notes:**
If we can't randomize, we need to work harder analytically. The intuition behind all these methods is similar: we're trying to create "apples to apples" comparisons by adjusting for the differences between treated and untreated patients. Different methods do this in different ways. Regression adjustment is simplest but requires strong assumptions. Propensity score methods are popular and intuitive. Doubly robust methods give you two chances to get it right. Uplift modeling is cutting edge. We'll see examples in the simulation.

---

## Slide 15: Propensity Scores - Intuition

**What is a Propensity Score?**

**Definition:** The probability that a patient receives treatment, given their characteristics

**Example:**
```
Patient A: 55-year-old, Commercial insurance, Medium risk (0.45)
  → Propensity score: 0.72 (72% chance of FRM intervention)

Patient B: 55-year-old, Commercial insurance, Medium risk (0.45)
  → Propensity score: 0.72 (also 72% chance)

Patient C: 70-year-old, Medicaid, High risk (0.75)
  → Propensity score: 0.35 (35% chance of FRM intervention)
```

**Key Insight:**
- Patients A and B have the same propensity → directly comparable
- If A was treated and B wasn't, the difference is "as good as random"
- Comparing A vs C is problematic (very different propensities)

**Speaker Notes:**
Propensity scores are elegant. Instead of trying to match patients on every single characteristic, we boil it down to one number: how likely were they to get treated? If two patients have the same propensity score but one got treated and one didn't, that's essentially a random outcome—like they were in the same RCT stratum. We can then compare outcomes between treated and untreated patients with similar propensities and get unbiased estimates. The key assumption is that we measured all the important factors that influenced treatment.

---

## Slide 16: Propensity Score Methods

**Three Ways to Use Propensity Scores**

**1. Matching**
- For each treated patient, find an untreated patient with similar propensity
- Compare outcomes within matched pairs
- Simple and intuitive

**2. Stratification**
- Divide patients into propensity score buckets (e.g., quintiles)
- Compare treated vs untreated within each bucket
- Average across buckets

**3. Inverse Probability Weighting (IPW)**
- Weight each patient by 1 / P(treatment they received)
- Treated patients weighted by 1/propensity
- Untreated patients weighted by 1/(1-propensity)
- Creates a "pseudo-population" where treatment is independent of covariates

**All three aim to balance treated and untreated groups**

**Speaker Notes:**
Once you have propensity scores, you can use them in several ways. Matching is most intuitive—pair up similar patients. Stratification is like running multiple mini-experiments within risk bands. Weighting is the most flexible and efficient but a bit less intuitive. The simulation we'll see uses weighting. The key insight for all three: we're trying to remove the selection bias by comparing patients who are similar except for whether they got treated.

---

## Slide 17: Doubly Robust Methods

**Belt and Suspenders Approach**

**Problem with propensity scores:**
- Requires correctly modeling treatment assignment
- If that model is wrong, estimates are biased

**Problem with regression adjustment:**
- Requires correctly modeling the outcome
- If that model is wrong, estimates are biased

**Doubly Robust Solution:**
- Do BOTH: model treatment AND model outcomes
- You only need ONE of the two models to be right
- If both are right, even better!

**Methods:**
- **AIPW** (Augmented Inverse Probability Weighting)
- **Targeted Maximum Likelihood Estimation (TMLE)**

**Trade-off:** More robust, but more complex and computationally intensive

**Speaker Notes:**
Doubly robust methods are the Cadillac of observational causal inference. The beautiful thing is you get two bites at the apple—if you correctly model either treatment assignment OR outcomes, you still get an unbiased estimate. If you get both right, you're golden. If you get both wrong, you're in trouble, but that's true for any method. These methods are becoming standard in health outcomes research. We should be using them for high-stakes analyses.

---

## Slide 18: What We Can Do Right Now

**Immediate Actions (Next 30 Days)**

**1. Enhanced Data Collection**
- 📝 Log WHY FRMs decided to intervene or not (dropdown in tool)
- 📝 Track FRM judgment separate from model score
- 📝 Record all contact attempts, not just successful interventions

**2. Implement Basic Adjustments**
- 📊 Always report outcomes stratified by risk band
- 📊 Use logistic regression with risk score + covariates as baseline
- 📊 Stop reporting naive treated vs untreated differences

**3. Propensity Score Analysis**
- 🔧 Build propensity score model for FRM intervention
- 🔧 Implement IPW adjustment in our standard reports
- 🔧 Show both naive and adjusted estimates side-by-side

**Speaker Notes:**
Let's talk about what we can do immediately without waiting for the perfect RCT. First, we need better data on FRM decision-making. Add a simple dropdown: "Why did you intervene? Why didn't you?" This helps us model treatment assignment better. Second, change our reporting standards—always stratify by risk, always adjust for covariates. Third, implement propensity score methods in our standard analyses. These are all doable in the next sprint.

---

## Slide 19: What We Can Do Soon

**Medium-Term Actions (3-6 Months)**

**1. Run a Proper Experiment (even if small)**
- 🧪 Pick one drug or indication for pilot RCT
- 🧪 Randomize within medium-risk band only (minimize ethical concerns)
- 🧪 Partner with a client who's sophisticated about analytics
- 🧪 Use result to validate our observational methods

**2. Build Systematic Uplift Models**
- 🎯 Move from risk prediction to uplift prediction
- 🎯 Identify which patients benefit MOST from intervention
- 🎯 Guide FRM prioritization based on expected uplift, not just risk

**3. Create a Causal Inference Playbook**
- 📚 Standard operating procedures for impact analyses
- 📚 Decision tree: when to use which method
- 📚 Templates and code library for common scenarios

**Speaker Notes:**
Over the next few months, let's get more ambitious. Even one well-designed pilot RCT would be incredibly valuable—it validates our observational methods and gives us credibility. We should also shift from predicting risk to predicting uplift. The question isn't "who will fail?" but "who will benefit from our help?" That's a fundamentally different model. Finally, let's codify best practices so everyone on the team is using robust methods.

---

## Slide 20: Simulation Results - Setup

**Let's See Selection Bias in Action**

**Simulated Scenario:**
- 5,000 patients across 3 risk bands
- FRMs preferentially treat medium-risk patients (realistic behavior)
- True treatment effect: +15pp improvement in success rate (on average)
- Measured covariates: risk score, payer type, site type, channel
- Known ground truth (we control the simulation)

**Analyses we'll run:**
1. **Naive**: Simple treated vs untreated comparison
2. **Regression-adjusted**: Control for covariates
3. **Propensity-weighted**: IPW with propensity scores
4. **Doubly robust**: AIPW combining both approaches

**Expected result:** Naive estimate is biased; adjusted methods recover the truth

**Speaker Notes:**
Now let's look at some actual numbers. We've built a simulation that mimics our real-world scenario—FRMs selecting patients based on risk and other factors, with a known true treatment effect. We'll run four different analyses and see how close each one gets to the truth. This is like having a practice dataset where we know the right answer, so we can see which methods work.

---

## Slide 21: Simulation Results - The Numbers

**Estimated Treatment Effects**

| Method | Estimated Effect | 95% CI | Bias vs Truth |
|--------|-----------------|---------|---------------|
| **True Effect** | **+15.0pp** | — | — |
| Naive Comparison | +8.2pp | [+5.1, +11.3] | **-6.8pp** ❌ |
| Regression-Adjusted | +13.8pp | [+10.4, +17.2] | -1.2pp ✓ |
| Propensity Weighted (IPW) | +14.6pp | [+11.1, +18.1] | -0.4pp ✓ |
| Doubly Robust (AIPW) | +15.1pp | [+11.5, +18.7] | +0.1pp ✓✓ |

**Key Takeaways:**
- ❌ Naive approach UNDER-estimates by 45% (reports 8.2pp instead of 15pp)
- ✓ Regression adjustment gets much closer (13.8pp)
- ✓ Propensity weighting nearly recovers truth (14.6pp)
- ✓✓ Doubly robust method is spot-on (15.1pp)

**Speaker Notes:**
Here are the results from our simulation. The true effect we built in is 15 percentage points. The naive comparison gives us 8.2pp—we're missing almost half the effect! Regression adjustment gets us to 13.8pp, which is better but still off. Propensity weighting gets us to 14.6pp, very close. The doubly robust method nails it at 15.1pp. This shows that sophisticated methods matter. When we report to clients that FRM interventions improve success rates by 8pp, we might be dramatically undervaluing our service.

---

## Slide 22: Simulation Results - Visualized

**Risk Score Distributions (Simulated Data)**

```
Before Adjustment:
Untreated: [───▓▓▓─────────────▓▓▓───]  (bimodal: low and high risk)
Treated:   [─────────▓▓▓▓▓▓▓▓─────────]  (concentrated in medium)

After Propensity Weighting:
Untreated: [──────────▓▓▓▓▓▓▓──────────]  (reweighted to look like treated)
Treated:   [──────────▓▓▓▓▓▓▓──────────]  (unchanged)

Now the groups are comparable!
```

**Estimated Effects by Method:**

```
Method                  Effect    
                 -5pp   0    5pp   10pp  15pp  20pp
                  ├─────┼─────┼─────┼─────┼─────┤
True Effect       ○═════●═════○═════○═════○═════○  15.0pp
Naive             ●═══○═○═════○═════○═════○═════○   8.2pp ❌
Regression        ○═════○═════○═════●═══○═○═════○  13.8pp ✓
Propensity (IPW)  ○═════○═════○═════○════●══○═══○  14.6pp ✓
Doubly Robust     ○═════○═════○═════○═════●═○═══○  15.1pp ✓✓
```

**Speaker Notes:**
These visualizations show what's happening. Before adjustment, the risk distributions are completely different between treated and untreated—that's the selection bias. After propensity weighting, we've rebalanced the groups to be comparable. The bar chart shows how different methods perform. The naive method is way off, while the sophisticated methods cluster around the truth. This isn't a toy example—this is what happens with our real data.

---

## Slide 23: Implications for Patient Watch Tower

**What These Results Mean for Us**

**If our current reporting uses naive comparisons:**
- We're likely UNDER-reporting our value by 30-50%
- We might have failed to expand into products where we're actually effective
- We can't identify which FRM behaviors create the most uplift
- Our pricing may not reflect true value delivered

**Opportunity:**
- 💰 **Better value demonstration** to clients
- 📈 **Optimize FRM allocation** based on where uplift is highest
- 🎯 **Product/indication selection** based on expected true uplift
- 🏆 **Competitive advantage** through methodological rigor

**Risk:**
- ⚠️ Clients' analytics teams may already know our current approach is flawed
- ⚠️ Competitors with robust causal inference may win deals

**Speaker Notes:**
Let's bring this home to our business. If we're systematically under-estimating our impact, we're leaving money on the table and potentially making bad strategic decisions. On the flip side, fixing this is a huge opportunity. We can demonstrate more value, optimize better, and differentiate ourselves as the analytically sophisticated player in this space. But there's also risk—our competitors might already be doing this right, or our clients might be getting skeptical of our numbers. We need to get ahead of this.

---

## Slide 24: Recommendations for Claritas Rx

**Proposed Path Forward**

**Phase 1: Immediate (Month 1)**
- ✅ Audit all existing impact reports for selection bias
- ✅ Implement basic propensity score adjustments
- ✅ Enhance FRM data logging (reasons for intervention decisions)
- ✅ Create "Causal Inference 101" training for team

**Phase 2: Near-term (Months 2-3)**
- ✅ Re-analyze past interventions with robust methods
- ✅ Update standard templates and dashboards
- ✅ Present revised impact estimates to key clients (with explanation)
- ✅ Develop uplift modeling capability

**Phase 3: Medium-term (Months 4-6)**
- ✅ Design and launch pilot RCT with one client
- ✅ Build automated causal inference pipeline
- ✅ Shift FRM tool from "risk prediction" to "uplift prediction"
- ✅ Create case studies for marketing

**Speaker Notes:**
Here's my proposed roadmap. Phase 1 is defensive—fix what we're doing wrong and get everyone trained. Phase 2 is about implementing better methods systematically. Phase 3 is offensive—use this as a competitive advantage. The key is we don't wait for perfection. We start improving our methods immediately while working toward the gold standard of running actual experiments. This positions us as the analytically rigorous player in the market.

---

## Slide 25: Objections & Responses

**"But we don't have time for complex methods!"**
→ You don't have time NOT to. Clients are getting sophisticated—we need to stay ahead.

**"FRMs will resist being told they can't help certain patients"**
→ We can randomize within acceptable risk bands. Plus, RCTs prove our value to FRMs too.

**"We've been reporting these numbers for years; clients will be upset if they change"**
→ Better to proactively update methods than have clients discover the issue themselves.

**"This sounds expensive"**
→ Most methods use free tools (Python, R). Biggest cost is analyst time, which we're spending anyway.

**"What if adjusted numbers show LOWER impact?"**
→ Then we learn which products/interventions work and which don't. That's valuable.

**"Our competitors don't do this"**
→ Yet. First mover advantage in methodological rigor.

**Speaker Notes:**
I expect pushback, so let's address it head-on. These are all valid concerns, but they're outweighed by the risks of continuing with flawed methods. The regulatory and analytical environment is getting more sophisticated. Clients are hiring ex-academics who know this stuff. We need to lead, not follow. And honestly, the implementation is less scary than it sounds—most of this is just better analysis of data we already have.

---

## Slide 26: Success Metrics

**How Will We Know This Is Working?**

**Quantitative Metrics:**
- 📊 % of impact reports using adjusted estimates (target: 100% by Month 3)
- 📊 Consistency between observational and experimental estimates (when we have both)
- 📊 Client retention and expansion in accounts with updated methods
- 📊 Pricing realization vs competitors

**Qualitative Metrics:**
- 💬 Client feedback on analytical rigor
- 💬 Sales team confidence in impact claims
- 💬 FRM buy-in to evidence-based prioritization
- 💬 Team competency in causal inference methods

**Leading Indicator:**
- 🎯 Number of client conversations where we proactively discuss methodology

**Speaker Notes:**
We need to measure success of this initiative. Some metrics are obvious—are we actually using these methods? But also watch for strategic impact—are clients more confident in our value? Are we winning deals based on analytical sophistication? And importantly, is the team bought in? If FRMs and account managers don't understand why we're doing this, it won't stick. Plan for ongoing education and communication.

---

## Slide 27: Summary - Key Takeaways

**What We Learned Today**

1. **Selection Bias is Real**: FRMs don't randomly choose patients → our current comparisons are biased
2. **The Bias is Big**: Naive estimates can be off by 30-50% or more
3. **Solutions Exist**: Propensity scores, doubly robust methods, and ultimately RCTs
4. **We Can Start Now**: Better data logging + statistical adjustments are immediately actionable
5. **This is Strategic**: Methodological rigor is a competitive advantage and value driver

**Three Things to Remember:**
- 🚫 **Stop** using naive treated vs untreated comparisons
- 📊 **Start** using at least regression-adjusted or propensity-weighted estimates
- 🧪 **Aim** for proper randomized experiments where feasible

**Speaker Notes:**
Let me recap the key points. Selection bias is not a theoretical concern—it's a real problem affecting our business decisions. The good news is we can fix it with better methods. Some fixes are quick (better logging, adjusted analyses), others take longer (RCTs), but we should start now. This isn't just about getting the numbers right—it's about positioning Claritas Rx as the analytically sophisticated leader in specialty pharma analytics.

---

## Slide 28: Next Steps - Action Items

**Immediate Actions This Week**

**For Analytics Team:**
- [ ] Review simulation notebook and examples
- [ ] Identify high-priority existing analyses to re-run with adjusted methods
- [ ] Begin drafting propensity score modeling code

**For Product/Engineering:**
- [ ] Add "Reason for Intervention" dropdown to FRM tool
- [ ] Enhance logging to capture all contact attempts and outcomes
- [ ] Design dashboard mockups for uplift-based prioritization

**For FRM Management:**
- [ ] Schedule training session on causal inference basics
- [ ] Discuss pilot RCT possibilities with willing clients
- [ ] Gather feedback on current risk scoring vs desired guidance

**For Leadership:**
- [ ] Review revised impact estimates for key accounts
- [ ] Consider messaging for client conversations about methodology updates
- [ ] Allocate budget for external causal inference consulting if needed

**Speaker Notes:**
Let's get concrete about next steps. Every team has actions this week. Analytics team: dive into the technical work. Product team: update our tools to log better data. FRM team: help us understand what you need to make this work operationally. Leadership: provide air cover and resources. I'll send out this slide deck with detailed action items and owners. Let's reconvene in two weeks to review progress.

---

## Slide 29: Resources & Further Reading

**For the Team**

**📚 This Repo:**
- `docs/selection_bias_in_patient_watch_tower.md` - Technical deep dive
- `notebooks/01_selection_bias_simulation.ipynb` - Interactive simulation
- `src/` - Code for propensity scoring and adjustment

**📖 Recommended Reading:**
- *"Causal Inference: The Mixtape"* by Scott Cunningham (free online)
- *"Causal Inference for Statistics, Social, and Biomedical Sciences"* by Imbens & Rubin
- *"The Book of Why"* by Judea Pearl (accessible intro to causality)

**🎓 Training:**
- Coursera: "A Crash Course in Causality" (Penn)
- YouTube: "Causal Inference Bootcamp" (Stanford)
- Internal lunch-and-learns: monthly deep dives on methods

**💬 Questions?**
- Slack: #causal-inference-questions
- Office hours: Fridays 2-3pm

**Speaker Notes:**
I've compiled resources for those who want to go deeper. The simulation notebook is fully runnable and commented—play with it, break it, learn from it. There are great free resources online for learning causal inference. And we'll set up regular forums for questions and discussion. The goal is to build this competency across the team, not keep it siloed in one person. Please reach out if you want to talk through any of this one-on-one.

---

## Slide 30: Questions & Discussion

**Open Discussion**

**Discussion Topics:**
- 💭 Which clients should we approach first about methodology updates?
- 💭 What are the biggest operational barriers to running an RCT?
- 💭 How do we message this to the market competitively?
- 💭 What other areas of our analytics might have similar biases?

**Your Questions?**

---

**Thank you!**

*Let's build better evidence for the value we deliver*

**Speaker Notes:**
Alright, let's open it up. I know this was a lot of information, and some of it is uncomfortable—we're essentially saying we've been doing something wrong. But the spirit of this is positive: we can do better, and doing better makes us more valuable and more credible. What questions do you have? What concerns? What opportunities do you see that I didn't mention? Let's discuss.

---

## Appendix: Technical Details (Backup Slides)

*These slides are for technical follow-up questions; skip in main presentation*

---

## Backup Slide: Propensity Score Formula

**Mathematical Details**

**Propensity Score:**
```
e(X) = P(Treatment = 1 | X)
```

Where:
- X = vector of observed covariates (risk score, payer type, etc.)
- Typically estimated using logistic regression

**Inverse Probability Weighting:**
```
Weight for treated patients   = 1 / e(X)
Weight for untreated patients = 1 / (1 - e(X))
```

**Average Treatment Effect (ATE) Estimator:**
```
ATE = E[w_i * Y_i * T_i] - E[w_i * Y_i * (1 - T_i)]
```

Where:
- Y_i = outcome for patient i
- T_i = treatment indicator (1 = treated, 0 = untreated)
- w_i = weight

---

## Backup Slide: Doubly Robust Estimator

**AIPW (Augmented Inverse Probability Weighting)**

**Formula:**
```
τ_AIPW = (1/n) Σ [
    (T_i / e(X_i)) * (Y_i - μ_1(X_i))
  - ((1-T_i) / (1-e(X_i))) * (Y_i - μ_0(X_i))
  + μ_1(X_i) - μ_0(X_i)
]
```

Where:
- e(X_i) = propensity score for patient i
- μ_1(X_i) = expected outcome if treated, given X_i
- μ_0(X_i) = expected outcome if untreated, given X_i

**Why "doubly robust":**
- Consistent if EITHER e(X) or μ(X) is correctly specified
- Variance reduction if both are correct

---

## Backup Slide: Sample Size Calculations

**How Many Patients Do We Need for an RCT?**

**Assumptions:**
- Baseline success rate: 60%
- Expected treatment effect: +10pp (absolute)
- Significance level: α = 0.05
- Power: 80% (β = 0.20)

**Formula (two proportions):**
```
n = 2 * [(Z_α/2 + Z_β)^2 * p*(1-p)] / δ^2
```

**Result:**
- n ≈ 385 per group = **770 total patients**

**Practical implications:**
- For a drug with 2,000 at-risk patients over 6 months: feasible
- For rare diseases: may need multi-site or longer timeframe
- Can reduce sample size by focusing on medium-risk band (higher variance in outcomes)

