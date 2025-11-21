"""
Data Simulation for Selection Bias Demonstration
Claritas Rx - Patient Watch Tower / FRM Intervention Analysis

This module generates synthetic patient data that mimics the real-world scenario where:
1. Field Reimbursement Managers (FRMs) select which patients to intervene on
2. Selection is based on risk scores, payer type, and other factors (not random)
3. Treatment has a known, true causal effect that we can compare estimates against

The simulation allows us to demonstrate selection bias and evaluate correction methods.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Default simulation size
DEFAULT_N_PATIENTS = 5000

# Risk band thresholds
RISK_THRESHOLDS = {
    'low': (0.0, 0.33),
    'medium': (0.33, 0.67),
    'high': (0.67, 1.0)
}

# True treatment effects by risk band
# These are the KNOWN causal effects we build into the simulation
TRUE_TREATMENT_EFFECTS = {
    'low': 0.02,      # +2pp improvement (baseline ~90%)
    'medium': 0.15,   # +15pp improvement (baseline ~60%)
    'high': 0.10      # +10pp improvement (baseline ~30%)
}

# FRM intervention probabilities by risk band
# This mimics realistic selection behavior: focus on "savable" medium-risk patients
FRM_INTERVENTION_PROBS = {
    'low': 0.10,      # FRMs rarely intervene on low-risk (will succeed anyway)
    'medium': 0.80,   # FRMs heavily target medium-risk (biggest opportunity)
    'high': 0.30      # FRMs sometimes try high-risk (but often seen as lost causes)
}

# Payer type distributions
PAYER_TYPES = ['Commercial', 'Medicare', 'Medicaid', 'PatientPay']
PAYER_DISTRIBUTION = [0.60, 0.25, 0.10, 0.05]

# Site type distributions
SITE_TYPES = ['Academic', 'Community', 'Specialty']
SITE_DISTRIBUTION = [0.30, 0.50, 0.20]

# Channel distributions
CHANNELS = ['Hub', 'NonHub']
CHANNEL_DISTRIBUTION = [0.70, 0.30]


# ============================================================================
# CORE SIMULATION FUNCTIONS
# ============================================================================

def simulate_patients(n_patients: int = DEFAULT_N_PATIENTS, 
                     random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic patient population with characteristics.
    
    This creates the "universe" of at-risk patients before any treatment assignment.
    
    Parameters
    ----------
    n_patients : int
        Number of patients to simulate
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        Patient data with columns:
        - patient_id: Unique identifier
        - risk_score: Continuous risk score (0-1)
        - risk_band: Categorical (Low/Medium/High)
        - payer_type: Commercial, Medicare, Medicaid, PatientPay
        - site_type: Academic, Community, Specialty
        - channel: Hub vs NonHub referral
        - age: Patient age
        - days_since_script: Days since prescription written
    """
    np.random.seed(random_state)
    
    # Generate patient IDs
    patient_ids = [f"PAT{i:06d}" for i in range(1, n_patients + 1)]
    
    # Generate risk scores from a mixture distribution
    # Real risk scores tend to be U-shaped or multimodal
    risk_scores = generate_risk_scores(n_patients)
    
    # Assign risk bands based on thresholds
    risk_bands = pd.cut(risk_scores, 
                        bins=[0, 0.33, 0.67, 1.0], 
                        labels=['Low', 'Medium', 'High'],
                        include_lowest=True)
    
    # Generate covariates
    payer_types = np.random.choice(PAYER_TYPES, size=n_patients, p=PAYER_DISTRIBUTION)
    site_types = np.random.choice(SITE_TYPES, size=n_patients, p=SITE_DISTRIBUTION)
    channels = np.random.choice(CHANNELS, size=n_patients, p=CHANNEL_DISTRIBUTION)
    
    # Age (specialty drugs typically adult population)
    ages = np.random.normal(55, 15, n_patients)
    ages = np.clip(ages, 18, 90).astype(int)
    
    # Days since prescription (most recent scripts)
    days_since_script = np.random.exponential(7, n_patients).astype(int)
    days_since_script = np.clip(days_since_script, 0, 30)
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'risk_score': risk_scores,
        'risk_band': risk_bands,
        'payer_type': payer_types,
        'site_type': site_types,
        'channel': channels,
        'age': ages,
        'days_since_script': days_since_script
    })
    
    return df


def generate_risk_scores(n: int) -> np.ndarray:
    """
    Generate realistic risk score distribution.
    
    Real risk scores tend to be somewhat U-shaped: many low-risk, many high-risk,
    fewer medium-risk. We simulate this with a mixture of Beta distributions.
    
    Parameters
    ----------
    n : int
        Number of scores to generate
        
    Returns
    -------
    np.ndarray
        Risk scores between 0 and 1
    """
    # Mixture of three Beta distributions
    # Component 1: Low risk (concentrated near 0)
    low_risk = np.random.beta(2, 8, size=int(n * 0.3))
    
    # Component 2: Medium risk (uniform-ish)
    medium_risk = np.random.beta(3, 3, size=int(n * 0.5))
    
    # Component 3: High risk (concentrated near 1)
    high_risk = np.random.beta(8, 2, size=int(n * 0.2))
    
    # Combine
    risk_scores = np.concatenate([low_risk, medium_risk, high_risk])
    
    # Shuffle
    np.random.shuffle(risk_scores)
    
    # Ensure exactly n scores
    if len(risk_scores) > n:
        risk_scores = risk_scores[:n]
    elif len(risk_scores) < n:
        # Pad with random uniform values
        padding = np.random.uniform(0, 1, n - len(risk_scores))
        risk_scores = np.concatenate([risk_scores, padding])
    
    return risk_scores


def assign_treatment(df: pd.DataFrame, 
                    random_state: int = 42) -> pd.DataFrame:
    """
    Assign FRM intervention based on realistic selection patterns.
    
    This is where SELECTION BIAS is introduced. FRMs don't randomly choose patients;
    they systematically select based on:
    - Risk score (focus on medium-risk)
    - Payer type (Commercial easier than Medicaid)
    - Site type (better relationships with some sites)
    - Other factors
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data from simulate_patients()
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Input df with added columns:
        - intervention_probability: P(treated | X)
        - frm_intervention: 1 if treated, 0 if not
    """
    np.random.seed(random_state)
    
    df = df.copy()
    
    # Base probability from risk band
    base_probs = df['risk_band'].map({
        'Low': FRM_INTERVENTION_PROBS['low'],
        'Medium': FRM_INTERVENTION_PROBS['medium'],
        'High': FRM_INTERVENTION_PROBS['high']
    })
    
    # Adjustments based on other factors
    
    # Payer type: Commercial easier than Medicaid
    payer_multiplier = df['payer_type'].map({
        'Commercial': 1.2,
        'Medicare': 1.0,
        'Medicaid': 0.7,
        'PatientPay': 0.8
    })
    
    # Site type: Academic centers have better FRM relationships
    site_multiplier = df['site_type'].map({
        'Academic': 1.1,
        'Community': 1.0,
        'Specialty': 1.05
    })
    
    # Channel: Hub referrals get more attention
    channel_multiplier = df['channel'].map({
        'Hub': 1.1,
        'NonHub': 0.9
    })
    
    # Days since script: Recent scripts get priority
    days_multiplier = 1.0 + 0.2 * (1 - df['days_since_script'] / 30)
    
    # Combine all factors
    intervention_prob = base_probs * payer_multiplier * site_multiplier * channel_multiplier * days_multiplier
    
    # Clip to [0, 1]
    intervention_prob = np.clip(intervention_prob, 0, 1)
    
    # Assign treatment via Bernoulli draws
    frm_intervention = np.random.binomial(1, intervention_prob)
    
    df['intervention_probability'] = intervention_prob
    df['frm_intervention'] = frm_intervention
    
    return df


def generate_outcomes(df: pd.DataFrame, 
                     random_state: int = 42) -> pd.DataFrame:
    """
    Generate patient outcomes based on covariates and treatment.
    
    This is where we define the TRUE causal effect of treatment.
    We know the ground truth because we're simulating it.
    
    Outcome model:
    - Baseline success probability depends on risk score and covariates
    - Treatment effect depends on risk band (bigger effect in medium-risk)
    - Outcomes are Bernoulli draws
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with treatment assignment
    random_state : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Input df with added columns:
        - baseline_success_prob: P(success) if untreated
        - treatment_effect: True causal effect for this patient
        - success_prob: P(success) given actual treatment
        - outcome: Observed outcome (0/1)
        - outcome_counterfactual: What outcome would have been under opposite treatment
    """
    np.random.seed(random_state)
    
    df = df.copy()
    
    # Baseline success probability (without treatment)
    # This depends on risk score and covariates
    baseline_prob = compute_baseline_success_probability(df)
    
    # Treatment effect (depends on risk band)
    treatment_effect = df['risk_band'].map(TRUE_TREATMENT_EFFECTS)
    
    # Success probability with treatment
    success_prob_treated = np.clip(baseline_prob + treatment_effect, 0, 1)
    success_prob_untreated = baseline_prob
    
    # Actual success probability given actual treatment
    success_prob = np.where(df['frm_intervention'] == 1, 
                            success_prob_treated,
                            success_prob_untreated)
    
    # Generate observed outcomes (Bernoulli)
    outcome = np.random.binomial(1, success_prob)
    
    # Generate counterfactual outcomes (for validation purposes)
    # What would have happened under the opposite treatment?
    counterfactual_prob = np.where(df['frm_intervention'] == 1,
                                    success_prob_untreated,  # What if they WEREN'T treated?
                                    success_prob_treated)    # What if they WERE treated?
    outcome_counterfactual = np.random.binomial(1, counterfactual_prob)
    
    df['baseline_success_prob'] = baseline_prob
    df['treatment_effect'] = treatment_effect
    df['success_prob'] = success_prob
    df['outcome'] = outcome
    df['outcome_counterfactual'] = outcome_counterfactual
    
    return df


def compute_baseline_success_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Compute baseline (untreated) success probability for each patient.
    
    This is a function of risk score and other covariates.
    Higher risk = lower baseline success probability.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data
        
    Returns
    -------
    np.ndarray
        Baseline success probabilities (0-1)
    """
    # Start with risk-based baseline
    # Low risk: high baseline success (~90%)
    # Medium risk: moderate baseline success (~60%)
    # High risk: low baseline success (~30%)
    
    # Use a smooth function based on risk score
    baseline = 0.95 - 0.70 * df['risk_score']  # Linear approximation
    
    # Adjustments based on covariates (smaller effects than treatment)
    
    # Payer: Commercial slightly better than Medicaid
    payer_adjustment = df['payer_type'].map({
        'Commercial': 0.03,
        'Medicare': 0.01,
        'Medicaid': -0.02,
        'PatientPay': 0.00
    })
    
    # Site: Academic centers slightly better
    site_adjustment = df['site_type'].map({
        'Academic': 0.02,
        'Community': 0.00,
        'Specialty': 0.01
    })
    
    # Channel: Hub referrals slightly better
    channel_adjustment = df['channel'].map({
        'Hub': 0.02,
        'NonHub': 0.00
    })
    
    # Age: Younger patients slightly better adherence
    age_adjustment = -0.001 * (df['age'] - 55)  # Centered at 55
    age_adjustment = np.clip(age_adjustment, -0.05, 0.05)
    
    # Combine
    baseline = baseline + payer_adjustment + site_adjustment + channel_adjustment + age_adjustment
    
    # Clip to [0, 1]
    baseline = np.clip(baseline, 0, 1)
    
    return baseline


def compute_true_effects(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute various true treatment effects from the simulated data.
    
    Since we know the counterfactual outcomes, we can compute exact causal effects.
    
    Parameters
    ----------
    df : pd.DataFrame
        Simulated data with outcomes and counterfactuals
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - ate: Average Treatment Effect (population)
        - att: Average Treatment Effect on the Treated
        - atc: Average Treatment Effect on the Controls
        - ate_by_risk: ATE by risk band
    """
    # ATE: Average effect if we treated everyone vs no one
    y1 = np.where(df['frm_intervention'] == 1, 
                  df['outcome'], 
                  df['outcome_counterfactual'])
    y0 = np.where(df['frm_intervention'] == 0, 
                  df['outcome'], 
                  df['outcome_counterfactual'])
    ate = (y1 - y0).mean()
    
    # ATT: Average effect for those who were actually treated
    treated = df[df['frm_intervention'] == 1]
    y1_treated = treated['outcome']
    y0_treated = treated['outcome_counterfactual']
    att = (y1_treated - y0_treated).mean()
    
    # ATC: Average effect for those who were NOT treated
    control = df[df['frm_intervention'] == 0]
    y1_control = control['outcome_counterfactual']
    y0_control = control['outcome']
    atc = (y1_control - y0_control).mean()
    
    # ATE by risk band
    ate_by_risk = {}
    for risk in ['Low', 'Medium', 'High']:
        subset = df[df['risk_band'] == risk]
        y1_subset = np.where(subset['frm_intervention'] == 1, 
                             subset['outcome'], 
                             subset['outcome_counterfactual'])
        y0_subset = np.where(subset['frm_intervention'] == 0, 
                             subset['outcome'], 
                             subset['outcome_counterfactual'])
        ate_by_risk[risk] = (y1_subset - y0_subset).mean()
    
    return {
        'ate': ate,
        'att': att,
        'atc': atc,
        'ate_by_risk': ate_by_risk
    }


# ============================================================================
# MAIN SIMULATION PIPELINE
# ============================================================================

def run_full_simulation(n_patients: int = DEFAULT_N_PATIENTS,
                       random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    """
    Run complete simulation pipeline.
    
    This is a convenience function that runs all steps:
    1. Generate patients
    2. Assign treatment (with selection bias)
    3. Generate outcomes
    4. Compute true effects
    
    Parameters
    ----------
    n_patients : int
        Number of patients to simulate
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    df : pd.DataFrame
        Complete simulated dataset
    true_effects : Dict
        True causal effects for comparison
    """
    print(f"Simulating {n_patients:,} patients...")
    
    # Step 1: Generate patient population
    df = simulate_patients(n_patients, random_state)
    print(f"  ✓ Generated patient characteristics")
    
    # Step 2: Assign FRM intervention (selection bias here!)
    df = assign_treatment(df, random_state)
    n_treated = df['frm_intervention'].sum()
    n_untreated = len(df) - n_treated
    print(f"  ✓ Assigned treatment: {n_treated:,} treated, {n_untreated:,} untreated")
    
    # Step 3: Generate outcomes
    df = generate_outcomes(df, random_state)
    print(f"  ✓ Generated outcomes")
    
    # Step 4: Compute true effects
    true_effects = compute_true_effects(df)
    print(f"\nTRUE CAUSAL EFFECTS:")
    print(f"  ATE (Average Treatment Effect):  {true_effects['ate']:.4f} ({true_effects['ate']*100:.2f}pp)")
    print(f"  ATT (Effect on Treated):         {true_effects['att']:.4f} ({true_effects['att']*100:.2f}pp)")
    print(f"  ATC (Effect on Untreated):       {true_effects['atc']:.4f} ({true_effects['atc']*100:.2f}pp)")
    print(f"  ATE by risk band:")
    for risk, effect in true_effects['ate_by_risk'].items():
        print(f"    {risk:8s}: {effect:.4f} ({effect*100:.2f}pp)")
    
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df, true_effects


# ============================================================================
# COMMAND-LINE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run simulation from command line.
    """
    print("="*70)
    print("SELECTION BIAS SIMULATION - Claritas Rx")
    print("="*70)
    print()
    
    # Run simulation
    df, true_effects = run_full_simulation(n_patients=5000, random_state=42)
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Overall stats
    print(f"\nOverall Success Rates:")
    print(f"  All patients:        {df['outcome'].mean():.3f} ({df['outcome'].mean()*100:.1f}%)")
    print(f"  Treated patients:    {df[df['frm_intervention']==1]['outcome'].mean():.3f}")
    print(f"  Untreated patients:  {df[df['frm_intervention']==0]['outcome'].mean():.3f}")
    
    # Naive estimate
    naive_effect = (df[df['frm_intervention']==1]['outcome'].mean() - 
                   df[df['frm_intervention']==0]['outcome'].mean())
    print(f"\nNAIVE ESTIMATE (Treated - Untreated):")
    print(f"  {naive_effect:.4f} ({naive_effect*100:.2f}pp)")
    
    # Compare to truth
    true_ate = true_effects['ate']
    bias = naive_effect - true_ate
    bias_pct = (bias / true_ate) * 100 if true_ate != 0 else 0
    
    print(f"\nCOMPARISON:")
    print(f"  True ATE:   {true_ate:.4f} ({true_ate*100:.2f}pp)")
    print(f"  Naive Est:  {naive_effect:.4f} ({naive_effect*100:.2f}pp)")
    print(f"  BIAS:       {bias:.4f} ({bias*100:.2f}pp) = {bias_pct:.1f}% error")
    
    if abs(bias) > 0.01:
        print(f"\n  ⚠️  SELECTION BIAS DETECTED!")
        if bias < 0:
            print(f"      Naive estimate UNDER-estimates true effect by {abs(bias_pct):.1f}%")
        else:
            print(f"      Naive estimate OVER-estimates true effect by {abs(bias_pct):.1f}%")
    
    # Risk distribution by treatment status
    print(f"\n" + "="*70)
    print("RISK DISTRIBUTION BY TREATMENT STATUS")
    print("="*70)
    
    for treatment, label in [(1, 'TREATED'), (0, 'UNTREATED')]:
        subset = df[df['frm_intervention'] == treatment]
        print(f"\n{label} (N={len(subset):,}):")
        risk_dist = subset['risk_band'].value_counts(normalize=True).sort_index()
        for risk, pct in risk_dist.items():
            print(f"  {risk:8s}: {pct*100:5.1f}%")
    
    print("\n" + "="*70)
    print("Simulation complete!")
    print("Use analysis_naive_vs_adjusted.py to run corrected analyses.")
    print("="*70)

