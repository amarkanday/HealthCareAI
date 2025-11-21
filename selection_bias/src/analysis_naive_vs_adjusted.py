"""
Causal Inference Analysis Methods
Claritas Rx - Selection Bias Correction

This module implements various methods to estimate treatment effects:
1. Naive comparison (biased)
2. Regression adjustment
3. Propensity score methods (matching, stratification, IPW)
4. Doubly robust estimation (AIPW)

All methods can be applied to both simulated and real data.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. NAIVE ESTIMATION (Biased Baseline)
# ============================================================================

def naive_estimate(treatment: np.ndarray, 
                  outcome: np.ndarray) -> float:
    """
    Naive treatment effect estimate: simple difference in means.
    
    This is BIASED when treatment is not randomly assigned!
    
    Formula:
        ATE_naive = E[Y | T=1] - E[Y | T=0]
    
    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator (1 = treated, 0 = untreated)
    outcome : np.ndarray
        Binary outcome (1 = success, 0 = failure)
        
    Returns
    -------
    float
        Naive treatment effect estimate
    """
    treated_mean = outcome[treatment == 1].mean()
    untreated_mean = outcome[treatment == 0].mean()
    
    return treated_mean - untreated_mean


def naive_estimate_with_ci(treatment: np.ndarray,
                           outcome: np.ndarray,
                           alpha: float = 0.05) -> Tuple[float, float, float]:
    """
    Naive estimate with confidence interval.
    
    Uses two-sample t-test for proportions.
    
    Parameters
    ----------
    treatment : np.ndarray
        Binary treatment indicator
    outcome : np.ndarray
        Binary outcome
    alpha : float
        Significance level for CI (default 0.05 for 95% CI)
        
    Returns
    -------
    estimate : float
        Point estimate
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    """
    treated = outcome[treatment == 1]
    untreated = outcome[treatment == 0]
    
    n1, n0 = len(treated), len(untreated)
    p1, p0 = treated.mean(), untreated.mean()
    
    # Standard error for difference in proportions
    se = np.sqrt(p1*(1-p1)/n1 + p0*(1-p0)/n0)
    
    # Z-score for confidence level
    z = stats.norm.ppf(1 - alpha/2)
    
    estimate = p1 - p0
    ci_lower = estimate - z * se
    ci_upper = estimate + z * se
    
    return estimate, ci_lower, ci_upper


# ============================================================================
# 2. REGRESSION ADJUSTMENT
# ============================================================================

def regression_adjusted_estimate(X: pd.DataFrame,
                                treatment: np.ndarray,
                                outcome: np.ndarray) -> Tuple[float, float]:
    """
    Regression-adjusted treatment effect estimate.
    
    Fits logistic regression: outcome ~ treatment + covariates
    Coefficient on treatment is the adjusted effect.
    
    Assumptions:
    - Correct model specification (linearity, additivity)
    - No unmeasured confounders
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix (WITHOUT treatment)
    treatment : np.ndarray
        Binary treatment indicator
    outcome : np.ndarray
        Binary outcome
        
    Returns
    -------
    estimate : float
        Adjusted treatment effect (marginal effect)
    se : float
        Standard error
    """
    # Prepare data: combine treatment with covariates
    X_with_treatment = X.copy()
    X_with_treatment['treatment'] = treatment
    
    # Standardize continuous covariates (helps with convergence)
    scaler = StandardScaler()
    continuous_cols = X_with_treatment.select_dtypes(include=[np.number]).columns
    continuous_cols = [c for c in continuous_cols if c != 'treatment']
    
    if len(continuous_cols) > 0:
        X_with_treatment[continuous_cols] = scaler.fit_transform(X_with_treatment[continuous_cols])
    
    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_with_treatment, outcome)
    
    # Estimate average marginal effect of treatment
    # Method: predict with treatment=1 vs treatment=0, average the difference
    X_treated = X_with_treatment.copy()
    X_treated['treatment'] = 1
    
    X_untreated = X_with_treatment.copy()
    X_untreated['treatment'] = 0
    
    pred_treated = model.predict_proba(X_treated)[:, 1]
    pred_untreated = model.predict_proba(X_untreated)[:, 1]
    
    marginal_effect = (pred_treated - pred_untreated).mean()
    
    # Estimate standard error via bootstrap
    se = bootstrap_se(lambda: regression_adjusted_estimate_single(X, treatment, outcome), 
                     n_bootstrap=100)
    
    return marginal_effect, se


def regression_adjusted_estimate_single(X: pd.DataFrame, 
                                       treatment: np.ndarray, 
                                       outcome: np.ndarray) -> float:
    """Helper function for bootstrap."""
    X_with_treatment = X.copy()
    X_with_treatment['treatment'] = treatment
    
    model = LogisticRegression(max_iter=1000, random_state=None)
    model.fit(X_with_treatment, outcome)
    
    X_treated = X_with_treatment.copy()
    X_treated['treatment'] = 1
    X_untreated = X_with_treatment.copy()
    X_untreated['treatment'] = 0
    
    pred_treated = model.predict_proba(X_treated)[:, 1]
    pred_untreated = model.predict_proba(X_untreated)[:, 1]
    
    return (pred_treated - pred_untreated).mean()


# ============================================================================
# 3. PROPENSITY SCORE METHODS
# ============================================================================

def estimate_propensity_scores(X: pd.DataFrame, 
                              treatment: np.ndarray) -> np.ndarray:
    """
    Estimate propensity scores: P(Treatment | X).
    
    Uses logistic regression to model treatment assignment.
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix
    treatment : np.ndarray
        Binary treatment indicator
        
    Returns
    -------
    np.ndarray
        Propensity scores (0-1)
    """
    # Standardize continuous covariates
    X_scaled = X.copy()
    scaler = StandardScaler()
    continuous_cols = X_scaled.select_dtypes(include=[np.number]).columns
    
    if len(continuous_cols) > 0:
        X_scaled[continuous_cols] = scaler.fit_transform(X_scaled[continuous_cols])
    
    # Fit propensity score model
    ps_model = LogisticRegression(max_iter=1000, random_state=42)
    ps_model.fit(X_scaled, treatment)
    
    # Predict propensity scores
    propensity_scores = ps_model.predict_proba(X_scaled)[:, 1]
    
    return propensity_scores


def propensity_weighted_estimate(X: pd.DataFrame,
                                treatment: np.ndarray,
                                outcome: np.ndarray,
                                trim_threshold: float = 0.01) -> Tuple[float, float]:
    """
    Inverse propensity weighting (IPW) estimate.
    
    Weights:
    - Treated: weight = 1 / propensity_score
    - Untreated: weight = 1 / (1 - propensity_score)
    
    This creates a "pseudo-population" where treatment is independent of X.
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix
    treatment : np.ndarray
        Binary treatment indicator
    outcome : np.ndarray
        Binary outcome
    trim_threshold : float
        Trim extreme propensity scores (default 0.01 = keep [0.01, 0.99])
        
    Returns
    -------
    estimate : float
        IPW treatment effect estimate
    se : float
        Standard error
    """
    # Estimate propensity scores
    ps = estimate_propensity_scores(X, treatment)
    
    # Trim extreme values (optional but recommended for stability)
    ps = np.clip(ps, trim_threshold, 1 - trim_threshold)
    
    # Calculate weights
    weights = np.where(treatment == 1, 
                      1 / ps,
                      1 / (1 - ps))
    
    # Weighted means
    weighted_treated_mean = np.average(outcome[treatment == 1], 
                                       weights=weights[treatment == 1])
    weighted_untreated_mean = np.average(outcome[treatment == 0], 
                                         weights=weights[treatment == 0])
    
    estimate = weighted_treated_mean - weighted_untreated_mean
    
    # Estimate standard error
    # Simplified formula (ignores PS estimation uncertainty)
    n = len(outcome)
    var_treated = np.average((outcome[treatment == 1] - weighted_treated_mean)**2,
                            weights=weights[treatment == 1])
    var_untreated = np.average((outcome[treatment == 0] - weighted_untreated_mean)**2,
                              weights=weights[treatment == 0])
    
    se = np.sqrt(var_treated / (treatment == 1).sum() + var_untreated / (treatment == 0).sum())
    
    return estimate, se


def check_covariate_balance(X: pd.DataFrame,
                           treatment: np.ndarray,
                           weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Check covariate balance between treated and untreated groups.
    
    Standardized Mean Difference (SMD) is the standard metric.
    Rule of thumb: |SMD| < 0.1 indicates good balance.
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix
    treatment : np.ndarray
        Binary treatment indicator
    weights : np.ndarray, optional
        Weights for weighted balance check (e.g., IPW weights)
        
    Returns
    -------
    pd.DataFrame
        Balance table with SMD for each covariate
    """
    balance_results = []
    
    for col in X.columns:
        if X[col].dtype in [np.float64, np.int64]:
            # Continuous variable
            if weights is None:
                mean_treated = X.loc[treatment == 1, col].mean()
                mean_untreated = X.loc[treatment == 0, col].mean()
                std_treated = X.loc[treatment == 1, col].std()
                std_untreated = X.loc[treatment == 0, col].std()
            else:
                mean_treated = np.average(X.loc[treatment == 1, col], 
                                         weights=weights[treatment == 1])
                mean_untreated = np.average(X.loc[treatment == 0, col], 
                                           weights=weights[treatment == 0])
                std_treated = np.sqrt(np.average((X.loc[treatment == 1, col] - mean_treated)**2,
                                                weights=weights[treatment == 1]))
                std_untreated = np.sqrt(np.average((X.loc[treatment == 0, col] - mean_untreated)**2,
                                                  weights=weights[treatment == 0]))
            
            # Pooled standard deviation
            pooled_std = np.sqrt((std_treated**2 + std_untreated**2) / 2)
            
            # Standardized mean difference
            smd = (mean_treated - mean_untreated) / pooled_std if pooled_std > 0 else 0
            
            balance_results.append({
                'covariate': col,
                'mean_treated': mean_treated,
                'mean_untreated': mean_untreated,
                'smd': smd,
                'balanced': abs(smd) < 0.1
            })
    
    return pd.DataFrame(balance_results)


# ============================================================================
# 4. DOUBLY ROBUST ESTIMATION (AIPW)
# ============================================================================

def doubly_robust_estimate(X: pd.DataFrame,
                          treatment: np.ndarray,
                          outcome: np.ndarray,
                          trim_threshold: float = 0.01) -> Tuple[float, float]:
    """
    Doubly robust (AIPW) treatment effect estimate.
    
    Combines propensity score weighting with outcome regression.
    Consistent if EITHER the propensity model OR the outcome model is correct.
    
    Formula:
        τ_AIPW = (1/n) Σ [
            (T_i / e(X_i)) * (Y_i - μ_1(X_i))
            - ((1-T_i) / (1-e(X_i))) * (Y_i - μ_0(X_i))
            + μ_1(X_i) - μ_0(X_i)
        ]
    
    Where:
    - e(X) = propensity score
    - μ_1(X) = E[Y | T=1, X]
    - μ_0(X) = E[Y | T=0, X]
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix
    treatment : np.ndarray
        Binary treatment indicator
    outcome : np.ndarray
        Binary outcome
    trim_threshold : float
        Trim extreme propensity scores
        
    Returns
    -------
    estimate : float
        Doubly robust treatment effect estimate
    se : float
        Standard error
    """
    n = len(outcome)
    
    # Step 1: Estimate propensity scores e(X)
    ps = estimate_propensity_scores(X, treatment)
    ps = np.clip(ps, trim_threshold, 1 - trim_threshold)
    
    # Step 2: Fit outcome models μ_1(X) and μ_0(X)
    X_scaled = X.copy()
    scaler = StandardScaler()
    continuous_cols = X_scaled.select_dtypes(include=[np.number]).columns
    if len(continuous_cols) > 0:
        X_scaled[continuous_cols] = scaler.fit_transform(X_scaled[continuous_cols])
    
    # Model for treated: μ_1(X) = E[Y | T=1, X]
    model_treated = LogisticRegression(max_iter=1000, random_state=42)
    model_treated.fit(X_scaled[treatment == 1], outcome[treatment == 1])
    mu1 = model_treated.predict_proba(X_scaled)[:, 1]
    
    # Model for untreated: μ_0(X) = E[Y | T=0, X]
    model_untreated = LogisticRegression(max_iter=1000, random_state=42)
    model_untreated.fit(X_scaled[treatment == 0], outcome[treatment == 0])
    mu0 = model_untreated.predict_proba(X_scaled)[:, 1]
    
    # Step 3: Compute AIPW estimator
    aipw_components = (
        (treatment / ps) * (outcome - mu1)
        - ((1 - treatment) / (1 - ps)) * (outcome - mu0)
        + (mu1 - mu0)
    )
    
    estimate = aipw_components.mean()
    
    # Standard error (empirical variance)
    se = aipw_components.std() / np.sqrt(n)
    
    return estimate, se


# ============================================================================
# 5. UTILITY FUNCTIONS
# ============================================================================

def bootstrap_se(estimator_func, n_bootstrap: int = 200, random_state: int = 42) -> float:
    """
    Estimate standard error via bootstrap.
    
    Parameters
    ----------
    estimator_func : callable
        Function that returns a point estimate (takes no arguments)
    n_bootstrap : int
        Number of bootstrap samples
    random_state : int
        Random seed
        
    Returns
    -------
    float
        Bootstrap standard error
    """
    np.random.seed(random_state)
    
    estimates = []
    for _ in range(n_bootstrap):
        try:
            est = estimator_func()
            estimates.append(est)
        except:
            pass  # Skip failed bootstrap samples
    
    if len(estimates) < n_bootstrap * 0.5:
        warnings.warn(f"Only {len(estimates)} out of {n_bootstrap} bootstrap samples succeeded")
    
    return np.std(estimates) if len(estimates) > 0 else np.nan


def compare_all_methods(df: pd.DataFrame,
                       covariate_cols: list,
                       true_ate: Optional[float] = None) -> pd.DataFrame:
    """
    Run all estimation methods and compare results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with columns:
        - frm_intervention: treatment indicator
        - outcome: binary outcome
        - covariate_cols: list of covariate names
    covariate_cols : list
        Names of covariate columns
    true_ate : float, optional
        True ATE if known (for simulated data)
        
    Returns
    -------
    pd.DataFrame
        Comparison table with estimates, standard errors, CIs, and bias
    """
    X = df[covariate_cols]
    treatment = df['frm_intervention'].values
    outcome = df['outcome'].values
    
    results = []
    
    # 1. Naive estimate
    naive_est, naive_ci_lower, naive_ci_upper = naive_estimate_with_ci(treatment, outcome)
    naive_se = (naive_ci_upper - naive_ci_lower) / (2 * 1.96)
    results.append({
        'method': 'Naive',
        'estimate': naive_est,
        'se': naive_se,
        'ci_lower': naive_ci_lower,
        'ci_upper': naive_ci_upper,
        'bias': naive_est - true_ate if true_ate is not None else np.nan,
        'bias_pct': ((naive_est - true_ate) / true_ate * 100) if true_ate is not None else np.nan
    })
    
    # 2. Regression adjustment
    try:
        reg_est, reg_se = regression_adjusted_estimate(X, treatment, outcome)
        reg_ci_lower = reg_est - 1.96 * reg_se
        reg_ci_upper = reg_est + 1.96 * reg_se
        results.append({
            'method': 'Regression-Adjusted',
            'estimate': reg_est,
            'se': reg_se,
            'ci_lower': reg_ci_lower,
            'ci_upper': reg_ci_upper,
            'bias': reg_est - true_ate if true_ate is not None else np.nan,
            'bias_pct': ((reg_est - true_ate) / true_ate * 100) if true_ate is not None else np.nan
        })
    except Exception as e:
        print(f"Regression adjustment failed: {e}")
    
    # 3. Propensity weighting
    try:
        ipw_est, ipw_se = propensity_weighted_estimate(X, treatment, outcome)
        ipw_ci_lower = ipw_est - 1.96 * ipw_se
        ipw_ci_upper = ipw_est + 1.96 * ipw_se
        results.append({
            'method': 'Propensity Weighted (IPW)',
            'estimate': ipw_est,
            'se': ipw_se,
            'ci_lower': ipw_ci_lower,
            'ci_upper': ipw_ci_upper,
            'bias': ipw_est - true_ate if true_ate is not None else np.nan,
            'bias_pct': ((ipw_est - true_ate) / true_ate * 100) if true_ate is not None else np.nan
        })
    except Exception as e:
        print(f"Propensity weighting failed: {e}")
    
    # 4. Doubly robust
    try:
        dr_est, dr_se = doubly_robust_estimate(X, treatment, outcome)
        dr_ci_lower = dr_est - 1.96 * dr_se
        dr_ci_upper = dr_est + 1.96 * dr_se
        results.append({
            'method': 'Doubly Robust (AIPW)',
            'estimate': dr_est,
            'se': dr_se,
            'ci_lower': dr_ci_lower,
            'ci_upper': dr_ci_upper,
            'bias': dr_est - true_ate if true_ate is not None else np.nan,
            'bias_pct': ((dr_est - true_ate) / true_ate * 100) if true_ate is not None else np.nan
        })
    except Exception as e:
        print(f"Doubly robust estimation failed: {e}")
    
    results_df = pd.DataFrame(results)
    
    # Add true ATE row if provided
    if true_ate is not None:
        true_row = pd.DataFrame([{
            'method': 'True ATE',
            'estimate': true_ate,
            'se': 0,
            'ci_lower': true_ate,
            'ci_upper': true_ate,
            'bias': 0,
            'bias_pct': 0
        }])
        results_df = pd.concat([true_row, results_df], ignore_index=True)
    
    return results_df


# ============================================================================
# COMMAND-LINE EXECUTION
# ============================================================================

if __name__ == "__main__":
    """
    Run analysis on simulated data.
    """
    print("="*70)
    print("CAUSAL INFERENCE ANALYSIS - Claritas Rx")
    print("="*70)
    print()
    
    # Import simulation module
    try:
        from data_simulation import run_full_simulation
    except ImportError:
        print("Error: Cannot import data_simulation module.")
        print("Make sure you're running from the src/ directory or have it in your PYTHONPATH")
        exit(1)
    
    # Generate data
    print("Generating simulated data...")
    df, true_effects = run_full_simulation(n_patients=5000, random_state=42)
    
    print("\n" + "="*70)
    print("RUNNING ALL ESTIMATION METHODS")
    print("="*70)
    
    # Define covariates
    covariate_cols = ['risk_score', 'payer_type', 'site_type', 'channel', 'age', 'days_since_script']
    
    # Encode categorical variables
    df_encoded = df.copy()
    df_encoded = pd.get_dummies(df_encoded, columns=['payer_type', 'site_type', 'channel'], drop_first=True)
    
    # Get encoded covariate columns
    covariate_cols_encoded = [col for col in df_encoded.columns 
                              if col.startswith(('risk_score', 'payer_type_', 'site_type_', 'channel_', 'age', 'days'))]
    
    # Compare all methods
    comparison = compare_all_methods(df_encoded, covariate_cols_encoded, true_ate=true_effects['ate'])
    
    print("\nRESULTS:")
    print(comparison.to_string(index=False))
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    # Find best method (smallest absolute bias)
    best_idx = comparison[comparison['method'] != 'True ATE']['bias'].abs().idxmin()
    best_method = comparison.loc[best_idx, 'method']
    best_bias = comparison.loc[best_idx, 'bias_pct']
    
    print(f"\nBest performing method: {best_method}")
    print(f"  Bias: {best_bias:.1f}%")
    
    naive_bias = comparison[comparison['method'] == 'Naive']['bias_pct'].values[0]
    print(f"\nNaive method bias: {naive_bias:.1f}%")
    
    if abs(naive_bias) > 10:
        print("  ⚠️  Naive estimate is severely biased!")
        print("  → Always use adjusted methods in observational studies")
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

