"""
Run Complete Incrementality Analysis
Demonstrates all causal inference methods on synthetic specialty pharmacy data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from incrementality_sample_data import IncrementalityDataGenerator
from incrementality_analysis import IncrementalityAnalyzer, QuasiExperimentalDesigns, StudyConfig

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def main():
    """Run complete incrementality analysis"""
    
    print("="*70)
    print(" "*15 + "SPECIALTY PHARMACY INCREMENTALITY ANALYSIS")
    print("="*70)
    print(f"\nAnalysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # =================================================================
    # 1. GENERATE DATA
    # =================================================================
    print("\n" + "="*70)
    print("STEP 1: GENERATE SYNTHETIC DATA")
    print("="*70)
    
    generator = IncrementalityDataGenerator(n_patients=2000)
    datasets = generator.generate_complete_dataset()
    
    analysis_df = datasets['analysis']
    
    # Display basic stats
    print(f"\n📊 Dataset Overview:")
    print(f"   Total Patients: {len(analysis_df):,}")
    print(f"   Treated: {analysis_df['treated'].sum():,} ({analysis_df['treated'].mean():.1%})")
    print(f"   Overall 6-Mo Persistence: {analysis_df['persistence_6mo'].mean():.1%}")
    print(f"   - Treated: {analysis_df[analysis_df['treated']==1]['persistence_6mo'].mean():.1%}")
    print(f"   - Untreated: {analysis_df[analysis_df['treated']==0]['persistence_6mo'].mean():.1%}")
    
    # =================================================================
    # 2. PREPARE ANALYSIS COHORT
    # =================================================================
    print("\n" + "="*70)
    print("STEP 2: PREPARE ANALYSIS COHORT")
    print("="*70)
    
    # Create analysis variables
    analysis_df['outcome'] = analysis_df['persistence_6mo'].astype(int)
    analysis_df['time'] = analysis_df['time_to_discontinuation_days']
    analysis_df['event'] = analysis_df['discontinued'].astype(int)
    
    # Add binary indicators
    analysis_df['gender_M'] = (analysis_df['gender'] == 'M').astype(int)
    analysis_df['payer_commercial'] = (analysis_df['payer_type'] == 'Commercial').astype(int)
    analysis_df['payer_medicare'] = (analysis_df['payer_type'] == 'Medicare').astype(int)
    analysis_df['high_cost_specialty'] = analysis_df['specialty_condition'].isin(
        ['Oncology', 'MS/Neurology', 'Hepatitis']
    ).astype(int)
    analysis_df['has_caregiver'] = analysis_df['has_caregiver'].astype(int)
    
    # Define covariates
    covariate_cols = [
        'age', 'risk_score', 'prior_adherence', 'cci_score',
        'distance_to_pharmacy_miles', 'prescriber_experience_years',
        'gender_M', 'payer_commercial', 'payer_medicare', 
        'high_cost_specialty', 'has_caregiver'
    ]
    
    # Handle missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    
    X = analysis_df[covariate_cols].values
    X = imputer.fit_transform(X)  # Impute missing values
    
    treatment = analysis_df['treated'].values
    outcome = analysis_df['outcome'].values
    time = analysis_df['time'].values
    event = analysis_df['event'].values
    
    print(f"\n✅ Cohort prepared:")
    print(f"   Patients: {len(analysis_df):,}")
    print(f"   Covariates: {len(covariate_cols)}")
    print(f"   Missing values imputed with median")
    
    # =================================================================
    # 3. CAUSAL INFERENCE ANALYSIS
    # =================================================================
    print("\n" + "="*70)
    print("STEP 3: CAUSAL INFERENCE ANALYSIS")
    print("="*70)
    
    # Initialize analyzer
    config = StudyConfig(
        treatment_window_days=7,
        outcome_window_days=180,
        washout_period_days=30,
        risk_threshold=0.5,
        balance_threshold=0.1,
        bootstrap_iterations=500,
        random_state=42
    )
    
    analyzer = IncrementalityAnalyzer(config)
    
    # 3.1 Propensity Scores
    print("\n📊 3.1 Estimating Propensity Scores...")
    ps_logistic = analyzer.estimate_propensity_scores(X, treatment, method='logistic')
    ps_gbm = analyzer.estimate_propensity_scores(X, treatment, method='gbm')
    
    print(f"   Propensity Score Summary:")
    print(f"   - Treated: {ps_logistic[treatment==1].mean():.3f} ± {ps_logistic[treatment==1].std():.3f}")
    print(f"   - Untreated: {ps_logistic[treatment==0].mean():.3f} ± {ps_logistic[treatment==0].std():.3f}")
    
    # 3.2 Calculate Weights and Check Balance
    print("\n⚖️  3.2 Calculating Weights and Checking Balance...")
    iptw_weights = analyzer.calculate_weights(ps_logistic, treatment, method='iptw')
    overlap_weights = analyzer.calculate_weights(ps_logistic, treatment, method='overlap')
    
    balance_unweighted = analyzer.check_balance(X, treatment, weights=None, feature_names=covariate_cols)
    balance_iptw = analyzer.check_balance(X, treatment, weights=iptw_weights, feature_names=covariate_cols)
    
    # Extract SMD values from DataFrame
    max_smd_unweighted = np.abs(balance_unweighted['smd']).max()
    max_smd_iptw = np.abs(balance_iptw['smd']).max()
    
    print(f"   Balance Assessment:")
    print(f"   - Max SMD (Unweighted): {max_smd_unweighted:.3f}")
    print(f"   - Max SMD (IPTW): {max_smd_iptw:.3f} {'✓ PASS' if max_smd_iptw < 0.1 else '✗ FAIL'}")
    
    # 3.3 AIPW Estimation
    print("\n🎯 3.3 Estimating Treatment Effects (AIPW - Doubly Robust)...")
    aipw_results = analyzer.estimate_aipw(X, treatment, outcome, ps_logistic)
    
    print(f"\n   PRIMARY RESULTS:")
    print(f"   - ATT: {aipw_results['att']:.3f} (95% CI: [{aipw_results['ci_lower']:.3f}, {aipw_results['ci_upper']:.3f}])")
    print(f"   - P-value: {aipw_results['p_value']:.4f}")
    print(f"   - N Treated: {aipw_results['n_treated']}")
    print(f"   - N Control: {aipw_results['n_control']}")
    
    # Calculate naive difference for comparison
    naive_diff = outcome[treatment==1].mean() - outcome[treatment==0].mean()
    print(f"\n   📊 Naive Difference: {naive_diff:.3f}")
    print(f"   📊 AIPW Estimate (ATT): {aipw_results['att']:.3f}")
    
    if aipw_results['att'] > 0:
        print(f"\n   ✅ Intervention INCREASES persistence by {aipw_results['att']:.1%} (absolute)")
        baseline_persistence = outcome[treatment==0].mean()
        relative_lift = aipw_results['att'] / baseline_persistence if baseline_persistence > 0 else 0
        print(f"   📈 Relative lift: {relative_lift:.1%}")
    
    # 3.4 Survival Analysis
    print("\n⏱️  3.4 Survival Analysis (Cox Proportional Hazards)...")
    survival_results = analyzer.survival_analysis(
        time_to_event=time,
        event=event,
        treatment=treatment,
        X=X,
        weights=iptw_weights
    )
    
    print(f"   Hazard Ratio: {survival_results['hazard_ratio']:.3f} ({survival_results['hr_ci'][0]:.3f}, {survival_results['hr_ci'][1]:.3f})")
    print(f"   Median Time Difference: {survival_results['median_time_treated'] - survival_results['median_time_untreated']:.1f} days")
    
    if survival_results['hazard_ratio'] < 1:
        reduction = (1 - survival_results['hazard_ratio']) * 100
        print(f"   ✅ Intervention REDUCES discontinuation risk by {reduction:.1f}%")
    
    # 3.5 Heterogeneous Effects
    print("\n🔍 3.5 Estimating Heterogeneous Treatment Effects (CATE)...")
    cate_results = analyzer.estimate_heterogeneous_effects(X, treatment, outcome)
    
    if 'cate_estimates' in cate_results:
        cate_scores = cate_results['cate_estimates']
        print(f"   CATE Summary:")
        print(f"   - Mean: {cate_scores.mean():.3f}")
        print(f"   - Range: [{cate_scores.min():.3f}, {cate_scores.max():.3f}]")
        print(f"   - High Responders (Top 25%): {cate_scores[cate_scores > np.percentile(cate_scores, 75)].mean():.3f}")
    elif 'subgroups' in cate_results:
        print(f"   Subgroup Analysis (simplified):")
        for subgroup_name, effect in list(cate_results['subgroups'].items())[:5]:
            if not np.isnan(effect):
                print(f"   - {subgroup_name}: {effect:.3f}")
        cate_scores = np.zeros(len(treatment))  # Placeholder
    
    # 3.6 Business Metrics
    print("\n💰 3.6 Calculating Business Metrics...")
    business_results = analyzer.calculate_business_metrics(
        treatment_effect=aipw_results['att'],
        n_treated=int(aipw_results['n_treated']),
        cost_per_intervention=250,
        value_per_persistent_patient=analysis_df['monthly_cost'].mean() * 6  # 6 months of therapy
    )
    
    print(f"   Business Impact:")
    print(f"   - NNT: {business_results['nnt']:.1f}")
    print(f"   - Avoided Discontinuations: {business_results['avoided_discontinuations']:.0f}")
    print(f"   - Total Cost: ${business_results['total_cost']:,.0f}")
    print(f"   - Total Value: ${business_results['total_value']:,.0f}")
    print(f"   - Net Value: ${business_results['total_value'] - business_results['total_cost']:,.0f}")
    print(f"   - ROI: {business_results['roi']:.1%}")
    print(f"   - Incremental Value per Intervention: ${business_results['incremental_value_per_intervention']:,.0f}")
    
    if business_results['roi'] > 1:
        print(f"\n   ✅ POSITIVE ROI: Every $1 invested returns ${business_results['roi']:.2f}")
    
    # 3.7 Sensitivity Analysis
    print("\n🔬 3.7 Running Sensitivity Analysis...")
    sensitivity_results = analyzer.sensitivity_analysis(
        treatment_effect=aipw_results['att'],
        se=aipw_results['se']
    )
    
    print(f"   E-Value: {sensitivity_results['e_value']:.3f}")
    print(f"   Breaking Gamma: {sensitivity_results['breaking_gamma']:.3f}")
    print(f"   Standard Error: {aipw_results['se']:.3f}")
    
    if sensitivity_results['e_value'] > 2:
        print(f"   ✅ Strong evidence - result is robust to unmeasured confounding")
    else:
        print(f"   ⚠️  Moderate robustness - some concern about unmeasured confounding")
    
    # =================================================================
    # 4. EXECUTIVE SUMMARY
    # =================================================================
    print("\n" + "="*70)
    print(" "*20 + "EXECUTIVE SUMMARY")
    print("="*70)
    
    print(f"\n🎯 PRIMARY FINDING:")
    print(f"   Intervention increases 6-month persistence by {aipw_results['att']:.1%}")
    print(f"   95% CI: ({aipw_results['ci_lower']:.1%}, {aipw_results['ci_upper']:.1%})")
    print(f"   Statistical Significance: p = {aipw_results['p_value']:.4f}")
    
    net_value = business_results['total_value'] - business_results['total_cost']
    print(f"\n💰 BUSINESS IMPACT:")
    print(f"   ROI: {business_results['roi']:.1%}")
    print(f"   Net Value: ${net_value:,.0f}")
    print(f"   NNT: {business_results['nnt']:.1f} patients")
    
    print(f"\n🔬 ROBUSTNESS:")
    print(f"   E-Value: {sensitivity_results['e_value']:.3f}")
    print(f"   Max Covariate Imbalance: {max_smd_iptw:.3f}")
    
    print(f"\n📊 RECOMMENDATION:")
    if aipw_results['att'] > 0 and business_results['roi'] > 1:
        print(f"   ✅ CONTINUE AND EXPAND PROGRAM")
        print(f"   - Strong evidence of positive impact")
        print(f"   - Excellent ROI")
        print(f"   - Focus on high-responder segments")
    else:
        print(f"   ⚠️  REVIEW PROGRAM DESIGN")
    
    # =================================================================
    # 5. SAVE RESULTS
    # =================================================================
    print("\n" + "="*70)
    print("STEP 4: SAVING RESULTS")
    print("="*70)
    
    # Save summary
    results_summary = pd.DataFrame([
        {
            'metric': 'ATT (AIPW)',
            'estimate': aipw_results['att'],
            'ci_lower': aipw_results['ci_lower'],
            'ci_upper': aipw_results['ci_upper'],
            'p_value': aipw_results['p_value']
        },
        {
            'metric': 'Hazard Ratio',
            'estimate': survival_results['hazard_ratio'],
            'ci_lower': survival_results['hr_ci'][0],
            'ci_upper': survival_results['hr_ci'][1],
            'p_value': survival_results['p_value']
        },
        {
            'metric': 'NNT',
            'estimate': business_results['nnt'],
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_value': np.nan
        },
        {
            'metric': 'ROI',
            'estimate': business_results['roi'],
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'p_value': np.nan
        }
    ])
    
    results_summary.to_csv('incrementality_results_summary.csv', index=False)
    analysis_df.to_csv('incrementality_analysis_cohort.csv', index=False)
    
    print("\n✅ Results saved:")
    print("   - incrementality_results_summary.csv")
    print("   - incrementality_analysis_cohort.csv")
    
    print("\n" + "="*70)
    print(" "*25 + "ANALYSIS COMPLETE!")
    print("="*70)
    
    return {
        'aipw': aipw_results,
        'survival': survival_results,
        'cate': cate_results,
        'business': business_results,
        'sensitivity': sensitivity_results,
        'datasets': datasets
    }


if __name__ == '__main__':
    results = main()

