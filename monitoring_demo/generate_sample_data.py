"""
Generate sample patient feature data and model predictions for monitoring demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


def generate_patient_features(date, n_patients=2500, introduce_drift=False, introduce_quality_issues=False):
    """
    Generate synthetic patient features for a given date
    
    Features:
    - age: Patient age (continuous)
    - days_on_therapy: Days since therapy start (continuous)
    - discontinuation_risk_score: Existing risk score 0-1 (continuous)
    - adherence_mpr: Medication Possession Ratio (continuous)
    - comorbidity_count: Number of comorbidities (discrete)
    - interactions_30d: FRM interactions in last 30 days (discrete)
    - region: Geographic region (categorical)
    - insurance_type: Type of insurance (categorical)
    """
    np.random.seed(hash(str(date)) % 2**32)
    
    # Baseline parameters
    age_mean, age_std = 55, 12
    days_mean, days_std = 180, 90
    risk_mean, risk_std = 0.25, 0.15
    mpr_mean, mpr_std = 0.80, 0.15
    
    # Introduce drift if specified
    if introduce_drift:
        age_mean += 5  # Population getting older
        risk_mean += 0.10  # Risk scores increasing
        mpr_mean -= 0.10  # Adherence decreasing
    
    data = {
        'patient_id': [f'P{i:05d}' for i in range(1, n_patients + 1)],
        'date': [date] * n_patients,
        'age': np.clip(np.random.normal(age_mean, age_std, n_patients), 18, 90),
        'days_on_therapy': np.clip(np.random.normal(days_mean, days_std, n_patients), 1, 730),
        'discontinuation_risk_score': np.clip(np.random.normal(risk_mean, risk_std, n_patients), 0, 1),
        'adherence_mpr': np.clip(np.random.normal(mpr_mean, mpr_std, n_patients), 0, 1),
        'comorbidity_count': np.random.poisson(2, n_patients),
        'interactions_30d': np.random.poisson(1.5, n_patients),
    }
    
    # Categorical features
    region_probs = [0.30, 0.25, 0.20, 0.15, 0.10]
    if introduce_drift:
        region_probs = [0.20, 0.20, 0.25, 0.20, 0.15]  # Distribution shift
    
    data['region'] = np.random.choice(
        ['Northeast', 'Southeast', 'Midwest', 'West', 'Southwest'],
        size=n_patients,
        p=region_probs
    )
    
    insurance_probs = [0.60, 0.25, 0.10, 0.05]
    data['insurance_type'] = np.random.choice(
        ['Commercial', 'Medicare', 'Medicaid', 'PatientPay'],
        size=n_patients,
        p=insurance_probs
    )
    
    df = pd.DataFrame(data)
    
    # Introduce quality issues if specified
    if introduce_quality_issues:
        # Randomly set some values to null (higher rate than baseline)
        null_mask_age = np.random.random(n_patients) < 0.20  # 20% nulls
        df.loc[null_mask_age, 'age'] = np.nan
        
        null_mask_mpr = np.random.random(n_patients) < 0.15  # 15% nulls
        df.loc[null_mask_mpr, 'adherence_mpr'] = np.nan
    else:
        # Baseline null rate (5%)
        null_mask = np.random.random(n_patients) < 0.05
        df.loc[null_mask, 'age'] = np.nan
    
    return df


def generate_30_days_data(output_dir='./data'):
    """Generate 30 days of patient feature data"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    start_date = datetime(2025, 1, 1)
    daily_data = []
    
    for day in range(30):
        current_date = start_date + timedelta(days=day)
        
        # Introduce drift on days 15-21
        drift = (15 <= day < 22)
        
        # Introduce quality issues on days 22-30
        quality_issues = (day >= 22)
        
        df = generate_patient_features(
            current_date,
            introduce_drift=drift,
            introduce_quality_issues=quality_issues
        )
        
        # Save individual day file
        filename = f"{output_dir}/patient_features_{current_date.strftime('%Y%m%d')}.parquet"
        df.to_parquet(filename, index=False)
        
        daily_data.append(df)
        print(f"Generated: Day {day+1:2d} ({current_date.strftime('%Y-%m-%d')}) - {len(df):,} patients")
    
    # Save baseline (first 14 days combined)
    baseline_df = pd.concat(daily_data[:14], ignore_index=True)
    baseline_df.to_parquet(f"{output_dir}/baseline_features.parquet", index=False)
    print(f"\n✓ Baseline data saved: {len(baseline_df):,} rows")
    
    return daily_data


def train_sample_model(baseline_df, output_dir='./models'):
    """
    Train a sample Random Forest model for uplift prediction
    This is a simplified T-Learner approach
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTraining sample Random Forest models...")
    
    # Prepare features
    feature_cols = ['age', 'days_on_therapy', 'discontinuation_risk_score', 
                    'adherence_mpr', 'comorbidity_count', 'interactions_30d']
    
    X = baseline_df[feature_cols].fillna(baseline_df[feature_cols].median())
    
    # Generate synthetic outcomes and treatment indicators
    np.random.seed(42)
    n = len(X)
    
    # Treatment assignment (60% treated)
    treatment = np.random.binomial(1, 0.6, n)
    
    # Generate outcomes with treatment effect
    # Base probability from risk score
    base_prob = baseline_df['discontinuation_risk_score'].values
    
    # Treatment effect (reduces discontinuation by ~15%)
    treatment_effect = -0.15 * treatment
    
    # Actual outcome
    outcome_prob = np.clip(base_prob + treatment_effect, 0, 1)
    outcome = np.random.binomial(1, outcome_prob)
    
    # Train two models (T-Learner)
    # Model for treated patients
    X_treated = X[treatment == 1]
    y_treated = outcome[treatment == 1]
    
    model_treated = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        random_state=42
    )
    model_treated.fit(X_treated, y_treated)
    
    # Model for control patients
    X_control = X[treatment == 0]
    y_control = outcome[treatment == 0]
    
    model_control = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=50,
        random_state=42
    )
    model_control.fit(X_control, y_control)
    
    # Save models
    with open(f"{output_dir}/model_treated.pkl", 'wb') as f:
        pickle.dump(model_treated, f)
    
    with open(f"{output_dir}/model_control.pkl", 'wb') as f:
        pickle.dump(model_control, f)
    
    print(f"✓ Models trained and saved")
    print(f"  - Model Treated: {model_treated.n_estimators} trees")
    print(f"  - Model Control: {model_control.n_estimators} trees")
    print(f"  - Features: {feature_cols}")
    
    return model_treated, model_control, feature_cols


def generate_predictions(daily_data, model_treated, model_control, feature_cols, output_dir='./data'):
    """Generate model predictions for all days"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nGenerating model predictions...")
    
    all_predictions = []
    
    for day_idx, df in enumerate(daily_data):
        X = df[feature_cols].fillna(df[feature_cols].median())
        
        # Get predictions from both models
        prob_treated = model_treated.predict_proba(X)[:, 1]
        prob_control = model_control.predict_proba(X)[:, 1]
        
        # Uplift score = P(success|treated) - P(success|control)
        # Success = NOT discontinuing, so flip the probabilities
        uplift_score = (1 - prob_treated) - (1 - prob_control)
        
        # Generate recommendation (top 20% by uplift)
        threshold = np.percentile(uplift_score, 80)
        recommendation = (uplift_score >= threshold).astype(int)
        
        predictions_df = pd.DataFrame({
            'patient_id': df['patient_id'],
            'date': df['date'],
            'model_treated_prob': prob_treated,
            'model_control_prob': prob_control,
            'uplift_score': uplift_score,
            'recommendation': recommendation
        })
        
        # Save predictions
        date_str = df['date'].iloc[0].strftime('%Y%m%d')
        filename = f"{output_dir}/predictions_{date_str}.parquet"
        predictions_df.to_parquet(filename, index=False)
        
        all_predictions.append(predictions_df)
        print(f"  Day {day_idx+1:2d}: {len(predictions_df):,} predictions, {recommendation.sum():,} recommended")
    
    # Save baseline predictions
    baseline_predictions = pd.concat(all_predictions[:14], ignore_index=True)
    baseline_predictions.to_parquet(f"{output_dir}/baseline_predictions.parquet", index=False)
    
    print(f"\n✓ Predictions generated for {len(all_predictions)} days")
    
    return all_predictions


if __name__ == "__main__":
    print("="*80)
    print("MONITORING DEMO - DATA GENERATION")
    print("="*80)
    
    # Generate feature data
    print("\n1. Generating patient feature data...")
    daily_data = generate_30_days_data(output_dir='./data')
    
    # Load baseline
    baseline_df = pd.read_parquet('./data/baseline_features.parquet')
    
    # Train models
    print("\n2. Training Random Forest models...")
    model_treated, model_control, feature_cols = train_sample_model(
        baseline_df,
        output_dir='./models'
    )
    
    # Generate predictions
    print("\n3. Generating model predictions...")
    predictions = generate_predictions(
        daily_data,
        model_treated,
        model_control,
        feature_cols,
        output_dir='./data'
    )
    
    print("\n" + "="*80)
    print("✓ DATA GENERATION COMPLETE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  - ./data/patient_features_*.parquet (30 files)")
    print(f"  - ./data/predictions_*.parquet (30 files)")
    print(f"  - ./data/baseline_features.parquet")
    print(f"  - ./data/baseline_predictions.parquet")
    print(f"  - ./models/model_treated.pkl")
    print(f"  - ./models/model_control.pkl")
    print(f"\nYou can now run the monitoring notebooks!")

