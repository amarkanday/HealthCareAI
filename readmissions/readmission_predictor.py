"""
Hospital Readmission Prediction & Prevention Models

⚠️ DISCLAIMER: This code uses synthetic data for educational purposes only.
No real patient data, clinical information, or proprietary algorithms are used.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_readmission_data(n_patients=10000):
    """Generate synthetic hospital readmission data"""
    
    print(f"Generating {n_patients:,} synthetic patient records...")
    
    # Patient demographics
    ages = np.random.normal(68, 16, n_patients)
    ages = np.clip(ages, 18, 98).astype(int)
    
    genders = np.random.choice(['M', 'F'], n_patients, p=[0.52, 0.48])
    
    # Primary diagnoses with different readmission risks
    diagnoses = np.random.choice([
        'Heart Failure', 'COPD', 'Pneumonia', 'AMI', 'Stroke', 'Sepsis', 
        'Diabetes', 'Kidney Disease', 'Hip/Knee Replacement', 'Other'
    ], n_patients, p=[0.15, 0.12, 0.11, 0.08, 0.07, 0.08, 0.06, 0.05, 0.04, 0.24])
    
    # Length of stay
    length_of_stay = np.random.lognormal(1.5, 0.6)
    length_of_stay = np.clip(length_of_stay, 1, 30).astype(int)
    
    # Comorbidities
    has_diabetes = np.random.binomial(1, 0.28, n_patients)
    has_chf = np.random.binomial(1, 0.18, n_patients)
    has_copd = np.random.binomial(1, 0.15, n_patients)
    charlson_score = np.random.poisson(2.5, n_patients)
    
    # Social factors
    lives_alone = np.random.binomial(1, 0.25, n_patients)
    has_caregiver = np.random.binomial(1, 0.75, n_patients)
    
    # Previous utilization
    prev_admissions = np.random.poisson(1.2, n_patients)
    
    # Calculate readmission risk
    risk_score = (
        0.15 * (diagnoses == 'Heart Failure').astype(int) +
        0.12 * (diagnoses == 'COPD').astype(int) +
        0.10 * (ages > 75).astype(int) +
        0.08 * (charlson_score > 4).astype(int) +
        0.06 * lives_alone +
        0.05 * (prev_admissions > 2).astype(int)
    )
    
    risk_score += np.random.normal(0, 0.1, n_patients)
    risk_score = np.clip(risk_score, 0, 1)
    
    # 30-day readmission outcome
    readmitted_30day = np.random.binomial(1, risk_score)
    
    return pd.DataFrame({
        'age': ages,
        'gender': genders,
        'primary_diagnosis': diagnoses,
        'length_of_stay': length_of_stay,
        'has_diabetes': has_diabetes,
        'has_chf': has_chf,
        'has_copd': has_copd,
        'charlson_score': charlson_score,
        'lives_alone': lives_alone,
        'has_caregiver': has_caregiver,
        'prev_admissions': prev_admissions,
        'readmitted_30day': readmitted_30day,
        'risk_score': risk_score
    })

def train_readmission_models(df):
    """Train machine learning models for readmission prediction"""
    
    print("Training readmission prediction models...")
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_diagnosis = LabelEncoder()
    
    df['gender_encoded'] = le_gender.fit_transform(df['gender'])
    df['diagnosis_encoded'] = le_diagnosis.fit_transform(df['primary_diagnosis'])
    
    # Select features
    features = ['age', 'gender_encoded', 'diagnosis_encoded', 'length_of_stay',
               'has_diabetes', 'has_chf', 'has_copd', 'charlson_score',
               'lives_alone', 'has_caregiver', 'prev_admissions']
    
    X = df[features]
    y = df['readmitted_30day']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'model': model,
            'auc': auc,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"{name}: AUC = {auc:.3f}")
    
    return results

def visualize_results(results, df):
    """Create visualizations for model results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ROC Curves
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        axes[0, 0].plot(fpr, tpr, label=f"{name} (AUC: {result['auc']:.3f})")
    
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Readmission by diagnosis
    dx_rates = df.groupby('primary_diagnosis')['readmitted_30day'].mean().sort_values(ascending=False)
    axes[0, 1].bar(range(len(dx_rates)), dx_rates.values)
    axes[0, 1].set_xticks(range(len(dx_rates)))
    axes[0, 1].set_xticklabels(dx_rates.index, rotation=45)
    axes[0, 1].set_title('Readmission Rate by Diagnosis')
    axes[0, 1].set_ylabel('Readmission Rate')
    
    # Age distribution
    axes[1, 0].hist(df[df['readmitted_30day']==1]['age'], bins=20, alpha=0.7, label='Readmitted')
    axes[1, 0].hist(df[df['readmitted_30day']==0]['age'], bins=20, alpha=0.7, label='Not Readmitted')
    axes[1, 0].set_xlabel('Age')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Age Distribution')
    axes[1, 0].legend()
    
    # Risk score distribution
    axes[1, 1].hist(df['risk_score'], bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Risk Score')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Risk Score Distribution')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    
    print("Hospital Readmission Prediction System")
    print("⚠️ Using Synthetic Data for Educational Purposes Only")
    print("="*55)
    
    # Generate data
    df = generate_readmission_data(10000)
    print(f"Readmission rate: {df['readmitted_30day'].mean()*100:.1f}%")
    
    # Train models
    results = train_readmission_models(df)
    
    # Visualize results
    visualize_results(results, df)
    
    # Business impact
    baseline_readmissions = df['readmitted_30day'].sum()
    potential_prevented = baseline_readmissions * 0.25  # 25% reduction
    cost_savings = potential_prevented * 15200  # $15,200 per readmission
    
    print(f"\nBusiness Impact Analysis:")
    print(f"Current readmissions: {baseline_readmissions:,}")
    print(f"Potential prevented: {potential_prevented:.0f}")
    print(f"Estimated cost savings: ${cost_savings:,.0f}")
    
    print("\n✅ Analysis Complete!")

if __name__ == "__main__":
    main() 