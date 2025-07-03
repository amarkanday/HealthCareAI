"""
Clinical Outcome Prediction Models - Treatment Response Prediction

⚠️ DISCLAIMER: This code uses synthetic data for educational purposes only.
No real patient data, clinical information, or proprietary treatment protocols are used.

This implementation demonstrates machine learning approaches for predicting
clinical outcomes and treatment responses in healthcare settings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core machine learning libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
import shap

print("Clinical Outcome Prediction Models - Educational Demo")
print("Synthetic Data Only - No Real Patient Information")
print("="*60)

class ClinicalDataGenerator:
    """Generate synthetic clinical and treatment response data"""
    
    def __init__(self, n_patients: int = 800, random_state: int = 42):
        self.n_patients = n_patients
        np.random.seed(random_state)
    
    def generate_patient_data(self) -> pd.DataFrame:
        """Generate comprehensive synthetic patient dataset"""
        print(f"Generating synthetic clinical data for {self.n_patients} patients")
        
        # Demographics and basic characteristics
        data = {
            'patient_id': [f'PAT_{i:04d}' for i in range(1, self.n_patients + 1)],
            'age': np.random.normal(65, 15, self.n_patients).clip(18, 95).astype(int),
            'gender': np.random.choice(['Male', 'Female'], self.n_patients),
            'bmi': np.random.normal(28, 6, self.n_patients).clip(16, 50),
            'smoking_status': np.random.choice(['Never', 'Former', 'Current'], self.n_patients),
            'diabetes': np.random.choice([0, 1], self.n_patients, p=[0.75, 0.25]),
            'hypertension': np.random.choice([0, 1], self.n_patients, p=[0.6, 0.4]),
            'heart_disease': np.random.choice([0, 1], self.n_patients, p=[0.8, 0.2])
        }
        
        df = pd.DataFrame(data)
        
        # Disease characteristics
        df['disease_stage'] = np.random.choice(['I', 'II', 'III', 'IV'], 
                                             self.n_patients, p=[0.2, 0.3, 0.3, 0.2])
        df['tumor_size'] = np.random.gamma(2, 2, self.n_patients).clip(0.5, 15)
        df['histology_grade'] = np.random.choice(['Low', 'Intermediate', 'High'], 
                                               self.n_patients, p=[0.3, 0.4, 0.3])
        
        # Laboratory values
        df['hemoglobin'] = np.random.normal(12.5, 2.5, self.n_patients).clip(7, 18)
        df['white_blood_cells'] = np.random.normal(7.5, 3.0, self.n_patients).clip(2, 20)
        df['creatinine'] = np.random.normal(1.1, 0.4, self.n_patients).clip(0.5, 4.0)
        df['albumin'] = np.random.normal(3.8, 0.6, self.n_patients).clip(2.0, 5.0)
        
        # Biomarkers (synthetic)
        df['biomarker_a'] = np.random.lognormal(2, 1, self.n_patients).clip(0.1, 100)
        df['biomarker_b'] = np.random.normal(50, 25, self.n_patients).clip(0, 200)
        df['genetic_mutation'] = np.random.choice([0, 1], self.n_patients, p=[0.7, 0.3])
        df['protein_expression'] = np.random.choice(['Low', 'Medium', 'High'], 
                                                  self.n_patients, p=[0.3, 0.4, 0.3])
        
        # Performance status and treatment history
        df['ecog_performance_status'] = np.random.choice([0, 1, 2, 3], 
                                                        self.n_patients, p=[0.3, 0.4, 0.2, 0.1])
        df['prior_treatments'] = np.random.poisson(1.5, self.n_patients).clip(0, 6)
        df['treatment_naive'] = (df['prior_treatments'] == 0).astype(int)
        
        # Assign treatments and outcomes
        df = self._assign_treatments_and_outcomes(df)
        
        return df
    
    def _assign_treatments_and_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign treatments and generate realistic outcomes"""
        
        # Treatment options
        treatments = ['Chemotherapy A', 'Chemotherapy B', 'Targeted Therapy', 
                     'Immunotherapy', 'Combination Therapy']
        
        # Simple treatment assignment
        df['assigned_treatment'] = np.random.choice(treatments, self.n_patients)
        
        # Generate treatment response based on patient characteristics
        response_prob = self._calculate_response_probability(df)
        df['treatment_response'] = np.random.binomial(1, response_prob, self.n_patients)
        
        # Survival outcomes
        base_pfs = 8.0
        pfs_multiplier = np.where(df['treatment_response'] == 1, 2.0, 0.6)
        pfs_multiplier *= np.where(df['disease_stage'].isin(['III', 'IV']), 0.7, 1.3)
        df['progression_free_survival'] = np.random.exponential(base_pfs * pfs_multiplier)
        df['overall_survival'] = df['progression_free_survival'] + np.random.exponential(6)
        
        # Event indicators
        df['progression_event'] = np.random.binomial(1, 0.7, self.n_patients)
        df['death_event'] = np.random.binomial(1, 0.4, self.n_patients)
        
        # Quality of life
        qol_change = np.where(df['treatment_response'] == 1, 
                             np.random.normal(10, 5), 
                             np.random.normal(-5, 8))
        df['quality_of_life_change'] = qol_change
        
        return df
    
    def _calculate_response_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate treatment response probability based on patient factors"""
        
        base_response = 0.4
        response_prob = np.full(self.n_patients, base_response)
        
        # Treatment-specific effects
        treatment_effects = {
            'Chemotherapy A': 1.0, 'Chemotherapy B': 0.9, 'Targeted Therapy': 1.3,
            'Immunotherapy': 1.1, 'Combination Therapy': 1.4
        }
        
        for treatment, effect in treatment_effects.items():
            mask = df['assigned_treatment'] == treatment
            response_prob[mask] *= effect
        
        # Patient factor adjustments
        response_prob *= np.where(df['disease_stage'].isin(['I', 'II']), 1.4, 0.8)
        response_prob *= np.where(df['ecog_performance_status'] <= 1, 1.2, 0.9)
        response_prob *= np.where(df['genetic_mutation'] == 1, 1.3, 1.0)
        response_prob *= np.where(df['protein_expression'] == 'High', 1.2, 1.0)
        response_prob *= np.where(df['biomarker_a'] > df['biomarker_a'].median(), 1.15, 0.9)
        
        return np.clip(response_prob, 0.05, 0.95)

class TreatmentResponsePredictor:
    """Machine learning models for treatment response prediction"""
    
    def __init__(self):
        self.models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_performance = {}
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for treatment response prediction"""
        
        features = pd.DataFrame()
        
        # Demographics
        features['age'] = df['age']
        features['gender_male'] = (df['gender'] == 'Male').astype(int)
        features['bmi'] = df['bmi']
        
        # Clinical characteristics
        features['current_smoker'] = (df['smoking_status'] == 'Current').astype(int)
        features['diabetes'] = df['diabetes']
        features['hypertension'] = df['hypertension']
        features['heart_disease'] = df['heart_disease']
        
        # Disease characteristics
        features['stage_advanced'] = df['disease_stage'].isin(['III', 'IV']).astype(int)
        features['tumor_size'] = df['tumor_size']
        features['high_grade'] = (df['histology_grade'] == 'High').astype(int)
        
        # Laboratory values
        features['hemoglobin'] = df['hemoglobin']
        features['wbc_elevated'] = (df['white_blood_cells'] > 10).astype(int)
        features['creatinine_elevated'] = (df['creatinine'] > 1.3).astype(int)
        features['albumin_low'] = (df['albumin'] < 3.5).astype(int)
        
        # Biomarkers
        features['biomarker_a_log'] = np.log(df['biomarker_a'] + 1)
        features['biomarker_b'] = df['biomarker_b']
        features['genetic_mutation'] = df['genetic_mutation']
        features['protein_expression_high'] = (df['protein_expression'] == 'High').astype(int)
        
        # Performance and treatment history
        features['good_performance_status'] = (df['ecog_performance_status'] <= 1).astype(int)
        features['treatment_naive'] = df['treatment_naive']
        features['prior_treatment_count'] = df['prior_treatments']
        
        # Risk scores
        features['comorbidity_score'] = (features['diabetes'] + features['hypertension'] + 
                                       features['heart_disease'])
        
        self.feature_names = features.columns.tolist()
        return features
    
    def train_models(self, features: pd.DataFrame, target: pd.Series):
        """Train multiple models for treatment response prediction"""
        
        print("Training treatment response prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Training {model_name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=StratifiedKFold(n_splits=5), scoring='roc_auc')
            
            # Fit model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Performance metrics
            train_accuracy = accuracy_score(y_train, y_pred_train)
            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_auc = roc_auc_score(y_test, y_pred_proba)
            
            results[model_name] = {
                'cv_auc_mean': cv_scores.mean(),
                'cv_auc_std': cv_scores.std(),
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
                'test_auc': test_auc,
                'model': model
            }
            
            print(f"  CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            print(f"  Test AUC: {test_auc:.3f}")
        
        self.model_performance = results
        return results
    
    def predict_optimal_treatment(self, patient_data: pd.DataFrame):
        """Predict optimal treatment for individual patients"""
        
        treatments = ['Chemotherapy A', 'Chemotherapy B', 'Targeted Therapy', 
                     'Immunotherapy', 'Combination Therapy']
        
        # Prepare features
        features = self.prepare_features(patient_data)
        features_scaled = self.scaler.transform(features)
        
        # Get best model
        best_model_name = max(self.model_performance.keys(), 
                            key=lambda k: self.model_performance[k]['test_auc'])
        best_model = self.model_performance[best_model_name]['model']
        
        # Predict response probability
        base_prob = best_model.predict_proba(features_scaled)[0, 1]
        
        # Treatment-specific adjustments (simplified)
        treatment_predictions = {}
        for treatment in treatments:
            if treatment == 'Targeted Therapy' and patient_data['genetic_mutation'].iloc[0] == 1:
                adjusted_prob = min(base_prob * 1.3, 0.95)
            elif treatment == 'Immunotherapy' and patient_data['protein_expression'].iloc[0] == 'High':
                adjusted_prob = min(base_prob * 1.2, 0.95)
            elif treatment == 'Combination Therapy':
                adjusted_prob = min(base_prob * 1.15, 0.95)
            else:
                adjusted_prob = base_prob
            
            treatment_predictions[treatment] = adjusted_prob
        
        # Recommend best treatment
        recommended_treatment = max(treatment_predictions.items(), key=lambda x: x[1])
        
        return {
            'patient_id': patient_data['patient_id'].iloc[0],
            'recommended_treatment': recommended_treatment[0],
            'response_probability': recommended_treatment[1],
            'all_predictions': treatment_predictions
        }

class SurvivalAnalysisPredictor:
    """Survival analysis for progression-free survival and overall survival"""
    
    def __init__(self):
        self.cox_models = {}
        self.kaplan_meier = {}
    
    def fit_survival_models(self, df: pd.DataFrame) -> Dict:
        """Fit Cox proportional hazards models for survival endpoints"""
        
        print("Fitting survival analysis models...")
        
        # Prepare survival data
        survival_data = self._prepare_survival_data(df)
        
        # Cox model for progression-free survival
        print("Fitting PFS Cox model...")
        pfs_data = survival_data[['progression_free_survival', 'progression_event'] + 
                                list(survival_data.columns[2:-2])].copy()
        
        cph_pfs = CoxPHFitter()
        cph_pfs.fit(pfs_data, duration_col='progression_free_survival', 
                   event_col='progression_event')
        
        self.cox_models['pfs'] = cph_pfs
        
        # Cox model for overall survival
        print("Fitting OS Cox model...")
        os_data = survival_data[['overall_survival', 'death_event'] + 
                               list(survival_data.columns[2:-2])].copy()
        
        cph_os = CoxPHFitter()
        cph_os.fit(os_data, duration_col='overall_survival', 
                  event_col='death_event')
        
        self.cox_models['os'] = cph_os
        
        # Kaplan-Meier estimators by treatment
        self._fit_kaplan_meier_curves(df)
        
        results = {
            'pfs_concordance': cph_pfs.concordance_index_,
            'os_concordance': cph_os.concordance_index_,
            'pfs_coefficients': cph_pfs.summary,
            'os_coefficients': cph_os.summary
        }
        
        print(f"PFS model concordance: {cph_pfs.concordance_index_:.3f}")
        print(f"OS model concordance: {cph_os.concordance_index_:.3f}")
        
        return results
    
    def _prepare_survival_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for survival analysis"""
        
        survival_data = pd.DataFrame()
        
        # Time and event variables
        survival_data['progression_free_survival'] = df['progression_free_survival']
        survival_data['progression_event'] = df['progression_event']
        survival_data['overall_survival'] = df['overall_survival']
        survival_data['death_event'] = df['death_event']
        
        # Covariates
        survival_data['age'] = df['age']
        survival_data['gender_male'] = (df['gender'] == 'Male').astype(int)
        survival_data['stage_advanced'] = df['disease_stage'].isin(['III', 'IV']).astype(int)
        survival_data['tumor_size'] = df['tumor_size']
        survival_data['ecog_0_1'] = (df['ecog_performance_status'] <= 1).astype(int)
        survival_data['genetic_mutation'] = df['genetic_mutation']
        survival_data['treatment_response'] = df['treatment_response']
        survival_data['biomarker_a_log'] = np.log(df['biomarker_a'] + 1)
        
        # Treatment variables (one-hot encoded)
        treatment_dummies = pd.get_dummies(df['assigned_treatment'], prefix='treatment')
        survival_data = pd.concat([survival_data, treatment_dummies], axis=1)
        
        return survival_data
    
    def _fit_kaplan_meier_curves(self, df: pd.DataFrame):
        """Fit Kaplan-Meier survival curves by treatment"""
        
        print("Fitting Kaplan-Meier curves...")
        
        treatments = df['assigned_treatment'].unique()
        
        for outcome in ['progression_free_survival', 'overall_survival']:
            event_col = 'progression_event' if outcome == 'progression_free_survival' else 'death_event'
            
            km_curves = {}
            
            for treatment in treatments:
                treatment_data = df[df['assigned_treatment'] == treatment]
                
                kmf = KaplanMeierFitter()
                kmf.fit(treatment_data[outcome], 
                       event_observed=treatment_data[event_col], 
                       label=treatment)
                
                km_curves[treatment] = kmf
            
            self.kaplan_meier[outcome] = km_curves
    
    def predict_survival_probability(self, patient_data: pd.DataFrame, 
                                   time_points: List[float] = [6, 12, 24]) -> Dict:
        """Predict survival probabilities at specific time points"""
        
        survival_data = self._prepare_survival_data(patient_data)
        
        predictions = {}
        
        for endpoint in ['pfs', 'os']:
            cox_model = self.cox_models[endpoint]
            
            # Get survival function for patient
            survival_function = cox_model.predict_survival_function(survival_data.iloc[0])
            
            # Extract probabilities at time points
            endpoint_predictions = {}
            for time_point in time_points:
                if time_point in survival_function.index:
                    prob = survival_function.loc[time_point].iloc[0]
                else:
                    # Interpolate if exact time point not available
                    prob = np.interp(time_point, survival_function.index, 
                                   survival_function.iloc[:, 0])
                
                endpoint_predictions[f'{time_point}_month'] = prob
            
            predictions[endpoint] = endpoint_predictions
        
        return predictions

class ClinicalVisualizationDashboard:
    """Visualization tools for clinical prediction results"""
    
    def create_model_performance_plot(self, model_results):
        """Create model performance comparison visualization"""
        
        plt.figure(figsize=(15, 10))
        
        # Extract performance metrics
        models = list(model_results.keys())
        cv_auc = [model_results[model]['cv_auc_mean'] for model in models]
        test_auc = [model_results[model]['test_auc'] for model in models]
        test_acc = [model_results[model]['test_accuracy'] for model in models]
        
        # AUC comparison
        plt.subplot(2, 2, 1)
        x_pos = np.arange(len(models))
        plt.bar(x_pos, cv_auc, alpha=0.7, color='skyblue', label='CV AUC')
        plt.bar(x_pos, test_auc, alpha=0.7, color='lightcoral', label='Test AUC')
        plt.xlabel('Model')
        plt.ylabel('AUC Score')
        plt.title('Model Performance Comparison (AUC)')
        plt.xticks(x_pos, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Accuracy comparison
        plt.subplot(2, 2, 2)
        plt.bar(models, test_acc, color='lightgreen', alpha=0.7)
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Test Accuracy Comparison')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Performance scatter plot
        plt.subplot(2, 2, 3)
        plt.scatter(cv_auc, test_auc, s=100, alpha=0.7)
        for i, model in enumerate(models):
            plt.annotate(model, (cv_auc[i], test_auc[i]), xytext=(5, 5), 
                        textcoords='offset points')
        plt.xlabel('Cross-Validation AUC')
        plt.ylabel('Test AUC')
        plt.title('CV vs Test Performance')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # Model comparison summary
        plt.subplot(2, 2, 4)
        performance_df = pd.DataFrame({
            'Model': models,
            'CV_AUC': cv_auc,
            'Test_AUC': test_auc,
            'Test_Accuracy': test_acc
        })
        plt.axis('tight')
        plt.axis('off')
        table = plt.table(cellText=performance_df.round(3).values,
                         colLabels=performance_df.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        plt.title('Model Performance Summary')
        
        plt.tight_layout()
        plt.show()
    
    def create_treatment_analysis_plot(self, df, patient_predictions):
        """Create treatment analysis visualization"""
        
        plt.figure(figsize=(15, 10))
        
        # Treatment response rates by assigned treatment
        plt.subplot(2, 2, 1)
        treatment_response = df.groupby('assigned_treatment')['treatment_response'].agg(['mean', 'count'])
        plt.bar(treatment_response.index, treatment_response['mean'], 
                color='steelblue', alpha=0.7)
        plt.xlabel('Treatment')
        plt.ylabel('Response Rate')
        plt.title('Treatment Response Rates')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Survival by treatment response
        plt.subplot(2, 2, 2)
        responders = df[df['treatment_response'] == 1]['progression_free_survival']
        non_responders = df[df['treatment_response'] == 0]['progression_free_survival']
        plt.hist([responders, non_responders], bins=20, alpha=0.7, 
                label=['Responders', 'Non-responders'], color=['green', 'red'])
        plt.xlabel('Progression-Free Survival (months)')
        plt.ylabel('Frequency')
        plt.title('PFS by Treatment Response')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Biomarker analysis
        plt.subplot(2, 2, 3)
        high_biomarker = df[df['biomarker_a'] > df['biomarker_a'].median()]
        low_biomarker = df[df['biomarker_a'] <= df['biomarker_a'].median()]
        biomarker_response = [
            high_biomarker['treatment_response'].mean(),
            low_biomarker['treatment_response'].mean()
        ]
        plt.bar(['High Biomarker A', 'Low Biomarker A'], biomarker_response, 
                color=['orange', 'purple'], alpha=0.7)
        plt.ylabel('Response Rate')
        plt.title('Response by Biomarker Level')
        plt.grid(True, alpha=0.3)
        
        # Recommended treatments distribution
        plt.subplot(2, 2, 4)
        if patient_predictions:
            recommended_treatments = [pred['recommended_treatment'] for pred in patient_predictions]
            treatment_counts = pd.Series(recommended_treatments).value_counts()
            plt.pie(treatment_counts.values, labels=treatment_counts.index, autopct='%1.1f%%')
            plt.title('Recommended Treatment Distribution')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main execution function for clinical outcome prediction analysis"""
    
    print("\n🏥 Clinical Outcome Prediction Models")
    print("Treatment Response Prediction Demo")
    print("="*60)
    
    # Generate synthetic clinical data
    print("\n1️⃣ Generating synthetic clinical dataset...")
    data_generator = ClinicalDataGenerator(n_patients=800, random_state=42)
    df = data_generator.generate_patient_data()
    
    print(f"✅ Generated data for {len(df)} patients")
    print(f"   Treatment response rate: {df['treatment_response'].mean():.1%}")
    print(f"   Mean PFS: {df['progression_free_survival'].mean():.1f} months")
    print(f"   Mean OS: {df['overall_survival'].mean():.1f} months")
    
    # Treatment response prediction
    print("\n2️⃣ Training treatment response prediction models...")
    predictor = TreatmentResponsePredictor()
    features = predictor.prepare_features(df)
    target = df['treatment_response']
    
    model_results = predictor.train_models(features, target)
    
    print("\n📊 Model Performance Summary:")
    for model_name, results in model_results.items():
        print(f"   {model_name}:")
        print(f"     CV AUC: {results['cv_auc_mean']:.3f} (+/- {results['cv_auc_std']*2:.3f})")
        print(f"     Test AUC: {results['test_auc']:.3f}")
        print(f"     Test Accuracy: {results['test_accuracy']:.3f}")
    
    # Individual patient predictions
    print("\n3️⃣ Generating individual patient treatment recommendations...")
    
    # Select sample patients
    sample_patients = df.sample(n=5, random_state=42)
    patient_predictions = []
    
    for idx, patient in sample_patients.iterrows():
        patient_df = patient.to_frame().T
        prediction = predictor.predict_optimal_treatment(patient_df)
        patient_predictions.append(prediction)
        
        print(f"\nPatient {prediction['patient_id']}:")
        print(f"  Recommended Treatment: {prediction['recommended_treatment']}")
        print(f"  Response Probability: {prediction['response_probability']:.3f}")
    
    # Create visualizations
    print("\n4️⃣ Creating visualization dashboard...")
    dashboard = ClinicalVisualizationDashboard()
    
    # Model performance plots
    dashboard.create_model_performance_plot(model_results)
    
    # Treatment analysis
    dashboard.create_treatment_analysis_plot(df, patient_predictions)
    
    # Clinical insights
    print("\n5️⃣ Clinical Insights Summary")
    print("="*50)
    
    # Treatment effectiveness
    treatment_response = df.groupby('assigned_treatment')['treatment_response'].agg(['mean', 'count'])
    print(f"\n📈 Treatment Response Rates:")
    for treatment in treatment_response.index:
        rate = treatment_response.loc[treatment, 'mean']
        count = treatment_response.loc[treatment, 'count']
        print(f"   {treatment}: {rate:.1%} (n={count})")
    
    # Biomarker impact
    high_biomarker = df[df['biomarker_a'] > df['biomarker_a'].median()]
    low_biomarker = df[df['biomarker_a'] <= df['biomarker_a'].median()]
    print(f"\n🧬 Biomarker Impact:")
    print(f"   High biomarker A response rate: {high_biomarker['treatment_response'].mean():.1%}")
    print(f"   Low biomarker A response rate: {low_biomarker['treatment_response'].mean():.1%}")
    
    # Stage impact
    early_stage = df[df['disease_stage'].isin(['I', 'II'])]
    advanced_stage = df[df['disease_stage'].isin(['III', 'IV'])]
    print(f"\n🎯 Disease Stage Impact:")
    print(f"   Early stage (I-II) response rate: {early_stage['treatment_response'].mean():.1%}")
    print(f"   Advanced stage (III-IV) response rate: {advanced_stage['treatment_response'].mean():.1%}")
    
    # Best model
    best_model = max(model_results.keys(), key=lambda k: model_results[k]['test_auc'])
    print(f"\n🏆 Best Performing Model: {best_model}")
    print(f"   Test AUC: {model_results[best_model]['test_auc']:.3f}")
    
    print(f"\n🎉 Clinical Outcome Prediction Analysis Complete!")
    print("This demonstrates machine learning for treatment response prediction")
    print("and personalized medicine approaches using synthetic data.")

if __name__ == "__main__":
    main() 