#!/usr/bin/env python3
"""
Emergency Room Visit Prediction Model with Social Determinants of Health (SDOH)

This module predicts the likelihood of emergency room visits within 6 months
based on patient demographics, medical history, health indicators, and importantly,
social determinants of health (SDOH) and past ER visit history.

Key enhancements over basic medical models:
- Social determinants of health (income, housing, food security, etc.)
- Past ER visit history (strong predictor)
- Comprehensive risk scoring
- Actionable recommendations including social interventions

Author: Healthcare AI Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class ERVisitPredictor:
    """
    Emergency Room Visit Predictor with Social Determinants of Health
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=25)
        self.label_encoders = {}
        self.best_model = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic emergency room visit data with SDOH and past ER visits
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic patient data including SDOH
        """
        np.random.seed(42)
        
        # Basic demographics
        ages = np.random.normal(50, 20, n_samples).clip(18, 95).astype(int)
        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        
        # Health indicators
        bmi = np.random.normal(27, 6, n_samples).clip(15, 50)
        blood_pressure_systolic = np.random.normal(130, 20, n_samples).clip(90, 200).astype(int)
        blood_pressure_diastolic = np.random.normal(80, 15, n_samples).clip(60, 120).astype(int)
        heart_rate = np.random.normal(75, 15, n_samples).clip(50, 150).astype(int)
        
        # Medical conditions
        chronic_conditions = np.random.choice(
            ['None', 'Diabetes', 'Hypertension', 'Heart Disease', 'Diabetes,Hypertension',
             'COPD', 'Kidney Disease', 'Diabetes,Heart Disease'],
            n_samples,
            p=[0.4, 0.15, 0.15, 0.1, 0.1, 0.05, 0.03, 0.02]
        )
        
        medication_count = np.random.poisson(2, n_samples).clip(0, 15)
        recent_hospitalizations = np.random.poisson(0.3, n_samples).clip(0, 5)
        
        # Past ER visits (key predictor)
        past_er_visits = np.random.poisson(1.2, n_samples).clip(0, 15)
        past_er_visits_6m = np.random.poisson(0.5, n_samples).clip(0, 5)
        
        # Insurance type
        insurance_types = np.random.choice(
            ['Private', 'Medicare', 'Medicaid', 'Uninsured'],
            n_samples,
            p=[0.55, 0.25, 0.15, 0.05]
        )
        
        # Social Determinants of Health (SDOH)
        income_level = np.random.choice(
            ['Low', 'Medium', 'High'],
            n_samples,
            p=[0.3, 0.5, 0.2]
        )
        
        education_level = np.random.choice(
            ['Less than High School', 'High School', 'Some College', 'Bachelor\'s', 'Graduate'],
            n_samples,
            p=[0.15, 0.25, 0.3, 0.2, 0.1]
        )
        
        employment_status = np.random.choice(
            ['Employed', 'Unemployed', 'Retired', 'Disabled'],
            n_samples,
            p=[0.6, 0.1, 0.25, 0.05]
        )
        
        housing_status = np.random.choice(
            ['Own Home', 'Rent', 'Public Housing', 'Homeless'],
            n_samples,
            p=[0.65, 0.25, 0.08, 0.02]
        )
        
        transportation_access = np.random.choice(
            ['Personal Vehicle', 'Public Transport', 'No Reliable Transport'],
            n_samples,
            p=[0.75, 0.2, 0.05]
        )
        
        food_security = np.random.choice(
            ['Food Secure', 'Mildly Insecure', 'Moderately Insecure', 'Severely Insecure'],
            n_samples,
            p=[0.8, 0.1, 0.07, 0.03]
        )
        
        social_support = np.random.choice(
            ['Strong', 'Moderate', 'Weak', 'None'],
            n_samples,
            p=[0.4, 0.35, 0.2, 0.05]
        )
        
        neighborhood_safety = np.random.choice(
            ['Very Safe', 'Safe', 'Moderate', 'Unsafe'],
            n_samples,
            p=[0.3, 0.4, 0.25, 0.05]
        )
        
        healthcare_access = np.random.choice(
            ['Excellent', 'Good', 'Fair', 'Poor'],
            n_samples,
            p=[0.3, 0.4, 0.25, 0.05]
        )
        
        # Lifestyle factors
        smoking_status = np.random.choice(
            ['Never', 'Former', 'Current'],
            n_samples,
            p=[0.6, 0.25, 0.15]
        )
        
        exercise_frequency = np.random.choice(
            ['Regular', 'Occasional', 'Rare', 'Never'],
            n_samples,
            p=[0.3, 0.35, 0.25, 0.1]
        )
        
        stress_level = np.random.choice(
            ['Low', 'Medium', 'High'],
            n_samples,
            p=[0.3, 0.5, 0.2]
        )
        
        mental_health_status = np.random.choice(
            ['None', 'Mild', 'Moderate', 'Severe'],
            n_samples,
            p=[0.6, 0.25, 0.1, 0.05]
        )

        # Create target variable (ER visits in next 6 months)
        # Enhanced risk score including SDOH and past ER visits
        risk_score = (
            # Medical factors
            (ages > 65).astype(float) * 0.3 +
            (bmi > 30).astype(float) * 0.2 +
            (blood_pressure_systolic > 140).astype(float) * 0.25 +
            (blood_pressure_diastolic > 90).astype(float) * 0.25 +
            (heart_rate > 100).astype(float) * 0.15 +
            (chronic_conditions != 'None').astype(float) * 0.4 +
            (medication_count > 3).astype(float) * 0.2 +
            (recent_hospitalizations > 0).astype(float) * 0.5 +
            
            # Past ER visits (strong predictor)
            (past_er_visits > 0).astype(float) * 0.6 +
            (past_er_visits_6m > 0).astype(float) * 0.8 +
            
            # Social determinants of health
            (insurance_types == 'Uninsured').astype(float) * 0.4 +
            (income_level == 'Low').astype(float) * 0.3 +
            np.isin(education_level, ['Less than High School', 'High School']).astype(float) * 0.2 +
            (employment_status == 'Unemployed').astype(float) * 0.3 +
            np.isin(housing_status, ['Public Housing', 'Homeless']).astype(float) * 0.4 +
            (transportation_access == 'No Reliable Transport').astype(float) * 0.3 +
            np.isin(food_security, ['Moderately Insecure', 'Severely Insecure']).astype(float) * 0.3 +
            np.isin(social_support, ['Weak', 'None']).astype(float) * 0.25 +
            (neighborhood_safety == 'Unsafe').astype(float) * 0.3 +
            np.isin(healthcare_access, ['Fair', 'Poor']).astype(float) * 0.3 +
            
            # Lifestyle and mental health
            (smoking_status == 'Current').astype(float) * 0.25 +
            (exercise_frequency == 'Never').astype(float) * 0.15 +
            (stress_level == 'High').astype(float) * 0.2 +
            np.isin(mental_health_status, ['Moderate', 'Severe']).astype(float) * 0.3
        )
        
        # Add some randomness
        risk_score += np.random.normal(0, 0.1, n_samples)
        
        # Convert to binary outcome (ER visit or not)
        er_visit = (risk_score > np.percentile(risk_score, 75)).astype(int)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': ages,
            'gender': genders,
            'bmi': bmi,
            'blood_pressure_systolic': blood_pressure_systolic,
            'blood_pressure_diastolic': blood_pressure_diastolic,
            'heart_rate': heart_rate,
            'chronic_conditions': chronic_conditions,
            'medication_count': medication_count,
            'recent_hospitalizations': recent_hospitalizations,
            'past_er_visits': past_er_visits,
            'past_er_visits_6m': past_er_visits_6m,
            'insurance_type': insurance_types,
            'income_level': income_level,
            'education_level': education_level,
            'employment_status': employment_status,
            'housing_status': housing_status,
            'transportation_access': transportation_access,
            'food_security': food_security,
            'social_support': social_support,
            'neighborhood_safety': neighborhood_safety,
            'healthcare_access': healthcare_access,
            'smoking_status': smoking_status,
            'exercise_frequency': exercise_frequency,
            'stress_level': stress_level,
            'mental_health_status': mental_health_status,
            'er_visit': er_visit
        })
        
        return data
    
    def preprocess_data(self, data, predict_mode=False):
        """
        Preprocess the data for modeling with SDOH features
        
        Args:
            data: Raw DataFrame
            predict_mode: If True, don't expect 'er_visit' column (for prediction)
            
        Returns:
            Preprocessed features and target (or None if predict_mode=True)
        """
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values
        if len(df) > 1:
            df = df.fillna(df.mode().iloc[0])
        else:
            # For single row, fill with reasonable defaults
            defaults = {
                'age': 50, 'bmi': 25, 'blood_pressure_systolic': 120,
                'blood_pressure_diastolic': 80, 'heart_rate': 70,
                'medication_count': 1, 'recent_hospitalizations': 0,
                'past_er_visits': 0, 'past_er_visits_6m': 0
            }
            for col, default_val in defaults.items():
                if col in df.columns:
                    df[col] = df[col].fillna(default_val)
        
        # Feature engineering
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 30, 50, 65, 100], 
                                labels=['Young', 'Middle', 'Senior', 'Elderly'])
        
        df['bmi_category'] = pd.cut(df['bmi'], 
                                   bins=[0, 18.5, 25, 30, 100], 
                                   labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        
        df['blood_pressure_category'] = np.where(
            (df['blood_pressure_systolic'] >= 140) | (df['blood_pressure_diastolic'] >= 90),
            'High', 'Normal'
        )
        
        df['heart_rate_category'] = pd.cut(df['heart_rate'], 
                                         bins=[0, 60, 100, 200], 
                                         labels=['Low', 'Normal', 'High'])
        
        # Create risk scores
        df['chronic_condition_count'] = df['chronic_conditions'].apply(
            lambda x: len(x.split(',')) if x != 'None' else 0
        )
        
        df['high_risk_medications'] = (df['medication_count'] > 3).astype(int)
        
        # SDOH risk scores
        df['socioeconomic_risk'] = (
            (df['income_level'] == 'Low').astype(int) +
            (df['education_level'].isin(['Less than High School', 'High School'])).astype(int) +
            (df['employment_status'] == 'Unemployed').astype(int) +
            (df['housing_status'].isin(['Public Housing', 'Homeless'])).astype(int)
        )
        
        df['access_risk'] = (
            (df['transportation_access'] == 'No Reliable Transport').astype(int) +
            (df['healthcare_access'].isin(['Fair', 'Poor'])).astype(int) +
            (df['insurance_type'] == 'Uninsured').astype(int)
        )
        
        df['social_risk'] = (
            (df['food_security'].isin(['Moderately Insecure', 'Severely Insecure'])).astype(int) +
            (df['social_support'].isin(['Weak', 'None'])).astype(int) +
            (df['neighborhood_safety'] == 'Unsafe').astype(int)
        )
        
        # Mental health risk
        df['mental_health_risk'] = (
            (df['mental_health_status'].isin(['Moderate', 'Severe'])).astype(int) +
            (df['stress_level'] == 'High').astype(int)
        )
        
        # ER visit patterns
        df['frequent_er_user'] = (df['past_er_visits'] > 2).astype(int)
        df['recent_er_user'] = (df['past_er_visits_6m'] > 0).astype(int)
        
        # Encode categorical variables
        categorical_columns = [
            'gender', 'chronic_conditions', 'insurance_type', 'income_level',
            'education_level', 'employment_status', 'housing_status', 'transportation_access',
            'food_security', 'social_support', 'neighborhood_safety', 'healthcare_access',
            'smoking_status', 'exercise_frequency', 'stress_level', 'mental_health_status',
            'age_group', 'bmi_category', 'blood_pressure_category', 'heart_rate_category'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                if predict_mode and col in self.label_encoders:
                    # Use existing label encoder for prediction
                    le = self.label_encoders[col]
                    try:
                        df[col + '_encoded'] = le.transform(df[col].astype(str))
                    except ValueError:
                        # Handle unseen labels by using the most common value
                        df[col + '_encoded'] = 0
                else:
                    # Train new label encoder
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
        
        # Select features for modeling
        feature_columns = [
            # Medical features
            'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'heart_rate', 'medication_count', 'recent_hospitalizations',
            'chronic_condition_count', 'high_risk_medications',
            
            # Past ER visits (strong predictors)
            'past_er_visits', 'past_er_visits_6m', 'frequent_er_user', 'recent_er_user',
            
            # SDOH features
            'socioeconomic_risk', 'access_risk', 'social_risk', 'mental_health_risk',
            
            # Encoded categorical features
            'gender_encoded', 'insurance_type_encoded', 'income_level_encoded',
            'education_level_encoded', 'employment_status_encoded', 'housing_status_encoded',
            'transportation_access_encoded', 'food_security_encoded', 'social_support_encoded',
            'neighborhood_safety_encoded', 'healthcare_access_encoded', 'smoking_status_encoded',
            'exercise_frequency_encoded', 'stress_level_encoded', 'mental_health_status_encoded',
            'age_group_encoded', 'bmi_category_encoded', 'blood_pressure_category_encoded',
            'heart_rate_category_encoded'
        ]
        
        # Remove any columns that don't exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns]
        
        if predict_mode:
            return X, None
        else:
            y = df['er_visit']
            return X, y
    
    def train_models(self, X, y):
        """
        Train multiple models for ER visit prediction
        
        Args:
            X: Feature matrix
            y: Target variable
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_classif, k=25)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Calculate metrics
            accuracy = model.score(X_test_selected, y_test)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
            
            # Store results
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Store model
            self.models[name] = model
        
        # Set best model based on AUC
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]['model']
        
        return results, (X_test_selected, y_test)
    
    def evaluate_model(self, X_test, y_test, model_name='Best Model'):
        """
        Evaluate the best model with detailed metrics
        
        Args:
            X_test: Test features
            y_test: Test targets
            model_name: Name of model to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = self.best_model.score(X_test, y_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Sensitivity and specificity
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        print(f"\n{model_name} Detailed Evaluation:")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return {
            'accuracy': accuracy,
            'auc': auc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def plot_results(self, X_test, y_test):
        """
        Plot model evaluation results
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        # Make predictions for all models
        plt.figure(figsize=(15, 10))
        
        # ROC curves
        plt.subplot(2, 3, 1)
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid(True)
        
        # Feature importance (using best model if Random Forest or Gradient Boosting)
        if hasattr(self.best_model, 'feature_importances_'):
            plt.subplot(2, 3, 2)
            feature_names = [f'Feature_{i}' for i in range(len(self.best_model.feature_importances_))]
            importance = self.best_model.feature_importances_
            
            # Sort by importance
            indices = np.argsort(importance)[::-1][:10]  # Top 10 features
            
            plt.bar(range(len(indices)), importance[indices])
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.title('Top 10 Feature Importance')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        
        # Confusion matrix for best model
        plt.subplot(2, 3, 3)
        y_pred = self.best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No ER Visit', 'ER Visit'],
                   yticklabels=['No ER Visit', 'ER Visit'])
        plt.title('Confusion Matrix (Best Model)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def predict_er_visit(self, patient_data):
        """
        Predict ER visit probability for a new patient
        
        Args:
            patient_data: Dictionary with patient information including SDOH
            
        Returns:
            Dictionary with prediction results
        """
        # Convert patient data to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess the data for prediction
        X, _ = self.preprocess_data(df, predict_mode=True)
        
        # Scale and select features
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        # Make prediction
        probability = self.best_model.predict_proba(X_selected)[0, 1]
        prediction = self.best_model.predict(X_selected)[0]
        
        # Risk level classification
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'er_visit_probability': probability,
            'er_visit_prediction': bool(prediction),
            'risk_level': risk_level,
            'recommendations': self._generate_recommendations(patient_data, probability)
        }
    
    def _generate_recommendations(self, patient_data, probability):
        """
        Generate recommendations based on patient data and risk probability
        
        Args:
            patient_data: Patient information including SDOH
            probability: ER visit probability
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Medical recommendations
        if probability > 0.6:
            recommendations.extend([
                "Schedule follow-up appointment within 1 week",
                "Monitor vital signs daily",
                "Consider preventive care interventions",
                "Review medication compliance"
            ])
        elif probability > 0.3:
            recommendations.extend([
                "Schedule follow-up appointment within 2 weeks",
                "Monitor health indicators weekly",
                "Consider lifestyle modifications"
            ])
        else:
            recommendations.extend([
                "Continue routine care",
                "Schedule annual check-up",
                "Maintain healthy lifestyle"
            ])
        
        # SDOH-based recommendations
        if patient_data.get('past_er_visits', 0) > 2:
            recommendations.append("Consider care coordination program for frequent ER users")
        
        if patient_data.get('income_level') == 'Low':
            recommendations.append("Connect with social services for financial assistance")
        
        if patient_data.get('housing_status') in ['Public Housing', 'Homeless']:
            recommendations.append("Refer to housing assistance programs")
        
        if patient_data.get('transportation_access') == 'No Reliable Transport':
            recommendations.append("Arrange transportation assistance for medical appointments")
        
        if patient_data.get('food_security') in ['Moderately Insecure', 'Severely Insecure']:
            recommendations.append("Connect with food assistance programs")
        
        if patient_data.get('social_support') in ['Weak', 'None']:
            recommendations.append("Consider social work referral for support services")
        
        if patient_data.get('mental_health_status') in ['Moderate', 'Severe']:
            recommendations.append("Refer to mental health services")
        
        if patient_data.get('insurance_type') == 'Uninsured':
            recommendations.append("Assist with insurance enrollment")
        
        return recommendations

if __name__ == "__main__":
    # Simple test
    predictor = ERVisitPredictor()
    data = predictor.generate_synthetic_data(n_samples=1000)
    print(f"Generated {len(data)} rows with {len(data.columns)} columns")
    print("✓ ER visit predictor is working correctly!") 