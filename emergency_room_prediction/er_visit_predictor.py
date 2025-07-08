"""
Emergency Room Visit Prediction Model
Predicts likelihood of emergency room visits based on patient demographics, 
medical history, and health indicators.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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
    Emergency Room Visit Prediction Model
    """
    
    def __init__(self):
        """Initialize the ER visit predictor"""
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.best_model = None
        self.feature_importance = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic emergency room visit data
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic ER visit data
        """
        np.random.seed(42)
        
        # Patient demographics
        ages = np.random.normal(45, 20, n_samples)
        ages = np.clip(ages, 18, 95)
        
        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.48, 0.52])
        
        # Health indicators
        bmi = np.random.normal(26, 6, n_samples)
        bmi = np.clip(bmi, 16, 50)
        
        blood_pressure_systolic = np.random.normal(130, 20, n_samples)
        blood_pressure_systolic = np.clip(blood_pressure_systolic, 90, 200)
        
        blood_pressure_diastolic = np.random.normal(80, 12, n_samples)
        blood_pressure_diastolic = np.clip(blood_pressure_diastolic, 50, 120)
        
        heart_rate = np.random.normal(75, 15, n_samples)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # Medical history
        chronic_conditions = np.random.choice(
            ['None', 'Diabetes', 'Hypertension', 'Heart Disease', 'Asthma', 'Multiple'],
            n_samples, 
            p=[0.4, 0.2, 0.2, 0.1, 0.05, 0.05]
        )
        
        # Medication count
        medication_count = np.random.poisson(2, n_samples)
        medication_count = np.clip(medication_count, 0, 10)
        
        # Recent hospitalizations
        recent_hospitalizations = np.random.poisson(0.3, n_samples)
        recent_hospitalizations = np.clip(recent_hospitalizations, 0, 5)
        
        # Insurance type
        insurance_types = np.random.choice(
            ['Private', 'Medicare', 'Medicaid', 'Uninsured'],
            n_samples,
            p=[0.5, 0.25, 0.2, 0.05]
        )
        
        # Socioeconomic factors
        income_level = np.random.choice(
            ['Low', 'Medium', 'High'],
            n_samples,
            p=[0.3, 0.5, 0.2]
        )
        
        # Lifestyle factors
        smoking_status = np.random.choice(
            ['Never', 'Former', 'Current'],
            n_samples,
            p=[0.6, 0.25, 0.15]
        )
        
        exercise_frequency = np.random.choice(
            ['Never', 'Rarely', 'Sometimes', 'Regular'],
            n_samples,
            p=[0.2, 0.3, 0.3, 0.2]
        )
        
        # Mental health indicators
        stress_level = np.random.choice(
            ['Low', 'Medium', 'High'],
            n_samples,
            p=[0.4, 0.4, 0.2]
        )
        
        # Create target variable (ER visits in next 6 months)
        # Higher risk factors increase probability of ER visit
        risk_score = (
            (ages > 65) * 0.3 +
            (bmi > 30) * 0.2 +
            (blood_pressure_systolic > 140) * 0.25 +
            (blood_pressure_diastolic > 90) * 0.25 +
            (heart_rate > 100) * 0.15 +
            (chronic_conditions != 'None') * 0.4 +
            (medication_count > 3) * 0.2 +
            (recent_hospitalizations > 0) * 0.5 +
            (insurance_types == 'Uninsured') * 0.3 +
            (income_level == 'Low') * 0.2 +
            (smoking_status == 'Current') * 0.25 +
            (exercise_frequency == 'Never') * 0.15 +
            (stress_level == 'High') * 0.2
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
            'insurance_type': insurance_types,
            'income_level': income_level,
            'smoking_status': smoking_status,
            'exercise_frequency': exercise_frequency,
            'stress_level': stress_level,
            'er_visit': er_visit
        })
        
        return data
    
    def preprocess_data(self, data):
        """
        Preprocess the data for modeling
        
        Args:
            data: Raw DataFrame
            
        Returns:
            Preprocessed features and target
        """
        # Create copy to avoid modifying original data
        df = data.copy()
        
        # Handle missing values
        df = df.fillna(df.mode().iloc[0])
        
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
        
        # Encode categorical variables
        categorical_columns = [
            'gender', 'chronic_conditions', 'insurance_type', 'income_level',
            'smoking_status', 'exercise_frequency', 'stress_level', 'age_group',
            'bmi_category', 'blood_pressure_category', 'heart_rate_category'
        ]
        
        for col in categorical_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Select features for modeling
        feature_columns = [
            'age', 'bmi', 'blood_pressure_systolic', 'blood_pressure_diastolic',
            'heart_rate', 'medication_count', 'recent_hospitalizations',
            'gender_encoded', 'chronic_condition_count', 'high_risk_medications',
            'insurance_type_encoded', 'income_level_encoded', 'smoking_status_encoded',
            'exercise_frequency_encoded', 'stress_level_encoded', 'age_group_encoded',
            'bmi_category_encoded', 'blood_pressure_category_encoded', 'heart_rate_category_encoded'
        ]
        
        # Remove any columns that don't exist
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns]
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
        self.feature_selector = SelectKBest(score_func=f_classif, k=15)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Define models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        
        # Train models
        results = {}
        for name, model in models.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_selected)
            y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
            
            # Evaluate
            accuracy = model.score(X_test_selected, y_test)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
        self.best_model = results[best_model_name]['model']
        
        # Get feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_[0])
        
        self.models = results
        return results, (X_test_selected, y_test)
    
    def evaluate_model(self, X_test, y_test, model_name='Best Model'):
        """
        Evaluate the model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model to evaluate
        """
        if model_name == 'Best Model':
            model = self.best_model
        else:
            model = self.models[model_name]['model']
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n{model_name} Confusion Matrix:")
        print(cm)
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        print(f"\n{model_name} Additional Metrics:")
        print(f"Sensitivity (Recall): {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        
        return {
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def plot_results(self, X_test, y_test):
        """
        Create visualization plots for model results
        
        Args:
            X_test: Test features
            y_test: Test targets
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curves
        ax1 = axes[0, 0]
        for name, result in self.models.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            ax1.plot(fpr, tpr, label=f"{name} (AUC = {result['auc']:.3f})")
        
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curves')
        ax1.legend()
        ax1.grid(True)
        
        # 2. Feature Importance
        ax2 = axes[0, 1]
        if self.feature_importance is not None:
            feature_names = [f"Feature {i}" for i in range(len(self.feature_importance))]
            sorted_idx = np.argsort(self.feature_importance)[::-1]
            ax2.bar(range(len(sorted_idx)), self.feature_importance[sorted_idx])
            ax2.set_xlabel('Features')
            ax2.set_ylabel('Importance')
            ax2.set_title('Feature Importance')
            ax2.set_xticks(range(len(sorted_idx)))
            ax2.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45)
        
        # 3. Confusion Matrix Heatmap
        ax3 = axes[1, 0]
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['auc'])
        cm = confusion_matrix(y_test, self.models[best_model_name]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3)
        ax3.set_title(f'{best_model_name} Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Model Comparison
        ax4 = axes[1, 1]
        model_names = list(self.models.keys())
        accuracies = [self.models[name]['accuracy'] for name in model_names]
        aucs = [self.models[name]['auc'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        ax4.bar(x - width/2, accuracies, width, label='Accuracy')
        ax4.bar(x + width/2, aucs, width, label='AUC')
        
        ax4.set_xlabel('Models')
        ax4.set_ylabel('Score')
        ax4.set_title('Model Performance Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(model_names, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('emergency_room_prediction/model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def predict_er_visit(self, patient_data):
        """
        Predict ER visit probability for a new patient
        
        Args:
            patient_data: Dictionary with patient information
            
        Returns:
            Dictionary with prediction results
        """
        # Convert patient data to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Preprocess the data
        X, _ = self.preprocess_data(df)
        
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
            patient_data: Patient information
            probability: ER visit probability
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
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
        
        # Specific recommendations based on patient data
        if patient_data.get('age', 0) > 65:
            recommendations.append("Consider geriatric assessment")
        
        if patient_data.get('chronic_conditions', 'None') != 'None':
            recommendations.append("Ensure chronic condition management")
        
        if patient_data.get('medication_count', 0) > 3:
            recommendations.append("Review medication interactions")
        
        return recommendations

def main():
    """
    Main function to demonstrate ER visit prediction
    """
    print("Emergency Room Visit Prediction Model")
    print("=" * 50)
    
    # Initialize predictor
    predictor = ERVisitPredictor()
    
    # Generate synthetic data
    print("Generating synthetic emergency room visit data...")
    data = predictor.generate_synthetic_data(n_samples=10000)
    
    print(f"Dataset shape: {data.shape}")
    print(f"ER visit rate: {data['er_visit'].mean():.2%}")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, y = predictor.preprocess_data(data)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Train models
    print("\nTraining models...")
    results, (X_test, y_test) = predictor.train_models(X, y)
    
    # Evaluate best model
    print("\nEvaluating best model...")
    evaluation = predictor.evaluate_model(X_test, y_test)
    
    # Plot results
    print("\nGenerating plots...")
    predictor.plot_results(X_test, y_test)
    
    # Example predictions
    print("\nExample predictions:")
    
    # High-risk patient
    high_risk_patient = {
        'age': 75,
        'gender': 'Male',
        'bmi': 32,
        'blood_pressure_systolic': 160,
        'blood_pressure_diastolic': 95,
        'heart_rate': 110,
        'chronic_conditions': 'Diabetes,Hypertension',
        'medication_count': 5,
        'recent_hospitalizations': 2,
        'insurance_type': 'Medicare',
        'income_level': 'Low',
        'smoking_status': 'Current',
        'exercise_frequency': 'Never',
        'stress_level': 'High'
    }
    
    prediction = predictor.predict_er_visit(high_risk_patient)
    print(f"\nHigh-risk patient prediction:")
    print(f"ER visit probability: {prediction['er_visit_probability']:.2%}")
    print(f"Risk level: {prediction['risk_level']}")
    print(f"Recommendations: {prediction['recommendations']}")
    
    # Low-risk patient
    low_risk_patient = {
        'age': 35,
        'gender': 'Female',
        'bmi': 24,
        'blood_pressure_systolic': 120,
        'blood_pressure_diastolic': 80,
        'heart_rate': 70,
        'chronic_conditions': 'None',
        'medication_count': 1,
        'recent_hospitalizations': 0,
        'insurance_type': 'Private',
        'income_level': 'High',
        'smoking_status': 'Never',
        'exercise_frequency': 'Regular',
        'stress_level': 'Low'
    }
    
    prediction = predictor.predict_er_visit(low_risk_patient)
    print(f"\nLow-risk patient prediction:")
    print(f"ER visit probability: {prediction['er_visit_probability']:.2%}")
    print(f"Risk level: {prediction['risk_level']}")
    print(f"Recommendations: {prediction['recommendations']}")
    
    print("\nModel training and evaluation completed successfully!")

if __name__ == "__main__":
    main() 