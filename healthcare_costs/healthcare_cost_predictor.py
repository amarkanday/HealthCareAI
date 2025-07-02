"""
Healthcare Cost Prediction using Deep Learning (LSTM)

⚠️ DISCLAIMER: This code uses synthetic data for educational purposes only.
No real patient data or proprietary healthcare information is used.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthcareCostPredictor:
    """
    Deep Learning model for predicting healthcare costs using patient data
    """
    
    def __init__(self, sequence_length=12, lstm_units=128, dense_units=64):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_cost = StandardScaler()
        self.label_encoders = {}
        
    def preprocess_data(self, df):
        """
        Preprocess the healthcare data for model training
        """
        # Handle missing values
        df = df.fillna(method='ffill').fillna(0)
        
        # Encode categorical variables
        categorical_cols = ['gender', 'chronic_conditions', 'medication_type', 'insurance_type']
        for col in categorical_cols:
            if col in df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col + '_encoded'] = self.label_encoders[col].fit_transform(df[col])
                else:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Create temporal features
        if 'visit_date' in df.columns:
            df['visit_date'] = pd.to_datetime(df['visit_date'])
            df['days_since_last_visit'] = df.groupby('patient_id')['visit_date'].diff().dt.days.fillna(0)
            df['visit_month'] = df['visit_date'].dt.month
            df['visit_year'] = df['visit_date'].dt.year
        
        # Aggregate features by patient
        feature_cols = ['age', 'num_medications', 'num_procedures', 'lab_results_abnormal',
                       'emergency_visits', 'hospitalization_days', 'chronic_condition_count',
                       'days_since_last_visit', 'gender_encoded']
        
        # Add encoded categorical columns
        for col in categorical_cols:
            if col + '_encoded' in df.columns:
                feature_cols.append(col + '_encoded')
        
        # Filter existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        return df, feature_cols
    
    def create_sequences(self, data, patient_ids, target, sequence_length):
        """
        Create sequences for LSTM input
        """
        sequences = []
        targets = []
        
        for patient_id in patient_ids.unique():
            patient_data = data[patient_ids == patient_id]
            patient_target = target[patient_ids == patient_id]
            
            if len(patient_data) >= sequence_length:
                for i in range(len(patient_data) - sequence_length + 1):
                    sequences.append(patient_data[i:i + sequence_length])
                    targets.append(patient_target.iloc[i + sequence_length - 1])
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self, input_shape):
        """
        Build LSTM model for healthcare cost prediction
        """
        model = keras.Sequential([
            # LSTM layers
            layers.LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units // 2, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(self.lstm_units // 4),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(self.dense_units, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.dense_units // 2, activation='relu'),
            layers.Dense(1)  # Output layer for cost prediction
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the LSTM model
        """
        # Scale features
        X_train_scaled = self.scaler_features.fit_transform(X_train.reshape(-1, X_train.shape[-1]))
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        
        X_val_scaled = self.scaler_features.transform(X_val.reshape(-1, X_val.shape[-1]))
        X_val_scaled = X_val_scaled.reshape(X_val.shape)
        
        # Scale target
        y_train_scaled = self.scaler_cost.fit_transform(y_train.reshape(-1, 1))
        y_val_scaled = self.scaler_cost.transform(y_val.reshape(-1, 1))
        
        # Early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Model checkpoint
        checkpoint = keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True
        )
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        return history
    
    def predict(self, X_test):
        """
        Make predictions on test data
        """
        X_test_scaled = self.scaler_features.transform(X_test.reshape(-1, X_test.shape[-1]))
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        predictions_scaled = self.model.predict(X_test_scaled)
        predictions = self.scaler_cost.inverse_transform(predictions_scaled)
        
        return predictions.flatten()
    
    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance
        """
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }
        
        return metrics
    
    def plot_predictions(self, y_true, y_pred, save_path='predictions_plot.png'):
        """
        Plot actual vs predicted costs
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Cost ($)')
        plt.ylabel('Predicted Cost ($)')
        plt.title('Healthcare Cost Predictions: Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
    
    def identify_high_risk_patients(self, patient_ids, predictions, threshold_percentile=90):
        """
        Identify high-risk patients based on predicted costs
        """
        threshold = np.percentile(predictions, threshold_percentile)
        high_risk_mask = predictions >= threshold
        high_risk_patients = patient_ids[high_risk_mask]
        
        return high_risk_patients, predictions[high_risk_mask]

# Example usage function
def generate_sample_data(n_patients=1000, n_records_per_patient=24):
    """
    Generate sample healthcare data for demonstration
    """
    np.random.seed(42)
    data = []
    
    for patient_id in range(n_patients):
        # Patient characteristics
        age = np.random.randint(18, 85)
        gender = np.random.choice(['M', 'F'])
        chronic_conditions = np.random.choice(['None', 'Diabetes', 'Hypertension', 'Both'], 
                                            p=[0.4, 0.2, 0.2, 0.2])
        
        # Generate time series data
        base_date = pd.Timestamp('2022-01-01')
        for month in range(n_records_per_patient):
            visit_date = base_date + pd.DateOffset(months=month)
            
            # Generate features based on patient characteristics
            if chronic_conditions == 'Both':
                base_cost = 5000
                num_medications = np.random.poisson(5)
                emergency_visits = np.random.poisson(0.3)
            elif chronic_conditions in ['Diabetes', 'Hypertension']:
                base_cost = 3000
                num_medications = np.random.poisson(3)
                emergency_visits = np.random.poisson(0.2)
            else:
                base_cost = 1000
                num_medications = np.random.poisson(1)
                emergency_visits = np.random.poisson(0.1)
            
            # Add age factor
            age_factor = 1 + (age - 40) * 0.02
            
            # Generate healthcare cost with some randomness
            healthcare_cost = base_cost * age_factor + np.random.normal(0, 500)
            healthcare_cost = max(0, healthcare_cost)
            
            record = {
                'patient_id': patient_id,
                'visit_date': visit_date,
                'age': age + month // 12,  # Age increases over time
                'gender': gender,
                'chronic_conditions': chronic_conditions,
                'num_medications': num_medications,
                'num_procedures': np.random.poisson(0.5),
                'lab_results_abnormal': np.random.binomial(1, 0.3),
                'emergency_visits': emergency_visits,
                'hospitalization_days': np.random.poisson(0.1),
                'chronic_condition_count': chronic_conditions.count(',') + 1 if chronic_conditions != 'None' else 0,
                'healthcare_cost': healthcare_cost
            }
            
            data.append(record)
    
    return pd.DataFrame(data)

# Main execution
if __name__ == "__main__":
    # Generate sample data
    print("Generating sample healthcare data...")
    df = generate_sample_data(n_patients=500, n_records_per_patient=24)
    
    # Initialize predictor
    predictor = HealthcareCostPredictor(sequence_length=12, lstm_units=128, dense_units=64)
    
    # Preprocess data
    print("Preprocessing data...")
    df_processed, feature_cols = predictor.preprocess_data(df)
    
    # Prepare features and target
    features = df_processed[feature_cols].values
    target = df_processed['healthcare_cost']
    patient_ids = df_processed['patient_id'].values
    
    # Create sequences
    print("Creating sequences for LSTM...")
    X, y = predictor.create_sequences(features, patient_ids, target, predictor.sequence_length)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Build model
    print("Building LSTM model...")
    model = predictor.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.summary()
    
    # Train model
    print("Training model...")
    history = predictor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    # Make predictions
    print("Making predictions...")
    y_pred = predictor.predict(X_test)
    
    # Evaluate model
    metrics = predictor.evaluate(y_test, y_pred)
    print("\nModel Performance:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")
    
    # Plot predictions
    predictor.plot_predictions(y_test, y_pred)
    
    # Identify high-risk patients
    test_patient_ids = np.array([i for i in range(len(y_test))])  # Simplified for demo
    high_risk_patients, high_risk_costs = predictor.identify_high_risk_patients(
        test_patient_ids, y_pred, threshold_percentile=90
    )
    
    print(f"\nIdentified {len(high_risk_patients)} high-risk patients")
    print(f"Average predicted cost for high-risk patients: ${np.mean(high_risk_costs):.2f}") 