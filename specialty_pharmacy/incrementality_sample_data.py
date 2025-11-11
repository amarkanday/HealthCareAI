"""
Sample Data Generator for Specialty Pharmacy Incrementality Study
Generates realistic synthetic data for testing causal inference methods
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

class IncrementalityDataGenerator:
    """Generate synthetic data for incrementality study"""
    
    def __init__(self, n_patients: int = 2000):
        self.n_patients = n_patients
        self.start_date = datetime(2023, 1, 1)
        
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate all required datasets for incrementality analysis"""
        
        print("🔬 Generating Synthetic Incrementality Study Data")
        print("=" * 60)
        
        # Generate patient timeline
        patients_df = self._generate_patient_timeline()
        
        # Generate risk scores and flags
        risk_scores_df = self._generate_risk_scores(patients_df)
        
        # Generate interventions (with confounding)
        interventions_df = self._generate_interventions(patients_df, risk_scores_df)
        
        # Generate outcomes (with treatment effect)
        outcomes_df = self._generate_outcomes(patients_df, risk_scores_df, interventions_df)
        
        # Generate covariates
        covariates_df = self._generate_covariates(patients_df)
        
        # Generate operational context
        operational_df = self._generate_operational_context(risk_scores_df)
        
        # Combine into analysis dataset
        analysis_df = self._combine_analysis_dataset(
            patients_df, risk_scores_df, interventions_df, 
            outcomes_df, covariates_df, operational_df
        )
        
        print(f"\n✅ Data Generation Complete")
        print(f"   Total Patients: {len(patients_df):,}")
        print(f"   Risk Score Records: {len(risk_scores_df):,}")
        print(f"   Interventions: {len(interventions_df):,}")
        print(f"   Treatment Rate: {(interventions_df.groupby('patient_id').size() > 0).sum() / len(patients_df):.1%}")
        
        return {
            'patients': patients_df,
            'risk_scores': risk_scores_df,
            'interventions': interventions_df,
            'outcomes': outcomes_df,
            'covariates': covariates_df,
            'operational': operational_df,
            'analysis': analysis_df
        }
    
    def _generate_patient_timeline(self) -> pd.DataFrame:
        """Generate patient demographics and index dates"""
        
        print("\n📋 Generating patient timeline...")
        
        patients = []
        for i in range(self.n_patients):
            # Demographics
            age = np.random.normal(55, 15)
            age = np.clip(age, 18, 85)
            
            gender = np.random.choice(['M', 'F'], p=[0.45, 0.55])
            
            # Specialty conditions
            specialty = np.random.choice([
                'Oncology', 'Rheumatology', 'MS/Neurology', 
                'Hepatitis', 'HIV', 'Dermatology'
            ], p=[0.30, 0.25, 0.15, 0.12, 0.10, 0.08])
            
            # Payer type
            payer = np.random.choice([
                'Commercial', 'Medicare', 'Medicaid', 'Cash'
            ], p=[0.55, 0.25, 0.15, 0.05])
            
            # Index date (when they start therapy)
            days_offset = np.random.randint(0, 365)
            index_date = self.start_date + timedelta(days=days_offset)
            
            # High-cost drugs
            high_cost = specialty in ['Oncology', 'MS/Neurology', 'Hepatitis']
            monthly_cost = np.random.normal(
                15000 if high_cost else 5000, 
                5000 if high_cost else 2000
            )
            monthly_cost = np.clip(monthly_cost, 1000, 50000)
            
            patients.append({
                'patient_id': f'PT_{i+1:05d}',
                'age': round(age, 1),
                'gender': gender,
                'specialty_condition': specialty,
                'payer_type': payer,
                'index_date': index_date,
                'monthly_cost': round(monthly_cost, 2)
            })
        
        return pd.DataFrame(patients)
    
    def _generate_risk_scores(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Generate ML risk scores over time"""
        
        print("📊 Generating risk scores...")
        
        risk_records = []
        
        for _, patient in patients_df.iterrows():
            # Each patient gets scored 1-4 times over 12 months
            n_scores = np.random.randint(1, 5)
            
            for j in range(n_scores):
                # Score dates after index
                days_after_index = np.random.randint(0, 365)
                score_date = patient['index_date'] + timedelta(days=days_after_index)
                
                # Base risk from patient characteristics
                base_risk = 0.3
                
                # Age effect
                if patient['age'] > 65:
                    base_risk += 0.15
                elif patient['age'] < 35:
                    base_risk += 0.10
                
                # Payer effect
                if patient['payer_type'] == 'Medicaid':
                    base_risk += 0.15
                elif patient['payer_type'] == 'Cash':
                    base_risk += 0.20
                
                # Specialty effect
                if patient['specialty_condition'] in ['Oncology', 'HIV']:
                    base_risk += 0.10
                
                # Time-varying component (risk increases over time)
                time_factor = (days_after_index / 365) * 0.15
                
                # Random noise
                noise = np.random.normal(0, 0.1)
                
                risk_score = np.clip(base_risk + time_factor + noise, 0, 1)
                
                # High-risk flag
                high_risk = risk_score >= 0.5
                
                risk_records.append({
                    'patient_id': patient['patient_id'],
                    'score_date': score_date,
                    'risk_score': round(risk_score, 3),
                    'high_risk_flag': high_risk,
                    'model_version': f'v{np.random.randint(1, 4)}.0',
                    'risk_decile': min(int(risk_score * 10) + 1, 10)
                })
        
        return pd.DataFrame(risk_records)
    
    def _generate_interventions(
        self, 
        patients_df: pd.DataFrame, 
        risk_scores_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate interventions with realistic selection bias"""
        
        print("📞 Generating interventions (with confounding)...")
        
        interventions = []
        
        # Get high-risk patients
        high_risk = risk_scores_df[risk_scores_df['high_risk_flag'] == True].copy()
        
        for _, score_record in high_risk.iterrows():
            patient = patients_df[
                patients_df['patient_id'] == score_record['patient_id']
            ].iloc[0]
            
            # Probability of intervention depends on:
            # 1. Risk score (higher = more likely)
            # 2. Payer (commercial = more likely)
            # 3. Specialty (oncology = more likely)
            # 4. Operational factors (capacity, day of week)
            
            intervention_prob = 0.3  # base probability
            
            # Risk score effect (confounding!)
            if score_record['risk_score'] > 0.7:
                intervention_prob += 0.30
            elif score_record['risk_score'] > 0.6:
                intervention_prob += 0.20
            elif score_record['risk_score'] > 0.5:
                intervention_prob += 0.10
            
            # Payer effect (confounding!)
            if patient['payer_type'] == 'Commercial':
                intervention_prob += 0.15
            elif patient['payer_type'] == 'Medicaid':
                intervention_prob -= 0.05
            
            # Specialty effect
            if patient['specialty_condition'] in ['Oncology', 'MS/Neurology']:
                intervention_prob += 0.10
            
            # Operational capacity (instrument candidate)
            day_of_week = score_record['score_date'].dayofweek
            if day_of_week in [0, 1]:  # Monday/Tuesday - high capacity
                intervention_prob += 0.15
            elif day_of_week in [4, 5]:  # Friday/Saturday - low capacity
                intervention_prob -= 0.10
            
            # Random capacity shock (instrument candidate)
            capacity_shock = np.random.normal(0, 0.1)
            intervention_prob += capacity_shock
            
            intervention_prob = np.clip(intervention_prob, 0, 0.95)
            
            # Decide if intervention happens
            if np.random.random() < intervention_prob:
                # Intervention happens within 0-7 days
                days_delay = np.random.randint(0, 8)
                intervention_date = score_record['score_date'] + timedelta(days=days_delay)
                
                # Multiple touches possible
                n_touches = np.random.choice([1, 2, 3, 4], p=[0.50, 0.30, 0.15, 0.05])
                
                for touch in range(n_touches):
                    touch_date = intervention_date + timedelta(days=touch * 7)
                    
                    interventions.append({
                        'patient_id': score_record['patient_id'],
                        'intervention_date': touch_date,
                        'score_date': score_record['score_date'],
                        'risk_score': score_record['risk_score'],
                        'channel': np.random.choice(
                            ['Phone', 'Text', 'Email', 'App'], 
                            p=[0.50, 0.25, 0.15, 0.10]
                        ),
                        'duration_minutes': np.random.exponential(8) if touch == 0 else np.random.exponential(5),
                        'success': np.random.choice([True, False], p=[0.7, 0.3]),
                        'frm_id': f'FRM_{np.random.randint(1, 21):03d}',
                        'touch_number': touch + 1,
                        'capacity_shock': capacity_shock,
                        'day_of_week': day_of_week
                    })
        
        return pd.DataFrame(interventions) if interventions else pd.DataFrame()
    
    def _generate_outcomes(
        self,
        patients_df: pd.DataFrame,
        risk_scores_df: pd.DataFrame,
        interventions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate outcomes with treatment effect"""
        
        print("🎯 Generating outcomes (with treatment effects)...")
        
        outcomes = []
        
        # Get patients with interventions
        treated_patients = set(interventions_df['patient_id'].unique()) if len(interventions_df) > 0 else set()
        
        for _, patient in patients_df.iterrows():
            # Get max risk score for this patient
            patient_scores = risk_scores_df[
                risk_scores_df['patient_id'] == patient['patient_id']
            ]
            
            if len(patient_scores) == 0:
                continue
            
            max_risk = patient_scores['risk_score'].max()
            
            # Was patient treated?
            treated = patient['patient_id'] in treated_patients
            
            # Calculate discontinuation probability
            # Base probability from risk score
            disc_prob = max_risk * 0.6  # High risk → higher discontinuation
            
            # Patient characteristics
            if patient['age'] > 70:
                disc_prob += 0.10
            if patient['payer_type'] in ['Medicaid', 'Cash']:
                disc_prob += 0.15
            
            # TRUE TREATMENT EFFECT (what we want to measure!)
            # Intervention reduces discontinuation by 15-25% (absolute)
            if treated:
                # Get intervention intensity
                patient_interventions = interventions_df[
                    interventions_df['patient_id'] == patient['patient_id']
                ]
                n_touches = len(patient_interventions)
                successful_touches = patient_interventions['success'].sum()
                
                # Effect increases with intensity
                treatment_effect = 0.12 + (n_touches * 0.03) + (successful_touches * 0.02)
                treatment_effect = min(treatment_effect, 0.30)  # Cap at 30%
                
                disc_prob -= treatment_effect
            
            disc_prob = np.clip(disc_prob, 0.05, 0.95)
            
            # Simulate discontinuation
            discontinued = np.random.random() < disc_prob
            
            if discontinued:
                # Time to discontinuation (days)
                ttd = int(np.random.exponential(90))
                ttd = int(np.clip(ttd, 1, 365))
                disc_date = patient['index_date'] + timedelta(days=ttd)
            else:
                ttd = 365  # Censored
                disc_date = None
            
            # 6-month persistence
            persistence_6mo = ttd >= 180
            
            # PDC (proportion of days covered)
            if discontinued:
                pdc = np.random.beta(2, 5) * 0.8  # Lower PDC if discontinued
            else:
                pdc = np.random.beta(8, 2) * 0.95 + 0.05  # Higher PDC if persistent
            
            # MPR (medication possession ratio)
            mpr = pdc + np.random.normal(0, 0.05)
            mpr = np.clip(mpr, 0, 1.2)
            
            # Number of refills
            n_refills = int((365 / 30) * pdc)
            
            outcomes.append({
                'patient_id': patient['patient_id'],
                'discontinued': discontinued,
                'discontinuation_date': disc_date,
                'time_to_discontinuation_days': ttd,
                'persistence_6mo': persistence_6mo,
                'persistence_12mo': ttd >= 365,
                'pdc': round(pdc, 3),
                'mpr': round(mpr, 3),
                'n_refills': n_refills,
                'treated': treated,
                'max_risk_score': round(max_risk, 3)
            })
        
        return pd.DataFrame(outcomes)
    
    def _generate_covariates(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional covariates"""
        
        print("📋 Generating covariates...")
        
        covariates = []
        
        for _, patient in patients_df.iterrows():
            # Prior adherence history
            prior_adherence = np.random.beta(6, 3)
            
            # Comorbidities (Charlson Comorbidity Index)
            cci = np.random.poisson(2)
            
            # SDOH proxies
            income_level = np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2])
            
            # Distance to pharmacy
            distance_miles = np.random.exponential(15)
            
            # Prescriber characteristics
            prescriber_id = f'PRX_{np.random.randint(1, 101):03d}'
            prescriber_experience = np.random.randint(1, 30)
            
            covariates.append({
                'patient_id': patient['patient_id'],
                'prior_adherence': round(prior_adherence, 3),
                'cci_score': cci,
                'income_level': income_level,
                'distance_to_pharmacy_miles': round(distance_miles, 1),
                'prescriber_id': prescriber_id,
                'prescriber_experience_years': prescriber_experience,
                'has_caregiver': np.random.choice([True, False], p=[0.4, 0.6]),
                'transportation_access': np.random.choice(['Good', 'Fair', 'Poor'], p=[0.6, 0.3, 0.1])
            })
        
        return pd.DataFrame(covariates)
    
    def _generate_operational_context(self, risk_scores_df: pd.DataFrame) -> pd.DataFrame:
        """Generate operational/capacity variables"""
        
        print("⚙️  Generating operational context...")
        
        operational = []
        
        # Create weekly capacity metrics
        dates = pd.date_range(
            risk_scores_df['score_date'].min(),
            risk_scores_df['score_date'].max(),
            freq='W'
        )
        
        for date in dates:
            # FRM capacity varies by week
            n_frms_available = np.random.randint(15, 21)
            avg_caseload = np.random.normal(30, 8)
            
            operational.append({
                'week_start': date,
                'n_frms_available': n_frms_available,
                'avg_caseload': round(avg_caseload, 1),
                'capacity_utilization': round(np.random.uniform(0.6, 0.95), 2),
                'region': np.random.choice(['Northeast', 'Southeast', 'Midwest', 'West'])
            })
        
        return pd.DataFrame(operational)
    
    def _combine_analysis_dataset(
        self,
        patients_df: pd.DataFrame,
        risk_scores_df: pd.DataFrame,
        interventions_df: pd.DataFrame,
        outcomes_df: pd.DataFrame,
        covariates_df: pd.DataFrame,
        operational_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Combine all datasets for analysis"""
        
        print("🔗 Combining datasets...")
        
        # Start with patients
        analysis = patients_df.copy()
        
        # Add first high-risk score as index_score
        high_risk_first = risk_scores_df[
            risk_scores_df['high_risk_flag'] == True
        ].groupby('patient_id').first().reset_index()
        
        analysis = analysis.merge(
            high_risk_first[['patient_id', 'score_date', 'risk_score', 'risk_decile']],
            on='patient_id',
            how='left'
        )
        analysis.rename(columns={'score_date': 'first_high_risk_date'}, inplace=True)
        
        # Add intervention flag and timing
        if len(interventions_df) > 0:
            intervention_summary = interventions_df.groupby('patient_id').agg({
                'intervention_date': 'min',
                'touch_number': 'max',
                'success': 'sum',
                'duration_minutes': 'sum'
            }).reset_index()
            
            intervention_summary.columns = [
                'patient_id', 'first_intervention_date', 
                'n_touches', 'n_successful_touches', 'total_duration_minutes'
            ]
            
            analysis = analysis.merge(intervention_summary, on='patient_id', how='left')
            analysis['treated'] = analysis['first_intervention_date'].notna().astype(int)
            analysis['n_touches'] = analysis['n_touches'].fillna(0)
            analysis['n_successful_touches'] = analysis['n_successful_touches'].fillna(0)
        else:
            analysis['treated'] = 0
            analysis['n_touches'] = 0
            analysis['n_successful_touches'] = 0
        
        # Add outcomes (drop 'treated' column if it exists to avoid conflict)
        if 'treated' in outcomes_df.columns:
            outcomes_df = outcomes_df.drop(columns=['treated'])
        analysis = analysis.merge(outcomes_df, on='patient_id', how='left')
        
        # Add covariates
        analysis = analysis.merge(covariates_df, on='patient_id', how='left')
        
        # Ensure treated column is properly filled
        if 'treated' not in analysis.columns:
            analysis['treated'] = 0
        analysis['treated'] = analysis['treated'].fillna(0).astype(int)
        
        return analysis


def main():
    """Generate and save sample datasets"""
    
    # Generate data
    generator = IncrementalityDataGenerator(n_patients=2000)
    datasets = generator.generate_complete_dataset()
    
    # Save datasets
    print("\n💾 Saving datasets...")
    for name, df in datasets.items():
        filename = f'incrementality_{name}_data.csv'
        df.to_csv(filename, index=False)
        print(f"   Saved: {filename} ({len(df):,} rows)")
    
    # Display summary statistics
    print("\n📊 Dataset Summary:")
    print("=" * 60)
    
    analysis_df = datasets['analysis']
    print(f"\nPatients: {len(analysis_df):,}")
    print(f"Treatment Rate: {analysis_df['treated'].mean():.1%}")
    print(f"6-Month Persistence: {analysis_df['persistence_6mo'].mean():.1%}")
    print(f"  - Treated: {analysis_df[analysis_df['treated']==1]['persistence_6mo'].mean():.1%}")
    print(f"  - Untreated: {analysis_df[analysis_df['treated']==0]['persistence_6mo'].mean():.1%}")
    print(f"\nAverage Risk Score: {analysis_df['risk_score'].mean():.3f}")
    print(f"Average PDC: {analysis_df['pdc'].mean():.3f}")
    
    return datasets


if __name__ == '__main__':
    datasets = main()

