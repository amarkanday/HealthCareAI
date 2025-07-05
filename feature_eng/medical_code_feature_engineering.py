import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

class MedicalCodeFeatureEngineer:
    """
    Feature engineering for ICD-10, CPT, and NDC codes in health insurance data
    """
    
    def __init__(self):
        # ICD-10 chronic condition mappings
        self.chronic_conditions = {
            'diabetes': ['E10', 'E11', 'E13'],
            'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15'],
            'heart_disease': ['I20', 'I21', 'I22', 'I25', 'I50'],
            'copd': ['J44', 'J43'],
            'chronic_kidney': ['N18', 'N19'],
            'cancer': ['C00-C96'],
            'mental_health': ['F20', 'F31', 'F32', 'F33', 'F41'],
            'substance_abuse': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15']
        }
        
        # CPT procedure categories
        self.cpt_categories = {
            'evaluation_management': range(99201, 99500),
            'anesthesia': range(100, 1999),
            'surgery': range(10000, 69999),
            'radiology': range(70000, 79999),
            'pathology': range(80000, 89999),
            'medicine': range(90000, 99199),
            'emergency': [99281, 99282, 99283, 99284, 99285],
            'preventive': [99381, 99382, 99383, 99384, 99385, 99386, 99387]
        }
        
        # High-cost procedure CPT codes
        self.high_cost_procedures = {
            '27447': 'knee_replacement',
            '27130': 'hip_replacement',
            '33533': 'cardiac_bypass',
            '43644': 'gastric_bypass',
            '22612': 'spine_fusion',
            '33405': 'aortic_valve_replacement'
        }
        
        # NDC therapeutic classes
        self.drug_classes = {
            'diabetes_meds': ['metformin', 'insulin', 'glipizide', 'januvia'],
            'cardiovascular': ['lisinopril', 'metoprolol', 'atorvastatin', 'clopidogrel'],
            'mental_health': ['sertraline', 'fluoxetine', 'risperidone', 'lithium'],
            'pain_management': ['oxycodone', 'hydrocodone', 'morphine', 'fentanyl'],
            'specialty_biologics': ['humira', 'enbrel', 'remicade', 'stelara']
        }
        
    def extract_icd10_features(self, claims_df):
        """
        Extract features from ICD-10 diagnosis codes
        """
        features = defaultdict(list)
        
        for member_id in claims_df['member_id'].unique():
            member_claims = claims_df[claims_df['member_id'] == member_id]
            
            # Get all unique ICD-10 codes for this member
            icd_codes = member_claims['icd10_codes'].dropna().str.split(',').sum()
            icd_codes = [code.strip() for code in icd_codes]
            
            # 1. Chronic condition flags
            chronic_flags = {}
            for condition, codes in self.chronic_conditions.items():
                has_condition = any(
                    any(icd.startswith(prefix) for prefix in codes) if isinstance(codes, list)
                    else (codes[0] <= icd[:3] <= codes[1]) if '-' in codes[0]
                    else icd.startswith(codes)
                    for icd in icd_codes
                )
                chronic_flags[f'has_{condition}'] = int(has_condition)
            
            # 2. Disease complexity score
            unique_chapters = set([icd[0] for icd in icd_codes if icd])
            disease_complexity = len(unique_chapters)
            
            # 3. Comorbidity count
            comorbidity_count = sum(chronic_flags.values())
            
            # 4. Specific high-risk combinations
            has_diabetic_complications = (
                chronic_flags.get('has_diabetes', 0) and 
                any(icd.startswith('E11.') and len(icd) > 4 for icd in icd_codes)
            )
            
            # 5. Mental health + chronic disease
            mental_chronic_combo = (
                chronic_flags.get('has_mental_health', 0) and 
                (chronic_flags.get('has_diabetes', 0) or chronic_flags.get('has_heart_disease', 0))
            )
            
            # 6. Diagnosis recency and frequency
            recent_diagnoses = member_claims[member_claims['claim_date'] >= 
                                           (datetime.now() - timedelta(days=90))]['icd10_codes'].count()
            
            # 7. Emergency diagnosis indicators
            emergency_diagnoses = sum(1 for icd in icd_codes if icd.startswith(('R', 'S', 'T')))
            
            # Store features
            features['member_id'].append(member_id)
            features['disease_complexity'].append(disease_complexity)
            features['comorbidity_count'].append(comorbidity_count)
            features['has_diabetic_complications'].append(int(has_diabetic_complications))
            features['mental_chronic_combo'].append(int(mental_chronic_combo))
            features['recent_diagnosis_count'].append(recent_diagnoses)
            features['emergency_diagnosis_count'].append(emergency_diagnoses)
            
            # Add chronic condition flags
            for flag, value in chronic_flags.items():
                features[flag].append(value)
        
        return pd.DataFrame(features)
    
    def extract_cpt_features(self, claims_df):
        """
        Extract features from CPT procedure codes
        """
        features = defaultdict(list)
        
        for member_id in claims_df['member_id'].unique():
            member_claims = claims_df[claims_df['member_id'] == member_id]
            
            # Get all CPT codes
            cpt_codes = member_claims['cpt_codes'].dropna().str.split(',').sum()
            cpt_codes = [code.strip() for code in cpt_codes]
            
            # 1. Procedure category counts
            category_counts = defaultdict(int)
            for cpt in cpt_codes:
                try:
                    cpt_num = int(cpt)
                    for category, code_range in self.cpt_categories.items():
                        if isinstance(code_range, range):
                            if cpt_num in code_range:
                                category_counts[category] += 1
                        elif cpt_num in code_range:
                            category_counts[category] += 1
                except:
                    continue
            
            # 2. High-cost procedure flags
            high_cost_procs = []
            for cpt in cpt_codes:
                if cpt in self.high_cost_procedures:
                    high_cost_procs.append(self.high_cost_procedures[cpt])
            
            # 3. Procedure diversity
            unique_categories = len([c for c in category_counts if category_counts[c] > 0])
            
            # 4. Emergency utilization
            emergency_visits = category_counts.get('emergency', 0)
            
            # 5. Preventive care compliance
            preventive_visits = category_counts.get('preventive', 0)
            
            # 6. Surgical complexity
            total_surgeries = category_counts.get('surgery', 0)
            complex_surgeries = len(high_cost_procs)
            
            # 7. Diagnostic intensity
            diagnostic_procedures = (category_counts.get('radiology', 0) + 
                                   category_counts.get('pathology', 0))
            
            # 8. Recent procedure count
            recent_procedures = member_claims[
                member_claims['claim_date'] >= (datetime.now() - timedelta(days=180))
            ]['cpt_codes'].count()
            
            # Store features
            features['member_id'].append(member_id)
            features['procedure_diversity'].append(unique_categories)
            features['emergency_visits_6mo'].append(emergency_visits)
            features['preventive_care_visits'].append(preventive_visits)
            features['total_surgeries'].append(total_surgeries)
            features['complex_surgeries'].append(complex_surgeries)
            features['diagnostic_intensity'].append(diagnostic_procedures)
            features['recent_procedure_count'].append(recent_procedures)
            features['has_high_cost_procedure'].append(int(len(high_cost_procs) > 0))
            
            # Add category counts
            for category in self.cpt_categories:
                features[f'cpt_{category}_count'].append(category_counts.get(category, 0))
        
        return pd.DataFrame(features)
    
    def extract_ndc_features(self, pharmacy_df):
        """
        Extract features from NDC drug codes
        """
        features = defaultdict(list)
        
        for member_id in pharmacy_df['member_id'].unique():
            member_drugs = pharmacy_df[pharmacy_df['member_id'] == member_id]
            
            # Get all drug names and NDC codes
            drug_names = member_drugs['drug_name'].dropna().str.lower().tolist()
            ndc_codes = member_drugs['ndc_code'].dropna().tolist()
            
            # 1. Therapeutic class flags and counts
            class_counts = defaultdict(int)
            for drug_class, drugs in self.drug_classes.items():
                matching_drugs = sum(1 for drug in drug_names 
                                   if any(med in drug for med in drugs))
                class_counts[drug_class] = matching_drugs
            
            # 2. Polypharmacy indicator
            unique_drugs = len(set(drug_names))
            polypharmacy = int(unique_drugs >= 5)
            extreme_polypharmacy = int(unique_drugs >= 10)
            
            # 3. Medication adherence (refill patterns)
            if len(member_drugs) > 0:
                # Calculate days between refills for chronic medications
                chronic_meds = member_drugs[member_drugs['days_supply'] >= 30]
                if len(chronic_meds) > 1:
                    chronic_meds = chronic_meds.sort_values('fill_date')
                    refill_gaps = chronic_meds['fill_date'].diff().dt.days.dropna()
                    adherence_score = sum(refill_gaps <= 35) / len(refill_gaps) if len(refill_gaps) > 0 else 1
                else:
                    adherence_score = 1
            else:
                adherence_score = 0
            
            # 4. Specialty drug usage
            specialty_drug_count = class_counts.get('specialty_biologics', 0)
            has_specialty_drugs = int(specialty_drug_count > 0)
            
            # 5. Opioid usage patterns
            opioid_prescriptions = sum(1 for drug in drug_names 
                                     if any(opioid in drug for opioid in 
                                           ['oxycodone', 'hydrocodone', 'morphine', 'fentanyl']))
            
            # 6. Drug interaction risk
            # Simplified: risk increases with number of drug classes
            active_drug_classes = sum(1 for count in class_counts.values() if count > 0)
            interaction_risk_score = min(active_drug_classes / 3, 1)  # Normalize to 0-1
            
            # 7. Generic vs brand usage
            generic_count = sum(1 for drug in member_drugs['generic_flag'] if drug == 1)
            brand_count = len(member_drugs) - generic_count
            generic_usage_rate = generic_count / len(member_drugs) if len(member_drugs) > 0 else 0
            
            # 8. Medication cost indicators
            total_drug_cost = member_drugs['drug_cost'].sum()
            avg_drug_cost = member_drugs['drug_cost'].mean() if len(member_drugs) > 0 else 0
            
            # Store features
            features['member_id'].append(member_id)
            features['unique_medications'].append(unique_drugs)
            features['polypharmacy'].append(polypharmacy)
            features['extreme_polypharmacy'].append(extreme_polypharmacy)
            features['medication_adherence'].append(adherence_score)
            features['has_specialty_drugs'].append(has_specialty_drugs)
            features['specialty_drug_count'].append(specialty_drug_count)
            features['opioid_prescription_count'].append(opioid_prescriptions)
            features['drug_interaction_risk'].append(interaction_risk_score)
            features['generic_usage_rate'].append(generic_usage_rate)
            features['total_drug_cost'].append(total_drug_cost)
            features['avg_drug_cost'].append(avg_drug_cost)
            features['active_drug_classes'].append(active_drug_classes)
            
            # Add therapeutic class counts
            for drug_class in self.drug_classes:
                features[f'drugs_{drug_class}_count'].append(class_counts.get(drug_class, 0))
        
        return pd.DataFrame(features)
    
    def create_combined_features(self, icd_features, cpt_features, ndc_features):
        """
        Create interaction features between ICD, CPT, and NDC codes
        """
        # Merge all features
        combined = icd_features.merge(cpt_features, on='member_id', how='outer')
        combined = combined.merge(ndc_features, on='member_id', how='outer').fillna(0)
        
        # Create interaction features
        
        # 1. Disease-procedure alignment
        combined['diabetes_endocrine_alignment'] = (
            combined['has_diabetes'] * combined['cpt_medicine_count']
        )
        
        # 2. Medication-condition alignment
        combined['diabetes_medication_alignment'] = (
            combined['has_diabetes'] * combined['drugs_diabetes_meds_count']
        )
        
        combined['cardiovascular_alignment'] = (
            combined['has_hypertension'] * combined['drugs_cardiovascular_count']
        )
        
        # 3. Emergency utilization patterns
        combined['emergency_complexity'] = (
            combined['emergency_visits_6mo'] * combined['comorbidity_count']
        )
        
        # 4. Preventive care gap
        combined['preventive_care_gap'] = (
            combined['comorbidity_count'] - combined['preventive_care_visits']
        ).clip(lower=0)
        
        # 5. Treatment intensity score
        combined['treatment_intensity'] = (
            combined['unique_medications'] + 
            combined['procedure_diversity'] + 
            combined['disease_complexity']
        ) / 3
        
        # 6. High-risk indicators
        # Ensure boolean columns are properly typed
        combined['has_diabetes'] = combined['has_diabetes'].astype(bool)
        combined['has_heart_disease'] = combined['has_heart_disease'].astype(bool)
        combined['has_high_cost_procedure'] = combined['has_high_cost_procedure'].astype(bool)
        combined['extreme_polypharmacy'] = combined['extreme_polypharmacy'].astype(bool)
        
        combined['high_risk_member'] = (
            (combined['has_diabetes'] & combined['has_heart_disease']) |
            (combined['comorbidity_count'] >= 3) |
            (combined['has_high_cost_procedure']) |
            (combined['extreme_polypharmacy'])
        ).astype(int)
        
        # 7. Cost driver score
        combined['cost_driver_score'] = (
            combined['complex_surgeries'] * 3 +
            combined['specialty_drug_count'] * 2 +
            combined['emergency_visits_6mo'] * 1.5 +
            combined['comorbidity_count'] * 1
        )
        
        # 8. Care coordination need
        combined['care_coordination_score'] = (
            combined['active_drug_classes'] * 0.3 +
            combined['procedure_diversity'] * 0.3 +
            combined['disease_complexity'] * 0.4
        )
        
        # 9. Adherence-outcome risk
        combined['adherence_risk'] = (
            (1 - combined['medication_adherence']) * combined['comorbidity_count']
        )
        
        return combined
    
    def create_temporal_features(self, claims_df, lookback_days=365):
        """
        Create time-based features from medical codes
        """
        features = defaultdict(list)
        
        for member_id in claims_df['member_id'].unique():
            member_claims = claims_df[claims_df['member_id'] == member_id].sort_values('claim_date')
            
            # 1. Trend features (comparing recent vs historical)
            cutoff_date = datetime.now() - timedelta(days=lookback_days/2)
            recent_claims = member_claims[member_claims['claim_date'] >= cutoff_date]
            historical_claims = member_claims[member_claims['claim_date'] < cutoff_date]
            
            # Claim frequency trend
            if len(historical_claims) > 0:
                recent_rate = len(recent_claims) / (lookback_days/2)
                historical_rate = len(historical_claims) / (lookback_days/2)
                claim_trend = (recent_rate - historical_rate) / (historical_rate + 0.001)
            else:
                claim_trend = 0
            
            # 2. Seasonality features
            claim_months = member_claims['claim_date'].dt.month.value_counts()
            winter_claims = claim_months.get(12, 0) + claim_months.get(1, 0) + claim_months.get(2, 0)
            summer_claims = claim_months.get(6, 0) + claim_months.get(7, 0) + claim_months.get(8, 0)
            
            # 3. Care continuity
            if len(member_claims) > 1:
                days_between_visits = member_claims['claim_date'].diff().dt.days.dropna()
                avg_days_between = days_between_visits.mean()
                visit_regularity = days_between_visits.std() / (avg_days_between + 1)
            else:
                avg_days_between = 365
                visit_regularity = 0
            
            # 4. Acute event indicators
            hospital_admissions = member_claims[
                member_claims['place_of_service'].isin(['21', '51'])  # Hospital codes
            ].shape[0]
            
            features['member_id'].append(member_id)
            features['claim_frequency_trend'].append(claim_trend)
            features['winter_utilization'].append(winter_claims)
            features['summer_utilization'].append(summer_claims)
            features['avg_days_between_visits'].append(avg_days_between)
            features['visit_regularity'].append(visit_regularity)
            features['hospital_admissions_year'].append(hospital_admissions)
        
        return pd.DataFrame(features)

def generate_sample_medical_data(n_members=1000):
    """
    Generate sample medical claims and pharmacy data for demonstration
    """
    np.random.seed(42)
    
    # Generate claims data
    claims_data = []
    pharmacy_data = []
    
    for member_id in range(n_members):
        # Member characteristics
        age = np.random.randint(18, 85)
        risk_level = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        
        # Number of claims based on risk
        n_claims = np.random.poisson(3 if risk_level == 'low' else 8 if risk_level == 'medium' else 15)
        
        for _ in range(n_claims):
            # Generate ICD-10 codes
            if risk_level == 'high':
                icd_codes = np.random.choice([
                    'E11.9,I10,I25.10',  # Diabetes, hypertension, heart disease
                    'J44.0,I10,E11.9',   # COPD, hypertension, diabetes
                    'N18.3,I10,E11.22'   # Kidney disease, hypertension, diabetic nephropathy
                ], p=[0.4, 0.3, 0.3])
            elif risk_level == 'medium':
                icd_codes = np.random.choice([
                    'I10',      # Hypertension only
                    'E11.9',    # Diabetes only
                    'I10,E11.9' # Both
                ], p=[0.4, 0.3, 0.3])
            else:
                icd_codes = np.random.choice([
                    'J06.9',    # Upper respiratory infection
                    'M79.3',    # Unspecified myalgia
                    'R05'       # Cough
                ], p=[0.4, 0.3, 0.3])
            
            # Generate CPT codes
            if 'emergency' in locals() and np.random.random() < 0.1:
                cpt_codes = '99284'  # Emergency visit
            elif risk_level == 'high' and np.random.random() < 0.05:
                cpt_codes = np.random.choice(['27447', '33533', '22612'])  # High-cost procedures
            else:
                cpt_codes = np.random.choice(['99213', '99214', '80053', '71020'])  # Common procedures
            
            claim_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            
            claims_data.append({
                'member_id': member_id,
                'claim_date': claim_date,
                'icd10_codes': icd_codes,
                'cpt_codes': cpt_codes,
                'place_of_service': np.random.choice(['11', '21', '51'], p=[0.8, 0.15, 0.05])
            })
        
        # Generate pharmacy data
        n_drugs = np.random.poisson(2 if risk_level == 'low' else 5 if risk_level == 'medium' else 8)
        
        for _ in range(n_drugs):
            if risk_level == 'high':
                drug_name = np.random.choice([
                    'metformin 1000mg', 'insulin glargine', 'lisinopril 20mg',
                    'atorvastatin 40mg', 'metoprolol 50mg'
                ])
            elif risk_level == 'medium':
                drug_name = np.random.choice([
                    'metformin 500mg', 'lisinopril 10mg', 'atorvastatin 20mg'
                ])
            else:
                drug_name = np.random.choice([
                    'ibuprofen 600mg', 'amoxicillin 500mg', 'omeprazole 20mg'
                ])
            
            fill_date = datetime.now() - timedelta(days=np.random.randint(0, 365))
            
            pharmacy_data.append({
                'member_id': member_id,
                'fill_date': fill_date,
                'drug_name': drug_name,
                'ndc_code': f"{np.random.randint(10000, 99999)}-{np.random.randint(100, 999)}-{np.random.randint(10, 99)}",
                'days_supply': 30 if 'metformin' in drug_name or 'lisinopril' in drug_name else 10,
                'generic_flag': 1 if np.random.random() < 0.7 else 0,
                'drug_cost': np.random.uniform(10, 500) if risk_level == 'high' else np.random.uniform(5, 100)
            })
    
    return pd.DataFrame(claims_data), pd.DataFrame(pharmacy_data)

# Example usage
if __name__ == "__main__":
    # Generate sample data
    print("Generating sample medical data...")
    claims_df, pharmacy_df = generate_sample_medical_data(n_members=500)
    
    # Initialize feature engineer
    engineer = MedicalCodeFeatureEngineer()
    
    # Extract features from each code type
    print("\nExtracting ICD-10 features...")
    icd_features = engineer.extract_icd10_features(claims_df)
    print(f"Created {len(icd_features.columns)} ICD-10 features")
    
    print("\nExtracting CPT features...")
    cpt_features = engineer.extract_cpt_features(claims_df)
    print(f"Created {len(cpt_features.columns)} CPT features")
    
    print("\nExtracting NDC features...")
    ndc_features = engineer.extract_ndc_features(pharmacy_df)
    print(f"Created {len(ndc_features.columns)} NDC features")
    
    # Create combined features
    print("\nCreating combined features...")
    combined_features = engineer.create_combined_features(icd_features, cpt_features, ndc_features)
    print(f"Total features: {len(combined_features.columns)}")
    
    # Create temporal features
    print("\nCreating temporal features...")
    temporal_features = engineer.create_temporal_features(claims_df)
    
    # Merge all features
    final_features = combined_features.merge(temporal_features, on='member_id', how='left')
    
    # Display sample results
    print("\nSample feature values for first 5 members:")
    display_cols = ['member_id', 'comorbidity_count', 'high_risk_member', 
                   'cost_driver_score', 'polypharmacy', 'treatment_intensity']
    print(final_features[display_cols].head())
    
    # Feature statistics
    print("\nFeature Statistics:")
    print(f"High-risk members: {final_features['high_risk_member'].sum()} ({final_features['high_risk_member'].mean()*100:.1f}%)")
    print(f"Average comorbidity count: {final_features['comorbidity_count'].mean():.2f}")
    print(f"Members with polypharmacy: {final_features['polypharmacy'].sum()}")
    print(f"Average cost driver score: {final_features['cost_driver_score'].mean():.2f}")
    
    # Save features
    final_features.to_csv('member_features.csv', index=False)
    print("\nFeatures saved to 'member_features.csv'")
