# Medical Code Feature Engineering for Health Insurance

## Overview

This documentation describes a comprehensive feature engineering approach for transforming ICD-10 diagnosis codes, CPT procedure codes, and NDC drug codes into actionable features for health insurance predictive modeling.

## Medical Code Types

### 1. ICD-10 (International Classification of Diseases, 10th Revision)
- **Purpose**: Standardized diagnosis coding
- **Format**: 3-7 alphanumeric characters (e.g., E11.9 for Type 2 diabetes)
- **Usage**: Claims processing, risk assessment, disease tracking

### 2. CPT (Current Procedural Terminology)
- **Purpose**: Medical procedure and service coding
- **Format**: 5-digit numeric codes (e.g., 99213 for office visit)
- **Usage**: Billing, utilization analysis, cost prediction

### 3. NDC (National Drug Code)
- **Purpose**: Pharmaceutical product identification
- **Format**: 10-11 digits in 3 segments (labeler-product-package)
- **Usage**: Pharmacy claims, medication tracking, adherence monitoring

## Feature Engineering Categories

### 1. ICD-10 Based Features

#### Chronic Condition Flags
Binary indicators for major chronic conditions:
```python
chronic_conditions = {
    'diabetes': ['E10', 'E11', 'E13'],
    'hypertension': ['I10', 'I11', 'I12', 'I13', 'I15'],
    'heart_disease': ['I20', 'I21', 'I22', 'I25', 'I50'],
    'copd': ['J44', 'J43'],
    'chronic_kidney': ['N18', 'N19'],
    'cancer': ['C00-C96'],
    'mental_health': ['F20', 'F31', 'F32', 'F33', 'F41'],
    'substance_abuse': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15']
}
```

#### Disease Complexity Metrics
- **Disease Complexity Score**: Number of unique ICD-10 chapters represented
- **Comorbidity Count**: Total number of chronic conditions present
- **Specific Complications**: Diabetic complications (E11.2-E11.9)

#### Risk Combinations
- **Mental-Chronic Combo**: Mental health + chronic physical condition
- **Diabetic Complications**: Diabetes with organ-specific complications
- **Multi-System Involvement**: Conditions affecting 3+ body systems

### 2. CPT Based Features

#### Procedure Categories
```python
cpt_categories = {
    'evaluation_management': (99201-99499),
    'anesthesia': (00100-01999),
    'surgery': (10000-69999),
    'radiology': (70000-79999),
    'pathology': (80000-89999),
    'medicine': (90000-99199),
    'emergency': [99281-99285],
    'preventive': [99381-99387]
}
```

#### Utilization Patterns
- **Emergency Visits**: Count of ED-related CPT codes
- **Preventive Care Compliance**: Annual wellness visits, screenings
- **Diagnostic Intensity**: Radiology + pathology procedure count
- **Surgical Complexity**: High-cost procedure indicators

#### High-Cost Procedure Flags
```python
high_cost_procedures = {
    '27447': 'knee_replacement',
    '27130': 'hip_replacement',
    '33533': 'cardiac_bypass',
    '43644': 'gastric_bypass',
    '22612': 'spine_fusion',
    '33405': 'aortic_valve_replacement'
}
```

### 3. NDC Based Features

#### Medication Classes
```python
drug_classes = {
    'diabetes_meds': ['metformin', 'insulin', 'glipizide', 'januvia'],
    'cardiovascular': ['lisinopril', 'metoprolol', 'atorvastatin', 'clopidogrel'],
    'mental_health': ['sertraline', 'fluoxetine', 'risperidone', 'lithium'],
    'pain_management': ['oxycodone', 'hydrocodone', 'morphine', 'fentanyl'],
    'specialty_biologics': ['humira', 'enbrel', 'remicade', 'stelara']
}
```

#### Pharmacy Utilization Metrics
- **Polypharmacy Indicators**: 5+ concurrent medications
- **Medication Adherence**: Refill gap analysis for chronic medications
- **Generic Usage Rate**: Cost-conscious prescribing patterns
- **Specialty Drug Usage**: High-cost biologic medications

#### Drug Risk Indicators
- **Opioid Usage Patterns**: Prescription count and duration
- **Drug Interaction Risk**: Based on number of active drug classes
- **Medication Complexity**: Unique medication count

### 4. Combined Features

#### Disease-Treatment Alignment
- **Diabetes-Medication Alignment**: Diabetic patients on appropriate medications
- **Cardiovascular Alignment**: Hypertension/heart disease with cardiovascular drugs
- **Treatment Intensity**: Combined measure of medications + procedures + diagnoses

#### Risk Stratification
```python
high_risk_member = (
    (has_diabetes & has_heart_disease) |
    (comorbidity_count >= 3) |
    (has_high_cost_procedure == 1) |
    (extreme_polypharmacy == 1)
)
```

#### Cost Drivers
```python
cost_driver_score = (
    complex_surgeries * 3 +
    specialty_drug_count * 2 +
    emergency_visits * 1.5 +
    comorbidity_count * 1
)
```

### 5. Temporal Features

#### Trend Analysis
- **Claim Frequency Trend**: Recent vs historical utilization
- **Seasonal Patterns**: Winter vs summer utilization
- **Care Continuity**: Regularity of visits

#### Time-Based Risk Indicators
- **Recent Diagnosis Count**: New conditions in last 90 days
- **Hospital Admission Frequency**: Acute events per year
- **Medication Changes**: New prescriptions in recent months

## Implementation Example

### Data Structure Requirements

#### Claims Data Format
| Field | Type | Description |
|-------|------|-------------|
| member_id | int | Unique member identifier |
| claim_date | datetime | Date of service |
| icd10_codes | string | Comma-separated diagnosis codes |
| cpt_codes | string | Comma-separated procedure codes |
| place_of_service | string | Service location code |

#### Pharmacy Data Format
| Field | Type | Description |
|-------|------|-------------|
| member_id | int | Unique member identifier |
| fill_date | datetime | Prescription fill date |
| drug_name | string | Medication name and strength |
| ndc_code | string | National Drug Code |
| days_supply | int | Days of medication supplied |
| generic_flag | int | 1=generic, 0=brand |
| drug_cost | float | Total drug cost |

### Feature Engineering Pipeline

```python
# 1. Initialize feature engineer
engineer = MedicalCodeFeatureEngineer()

# 2. Extract features from each code type
icd_features = engineer.extract_icd10_features(claims_df)
cpt_features = engineer.extract_cpt_features(claims_df)
ndc_features = engineer.extract_ndc_features(pharmacy_df)

# 3. Create combined features
combined_features = engineer.create_combined_features(
    icd_features, cpt_features, ndc_features
)

# 4. Add temporal features
temporal_features = engineer.create_temporal_features(claims_df)

# 5. Merge all features
final_features = combined_features.merge(
    temporal_features, on='member_id', how='left'
)
```

## Feature Descriptions

### Core Health Indicators

1. **Comorbidity Count**
   - Range: 0-8
   - Interpretation: Higher values indicate multiple chronic conditions
   - Use: Risk stratification, care management

2. **Disease Complexity**
   - Range: 1-22 (ICD-10 chapters)
   - Interpretation: Diversity of health conditions
   - Use: Care coordination needs

3. **Treatment Intensity**
   - Calculation: (medications + procedures + diagnoses) / 3
   - Interpretation: Overall healthcare utilization
   - Use: Resource allocation

### Utilization Metrics

1. **Emergency Visits (6 months)**
   - Range: 0+
   - Risk Threshold: >2 visits
   - Use: Acute care management

2. **Preventive Care Gap**
   - Calculation: comorbidities - preventive visits
   - Interpretation: Unmet preventive care needs
   - Use: Outreach programs

3. **Procedure Diversity**
   - Range: 0-8 (CPT categories)
   - Interpretation: Variety of medical services used
   - Use: Care complexity assessment

### Medication Management

1. **Polypharmacy Flag**
   - Binary: 1 if ≥5 medications
   - Risk: Drug interactions, adherence issues
   - Use: Medication therapy management

2. **Medication Adherence Score**
   - Range: 0-1
   - Calculation: Proportion of on-time refills
   - Use: Intervention targeting

3. **Specialty Drug Count**
   - Range: 0+
   - Cost Impact: High (biologics cost $1000s/month)
   - Use: Financial risk assessment

### Risk Scores

1. **High Risk Member Flag**
   - Binary indicator
   - Criteria: Multiple chronic conditions, high-cost procedures
   - Use: Intensive case management

2. **Cost Driver Score**
   - Weighted combination of cost factors
   - Higher weight for surgeries and specialty drugs
   - Use: Premium calculation, reserves

3. **Care Coordination Score**
   - Range: 0+
   - Factors: Drug classes, procedures, diagnoses
   - Use: Care team assignment

## Best Practices

### 1. Data Quality
- **Validate Codes**: Check for valid ICD-10/CPT/NDC formats
- **Handle Missing Data**: Use appropriate imputation strategies
- **Remove Duplicates**: Deduplicate claims for same service

### 2. Feature Selection
- **Avoid Redundancy**: Remove highly correlated features
- **Clinical Relevance**: Ensure features have medical meaning
- **Predictive Power**: Test feature importance

### 3. Privacy Considerations
- **De-identification**: Remove PHI before modeling
- **Aggregation**: Use counts rather than specific codes
- **Compliance**: Follow HIPAA guidelines

### 4. Model Integration
- **Standardization**: Scale features appropriately
- **Handling Imbalance**: Address rare conditions
- **Temporal Alignment**: Ensure proper time windows

## Business Applications

### 1. Risk Stratification
```python
risk_tiers = pd.cut(
    final_features['cost_driver_score'],
    bins=[0, 5, 15, 30, np.inf],
    labels=['Low', 'Medium', 'High', 'Very High']
)
```

### 2. Intervention Targeting
- **Diabetes Management**: Members with poor medication alignment
- **Emergency Diversion**: High ED utilizers
- **Specialty Drug Management**: Biologic therapy optimization

### 3. Premium Pricing
- Use cost driver scores for actuarial modeling
- Adjust for geographic and demographic factors
- Incorporate temporal trends

### 4. Quality Metrics
- **HEDIS Compliance**: Preventive care gaps
- **Medication Adherence**: PDC calculations
- **Care Coordination**: Multi-provider utilization

## Advanced Techniques

### 1. Hierarchical Condition Categories (HCC)
- Map ICD-10 codes to HCC categories
- Calculate RAF (Risk Adjustment Factor) scores
- Use for Medicare Advantage risk adjustment

### 2. Clinical Groupers
- Episode Treatment Groups (ETGs)
- Diagnosis Related Groups (DRGs)
- Adjusted Clinical Groups (ACGs)

### 3. Machine Learning Enhancements
- **Embedding**: Learn code representations
- **Sequence Modeling**: LSTM for temporal patterns
- **Graph Networks**: Model code relationships

### 4. Natural Language Processing
- Extract features from clinical notes
- Identify undercoded conditions
- Enhance risk prediction

## Validation and Monitoring

### 1. Feature Validation
- Clinical review by medical professionals
- Statistical validation (distributions, outliers)
- Temporal stability checks

### 2. Model Performance
- Predictive accuracy metrics
- Business impact measurement
- Fairness and bias assessment

### 3. Ongoing Monitoring
- Feature drift detection
- Code set updates (annual ICD/CPT updates)
- Regulatory compliance

## Conclusion

Effective feature engineering from medical codes is crucial for health insurance analytics. This framework provides:

1. **Comprehensive Coverage**: All major code types (ICD-10, CPT, NDC)
2. **Clinical Relevance**: Features aligned with medical knowledge
3. **Business Value**: Direct application to insurance use cases
4. **Scalability**: Efficient processing of large member populations

The combination of diagnosis, procedure, and medication data creates a rich feature set for predicting costs, identifying risks, and improving member outcomes.