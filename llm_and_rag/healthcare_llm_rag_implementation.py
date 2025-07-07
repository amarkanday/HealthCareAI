"""
Healthcare LLM and RAG Implementation
Demonstrates various use cases for LLM and RAG in healthcare settings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class HealthcareLLMRAG:
    """
    Main class for healthcare LLM and RAG applications
    """
    
    def __init__(self, knowledge_base_path: str = None):
        """
        Initialize the healthcare LLM and RAG system
        
        Args:
            knowledge_base_path: Path to medical knowledge base
        """
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.clinical_guidelines = self._load_clinical_guidelines()
        self.drug_database = self._load_drug_database()
        self.coding_guidelines = self._load_coding_guidelines()
        
    def _load_knowledge_base(self, path: str) -> Dict:
        """Load medical knowledge base"""
        # In practice, this would load from a real medical database
        return {
            'conditions': {
                'diabetes': {
                    'symptoms': ['frequent urination', 'increased thirst', 'fatigue'],
                    'treatments': ['metformin', 'insulin', 'lifestyle changes'],
                    'guidelines': 'ADA 2024 guidelines for diabetes management'
                },
                'hypertension': {
                    'symptoms': ['headache', 'shortness of breath', 'chest pain'],
                    'treatments': ['ACE inhibitors', 'beta blockers', 'lifestyle changes'],
                    'guidelines': 'JNC 8 guidelines for hypertension management'
                }
            },
            'medications': {
                'metformin': {
                    'indications': ['diabetes type 2'],
                    'contraindications': ['kidney disease', 'heart failure'],
                    'side_effects': ['nausea', 'diarrhea', 'lactic acidosis']
                }
            }
        }
    
    def _load_clinical_guidelines(self) -> Dict:
        """Load clinical practice guidelines"""
        return {
            'diabetes_screening': 'Screen adults 45+ or high risk',
            'hypertension_treatment': 'Target BP <130/80 for most adults',
            'medication_reconciliation': 'Perform at every transition of care'
        }
    
    def _load_drug_database(self) -> Dict:
        """Load drug interaction database"""
        return {
            'metformin': {
                'interactions': ['alcohol', 'contrast_dye'],
                'monitoring': ['kidney_function', 'lactic_acid']
            },
            'warfarin': {
                'interactions': ['aspirin', 'vitamin_k', 'many_antibiotics'],
                'monitoring': ['INR', 'bleeding_signs']
            }
        }
    
    def _load_coding_guidelines(self) -> Dict:
        """Load medical coding guidelines"""
        return {
            'diabetes': {
                'ICD10': ['E11.9', 'E11.65', 'E11.22'],
                'documentation_requirements': ['HbA1c', 'complications', 'medications']
            },
            'hypertension': {
                'ICD10': ['I10', 'I11.9', 'I12.9'],
                'documentation_requirements': ['BP_readings', 'target_organ_damage']
            }
        }

class ClinicalDocumentationSystem(HealthcareLLMRAG):
    """
    Clinical documentation and summarization system
    """
    
    def summarize_clinical_notes(self, raw_notes: str, patient_context: Dict) -> Dict:
        """
        Summarize clinical notes using RAG-enhanced LLM
        
        Args:
            raw_notes: Raw clinical documentation
            patient_context: Patient demographics and history
            
        Returns:
            Structured summary with key findings
        """
        # Retrieve relevant medical guidelines
        relevant_guidelines = self._retrieve_relevant_guidelines(raw_notes)
        
        # Generate structured summary
        summary = {
            'chief_complaint': self._extract_chief_complaint(raw_notes),
            'assessment': self._generate_assessment(raw_notes, relevant_guidelines),
            'plan': self._generate_treatment_plan(raw_notes, patient_context),
            'medications': self._extract_medications(raw_notes),
            'follow_up': self._generate_follow_up_plan(patient_context),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _retrieve_relevant_guidelines(self, notes: str) -> List[str]:
        """Retrieve relevant clinical guidelines based on note content"""
        guidelines = []
        
        if 'diabetes' in notes.lower():
            guidelines.append(self.clinical_guidelines['diabetes_screening'])
        
        if 'hypertension' in notes.lower():
            guidelines.append(self.clinical_guidelines['hypertension_treatment'])
        
        return guidelines
    
    def _extract_chief_complaint(self, notes: str) -> str:
        """Extract chief complaint from clinical notes"""
        # In practice, this would use NLP to identify chief complaint
        complaints = ['chest pain', 'shortness of breath', 'fatigue', 'headache']
        for complaint in complaints:
            if complaint in notes.lower():
                return complaint
        return "General evaluation"
    
    def _generate_assessment(self, notes: str, guidelines: List[str]) -> str:
        """Generate clinical assessment based on notes and guidelines"""
        assessment = "Based on clinical evaluation"
        
        if guidelines:
            assessment += f" and following {', '.join(guidelines)}"
        
        return assessment
    
    def _generate_treatment_plan(self, notes: str, patient_context: Dict) -> List[str]:
        """Generate treatment plan based on clinical findings"""
        plan = []
        
        if 'diabetes' in notes.lower():
            plan.extend([
                "Continue metformin as prescribed",
                "Monitor blood glucose levels",
                "Lifestyle modifications: diet and exercise"
            ])
        
        if 'hypertension' in notes.lower():
            plan.extend([
                "Continue antihypertensive medications",
                "Monitor blood pressure at home",
                "Reduce sodium intake"
            ])
        
        return plan
    
    def _extract_medications(self, notes: str) -> List[str]:
        """Extract medications from clinical notes"""
        medications = []
        
        # Simple extraction - in practice would use more sophisticated NLP
        med_keywords = ['metformin', 'insulin', 'lisinopril', 'amlodipine']
        for med in med_keywords:
            if med in notes.lower():
                medications.append(med)
        
        return medications
    
    def _generate_follow_up_plan(self, patient_context: Dict) -> str:
        """Generate follow-up plan based on patient context"""
        age = patient_context.get('age', 50)
        
        if age > 65:
            return "Follow up in 2 weeks with comprehensive geriatric assessment"
        else:
            return "Follow up in 4 weeks for routine care"

class ClinicalDecisionSupport(HealthcareLLMRAG):
    """
    Clinical decision support system
    """
    
    def generate_treatment_recommendations(self, patient_data: Dict, clinical_question: str) -> Dict:
        """
        Generate evidence-based treatment recommendations
        
        Args:
            patient_data: Patient demographics, history, and current status
            clinical_question: Specific clinical question or concern
            
        Returns:
            Treatment recommendations with supporting evidence
        """
        # Retrieve relevant clinical evidence
        evidence = self._retrieve_clinical_evidence(patient_data, clinical_question)
        
        # Generate recommendations
        recommendations = {
            'primary_recommendation': self._generate_primary_recommendation(patient_data, evidence),
            'alternative_options': self._generate_alternatives(patient_data, evidence),
            'supporting_evidence': evidence,
            'risk_considerations': self._assess_risks(patient_data),
            'monitoring_plan': self._generate_monitoring_plan(patient_data),
            'confidence_score': self._calculate_confidence(patient_data, evidence)
        }
        
        return recommendations
    
    def _retrieve_clinical_evidence(self, patient_data: Dict, question: str) -> List[str]:
        """Retrieve relevant clinical evidence"""
        evidence = []
        
        if 'diabetes' in question.lower():
            evidence.append("ADA 2024 guidelines recommend metformin as first-line therapy")
            evidence.append("HbA1c target <7% for most adults with diabetes")
        
        if 'hypertension' in question.lower():
            evidence.append("JNC 8 guidelines recommend target BP <130/80")
            evidence.append("ACE inhibitors preferred for patients with diabetes")
        
        return evidence
    
    def _generate_primary_recommendation(self, patient_data: Dict, evidence: List[str]) -> str:
        """Generate primary treatment recommendation"""
        age = patient_data.get('age', 50)
        conditions = patient_data.get('conditions', [])
        
        if 'diabetes' in conditions:
            return "Start metformin 500mg twice daily with meals"
        elif 'hypertension' in conditions:
            return "Start lisinopril 10mg daily"
        else:
            return "Continue current treatment plan with close monitoring"
    
    def _generate_alternatives(self, patient_data: Dict, evidence: List[str]) -> List[str]:
        """Generate alternative treatment options"""
        alternatives = []
        
        if 'diabetes' in patient_data.get('conditions', []):
            alternatives.extend([
                "Sulfonylurea (glipizide) if metformin contraindicated",
                "DPP-4 inhibitor (sitagliptin) for additional glycemic control",
                "GLP-1 receptor agonist for weight loss benefits"
            ])
        
        return alternatives
    
    def _assess_risks(self, patient_data: Dict) -> List[str]:
        """Assess treatment risks based on patient factors"""
        risks = []
        age = patient_data.get('age', 50)
        conditions = patient_data.get('conditions', [])
        
        if age > 65:
            risks.append("Increased risk of medication side effects")
        
        if 'kidney_disease' in conditions:
            risks.append("Dose adjustment may be required for kidney function")
        
        return risks
    
    def _generate_monitoring_plan(self, patient_data: Dict) -> List[str]:
        """Generate monitoring plan for treatment"""
        monitoring = []
        
        if 'diabetes' in patient_data.get('conditions', []):
            monitoring.extend([
                "Monitor blood glucose daily",
                "Check HbA1c every 3 months",
                "Monitor kidney function annually"
            ])
        
        return monitoring
    
    def _calculate_confidence(self, patient_data: Dict, evidence: List[str]) -> float:
        """Calculate confidence score for recommendations"""
        # Simple confidence calculation based on evidence strength
        base_confidence = 0.7
        evidence_bonus = len(evidence) * 0.05
        return min(0.95, base_confidence + evidence_bonus)

class DrugInteractionAnalyzer(HealthcareLLMRAG):
    """
    Drug interaction analysis system
    """
    
    def analyze_drug_interactions(self, medications: List[str], patient_data: Dict) -> Dict:
        """
        Analyze potential drug interactions
        
        Args:
            medications: List of current medications
            patient_data: Patient demographics and medical history
            
        Returns:
            Interaction analysis with safety recommendations
        """
        interactions = []
        warnings = []
        recommendations = []
        
        for i, med1 in enumerate(medications):
            for med2 in medications[i+1:]:
                interaction = self._check_interaction(med1, med2)
                if interaction:
                    interactions.append(interaction)
        
        # Check individual medication safety
        for med in medications:
            safety_check = self._check_medication_safety(med, patient_data)
            if safety_check['warnings']:
                warnings.extend(safety_check['warnings'])
            if safety_check['recommendations']:
                recommendations.extend(safety_check['recommendations'])
        
        return {
            'interactions': interactions,
            'warnings': warnings,
            'recommendations': recommendations,
            'risk_level': self._calculate_interaction_risk(interactions),
            'monitoring_required': self._determine_monitoring_needs(interactions)
        }
    
    def _check_interaction(self, med1: str, med2: str) -> Optional[Dict]:
        """Check for drug interaction between two medications"""
        # Simplified interaction checking - in practice would use comprehensive database
        interaction_pairs = {
            ('warfarin', 'aspirin'): {
                'severity': 'high',
                'description': 'Increased bleeding risk',
                'recommendation': 'Monitor INR closely, consider alternative'
            },
            ('metformin', 'alcohol'): {
                'severity': 'moderate',
                'description': 'Increased risk of lactic acidosis',
                'recommendation': 'Limit alcohol consumption'
            }
        }
        
        pair = tuple(sorted([med1.lower(), med2.lower()]))
        return interaction_pairs.get(pair)
    
    def _check_medication_safety(self, medication: str, patient_data: Dict) -> Dict:
        """Check medication safety for specific patient"""
        warnings = []
        recommendations = []
        
        age = patient_data.get('age', 50)
        conditions = patient_data.get('conditions', [])
        
        if medication.lower() == 'metformin' and 'kidney_disease' in conditions:
            warnings.append("Metformin contraindicated in severe kidney disease")
            recommendations.append("Consider alternative diabetes medication")
        
        if medication.lower() == 'warfarin' and age > 75:
            warnings.append("Increased bleeding risk in elderly patients")
            recommendations.append("Monitor INR more frequently")
        
        return {'warnings': warnings, 'recommendations': recommendations}
    
    def _calculate_interaction_risk(self, interactions: List[Dict]) -> str:
        """Calculate overall interaction risk level"""
        if not interactions:
            return 'low'
        
        high_risk = sum(1 for i in interactions if i['severity'] == 'high')
        moderate_risk = sum(1 for i in interactions if i['severity'] == 'moderate')
        
        if high_risk > 0:
            return 'high'
        elif moderate_risk > 0:
            return 'moderate'
        else:
            return 'low'
    
    def _determine_monitoring_needs(self, interactions: List[Dict]) -> List[str]:
        """Determine required monitoring based on interactions"""
        monitoring = []
        
        for interaction in interactions:
            if 'warfarin' in str(interaction):
                monitoring.append("Monitor INR weekly")
            if 'metformin' in str(interaction):
                monitoring.append("Monitor kidney function")
        
        return monitoring

class PatientEducationSystem(HealthcareLLMRAG):
    """
    Patient education and communication system
    """
    
    def generate_patient_education(self, diagnosis: str, patient_data: Dict) -> Dict:
        """
        Generate personalized patient education materials
        
        Args:
            diagnosis: Patient's diagnosis
            patient_data: Patient demographics and preferences
            
        Returns:
            Personalized educational content
        """
        # Retrieve relevant educational content
        educational_content = self._retrieve_educational_content(diagnosis, patient_data)
        
        # Generate personalized explanation
        explanation = self._generate_explanation(diagnosis, educational_content, patient_data)
        
        # Create action plan
        action_plan = self._create_action_plan(diagnosis, patient_data)
        
        return {
            'explanation': explanation,
            'action_plan': action_plan,
            'warning_signs': self._generate_warning_signs(diagnosis),
            'follow_up_instructions': self._generate_follow_up_instructions(patient_data),
            'resources': self._provide_additional_resources(diagnosis)
        }
    
    def _retrieve_educational_content(self, diagnosis: str, patient_data: Dict) -> Dict:
        """Retrieve educational content based on diagnosis and patient factors"""
        literacy_level = patient_data.get('education_level', 'high')
        language = patient_data.get('preferred_language', 'english')
        
        content = {
            'diabetes': {
                'high_literacy': "Diabetes is a chronic condition affecting blood sugar regulation...",
                'low_literacy': "Diabetes means your body has trouble controlling sugar in your blood...",
                'warning_signs': ['very thirsty', 'frequent urination', 'tired all the time'],
                'lifestyle_tips': ['eat healthy foods', 'exercise regularly', 'check blood sugar']
            },
            'hypertension': {
                'high_literacy': "Hypertension is elevated blood pressure that can damage organs...",
                'low_literacy': "High blood pressure means your heart works too hard...",
                'warning_signs': ['headache', 'chest pain', 'shortness of breath'],
                'lifestyle_tips': ['reduce salt', 'exercise', 'manage stress']
            }
        }
        
        return content.get(diagnosis.lower(), {})
    
    def _generate_explanation(self, diagnosis: str, content: Dict, patient_data: Dict) -> str:
        """Generate personalized explanation"""
        literacy_level = patient_data.get('education_level', 'high')
        
        if literacy_level == 'low':
            return content.get('low_literacy', f"You have {diagnosis}. Please follow your doctor's advice.")
        else:
            return content.get('high_literacy', f"Your diagnosis is {diagnosis}. This requires ongoing management.")
    
    def _create_action_plan(self, diagnosis: str, patient_data: Dict) -> List[str]:
        """Create personalized action plan"""
        plan = []
        
        if diagnosis.lower() == 'diabetes':
            plan.extend([
                "Check blood sugar as directed by your doctor",
                "Take medications exactly as prescribed",
                "Follow a healthy diet plan",
                "Exercise regularly",
                "Keep all follow-up appointments"
            ])
        
        return plan
    
    def _generate_warning_signs(self, diagnosis: str) -> List[str]:
        """Generate warning signs for patient to watch for"""
        warning_signs = {
            'diabetes': [
                "Very high or very low blood sugar",
                "Excessive thirst or urination",
                "Unexplained weight loss",
                "Blurred vision"
            ],
            'hypertension': [
                "Severe headache",
                "Chest pain",
                "Shortness of breath",
                "Vision changes"
            ]
        }
        
        return warning_signs.get(diagnosis.lower(), ["Contact your doctor if you feel unwell"])

class MedicalCodingSystem(HealthcareLLMRAG):
    """
    Medical coding and billing support system
    """
    
    def generate_medical_codes(self, clinical_documentation: str, patient_data: Dict) -> Dict:
        """
        Generate appropriate medical codes for billing
        
        Args:
            clinical_documentation: Clinical notes and documentation
            patient_data: Patient demographics and history
            
        Returns:
            Medical codes with supporting documentation
        """
        # Extract diagnoses and procedures
        diagnoses = self._extract_diagnoses(clinical_documentation)
        procedures = self._extract_procedures(clinical_documentation)
        
        # Generate appropriate codes
        icd_codes = self._generate_icd_codes(diagnoses, patient_data)
        cpt_codes = self._generate_cpt_codes(procedures)
        
        # Validate coding
        validation = self._validate_coding(icd_codes, cpt_codes, clinical_documentation)
        
        return {
            'icd_codes': icd_codes,
            'cpt_codes': cpt_codes,
            'validation': validation,
            'documentation_requirements': self._check_documentation_requirements(icd_codes),
            'billing_notes': self._generate_billing_notes(icd_codes, cpt_codes)
        }
    
    def _extract_diagnoses(self, documentation: str) -> List[str]:
        """Extract diagnoses from clinical documentation"""
        diagnoses = []
        
        # Simple extraction - in practice would use more sophisticated NLP
        diagnosis_keywords = ['diabetes', 'hypertension', 'heart failure', 'asthma']
        for diagnosis in diagnosis_keywords:
            if diagnosis in documentation.lower():
                diagnoses.append(diagnosis)
        
        return diagnoses
    
    def _extract_procedures(self, documentation: str) -> List[str]:
        """Extract procedures from clinical documentation"""
        procedures = []
        
        procedure_keywords = ['physical exam', 'blood draw', 'x-ray', 'consultation']
        for procedure in procedure_keywords:
            if procedure in documentation.lower():
                procedures.append(procedure)
        
        return procedures
    
    def _generate_icd_codes(self, diagnoses: List[str], patient_data: Dict) -> List[str]:
        """Generate ICD-10 codes for diagnoses"""
        icd_codes = []
        
        for diagnosis in diagnoses:
            if diagnosis.lower() == 'diabetes':
                icd_codes.append('E11.9')  # Type 2 diabetes without complications
            elif diagnosis.lower() == 'hypertension':
                icd_codes.append('I10')   # Essential hypertension
        
        return icd_codes
    
    def _generate_cpt_codes(self, procedures: List[str]) -> List[str]:
        """Generate CPT codes for procedures"""
        cpt_codes = []
        
        for procedure in procedures:
            if 'physical exam' in procedure.lower():
                cpt_codes.append('99213')  # Office visit, established patient
            elif 'consultation' in procedure.lower():
                cpt_codes.append('99242')  # Office consultation
        
        return cpt_codes
    
    def _validate_coding(self, icd_codes: List[str], cpt_codes: List[str], documentation: str) -> Dict:
        """Validate coding against documentation"""
        validation = {
            'valid': True,
            'warnings': [],
            'recommendations': []
        }
        
        if not icd_codes:
            validation['warnings'].append("No ICD codes generated - check documentation")
            validation['valid'] = False
        
        if not cpt_codes:
            validation['warnings'].append("No CPT codes generated - check documentation")
            validation['valid'] = False
        
        return validation

def main():
    """
    Demonstrate various healthcare LLM and RAG use cases
    """
    print("Healthcare LLM and RAG Use Cases Demonstration")
    print("=" * 50)
    
    # Initialize systems
    doc_system = ClinicalDocumentationSystem()
    decision_support = ClinicalDecisionSupport()
    drug_analyzer = DrugInteractionAnalyzer()
    education_system = PatientEducationSystem()
    coding_system = MedicalCodingSystem()
    
    # Example patient data
    patient_data = {
        'age': 65,
        'gender': 'female',
        'conditions': ['diabetes', 'hypertension'],
        'medications': ['metformin', 'lisinopril'],
        'education_level': 'high',
        'preferred_language': 'english'
    }
    
    # Example clinical notes
    clinical_notes = """
    Patient presents for follow-up of diabetes and hypertension. 
    Blood pressure 145/90, blood glucose 180. 
    Patient reports taking metformin and lisinopril as prescribed.
    Physical exam unremarkable. Continue current medications.
    """
    
    print("\n1. Clinical Documentation Summarization")
    print("-" * 40)
    summary = doc_system.summarize_clinical_notes(clinical_notes, patient_data)
    print(f"Summary: {json.dumps(summary, indent=2)}")
    
    print("\n2. Clinical Decision Support")
    print("-" * 40)
    recommendations = decision_support.generate_treatment_recommendations(
        patient_data, "How should I manage this patient's diabetes?"
    )
    print(f"Recommendations: {json.dumps(recommendations, indent=2)}")
    
    print("\n3. Drug Interaction Analysis")
    print("-" * 40)
    interactions = drug_analyzer.analyze_drug_interactions(
        patient_data['medications'], patient_data
    )
    print(f"Interactions: {json.dumps(interactions, indent=2)}")
    
    print("\n4. Patient Education")
    print("-" * 40)
    education = education_system.generate_patient_education('diabetes', patient_data)
    print(f"Education: {json.dumps(education, indent=2)}")
    
    print("\n5. Medical Coding")
    print("-" * 40)
    coding = coding_system.generate_medical_codes(clinical_notes, patient_data)
    print(f"Coding: {json.dumps(coding, indent=2)}")
    
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main() 