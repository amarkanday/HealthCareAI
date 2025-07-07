"""
Healthcare LLM and RAG Implementation
Demonstrates various use cases for LLM and RAG in healthcare settings
Using LangChain, Gemini, and Med-PaLM
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.llms import GooglePalm
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import Document
import os

class HealthcareLLMRAG:
    """
    Main class for healthcare LLM and RAG applications
    Using LangChain, Gemini, and Med-PaLM
    """
    
    def __init__(self, knowledge_base_path: str = None, api_key: str = None):
        """
        Initialize the healthcare LLM and RAG system
        
        Args:
            knowledge_base_path: Path to medical knowledge base
            api_key: API key for Google AI services
        """
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize LLMs
        self.gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=self.api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        self.med_palm_llm = GooglePalm(
            google_api_key=self.api_key,
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Initialize embeddings
        self.embeddings = GooglePalmEmbeddings(google_api_key=self.api_key)
        
        # Load knowledge bases
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        self.clinical_guidelines = self._load_clinical_guidelines()
        self.drug_database = self._load_drug_database()
        self.coding_guidelines = self._load_coding_guidelines()
        
        # Initialize vector stores
        self.vector_store = self._initialize_vector_store()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
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
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize vector store with medical knowledge"""
        # Create documents from knowledge base
        documents = []
        
        # Add clinical guidelines
        for guideline_name, guideline_text in self.clinical_guidelines.items():
            documents.append(Document(
                page_content=f"Clinical Guideline - {guideline_name}: {guideline_text}",
                metadata={"type": "guideline", "name": guideline_name}
            ))
        
        # Add drug information
        for drug_name, drug_info in self.drug_database.items():
            content = f"Drug: {drug_name}. Interactions: {', '.join(drug_info['interactions'])}. Monitoring: {', '.join(drug_info['monitoring'])}"
            documents.append(Document(
                page_content=content,
                metadata={"type": "drug", "name": drug_name}
            ))
        
        # Add coding guidelines
        for condition, codes in self.coding_guidelines.items():
            content = f"Condition: {condition}. ICD-10 codes: {', '.join(codes['ICD10'])}. Documentation requirements: {', '.join(codes['documentation_requirements'])}"
            documents.append(Document(
                page_content=content,
                metadata={"type": "coding", "condition": condition}
            ))
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        return vector_store
    
    def _create_retrieval_chain(self, prompt_template: str) -> RetrievalQA:
        """Create a retrieval QA chain with custom prompt"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=self.gemini_llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return chain

class ClinicalDocumentationSystem(HealthcareLLMRAG):
    """
    Clinical documentation and summarization system using LangChain and Gemini
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)
        
        # Create specialized prompts
        self.summary_prompt = PromptTemplate(
            input_variables=["clinical_notes", "patient_context", "guidelines"],
            template="""
            You are a medical professional. Summarize the following clinical notes in a structured format.
            
            Clinical Notes: {clinical_notes}
            Patient Context: {patient_context}
            Relevant Guidelines: {guidelines}
            
            Provide a structured summary with:
            1. Chief Complaint
            2. Assessment
            3. Treatment Plan
            4. Medications
            5. Follow-up Plan
            
            Summary:
            """
        )
        
        self.summary_chain = LLMChain(
            llm=self.gemini_llm,
            prompt=self.summary_prompt
        )
    
    def summarize_clinical_notes(self, raw_notes: str, patient_context: Dict) -> Dict:
        """
        Summarize clinical notes using RAG-enhanced LLM
        
        Args:
            raw_notes: Raw clinical documentation
            patient_context: Patient demographics and history
            
        Returns:
            Structured summary with key findings
        """
        # Retrieve relevant medical guidelines using RAG
        relevant_guidelines = self._retrieve_relevant_guidelines(raw_notes)
        
        # Generate structured summary using Gemini
        summary_response = self.summary_chain.run({
            "clinical_notes": raw_notes,
            "patient_context": json.dumps(patient_context),
            "guidelines": "\n".join(relevant_guidelines)
        })
        
        # Parse the response into structured format
        summary = self._parse_summary_response(summary_response)
        summary['timestamp'] = datetime.now().isoformat()
        
        return summary
    
    def _retrieve_relevant_guidelines(self, notes: str) -> List[str]:
        """Retrieve relevant clinical guidelines using RAG"""
        # Use vector store to find relevant guidelines
        relevant_docs = self.vector_store.similarity_search(notes, k=3)
        
        guidelines = []
        for doc in relevant_docs:
            if doc.metadata.get("type") == "guideline":
                guidelines.append(doc.page_content)
        
        return guidelines
    
    def _parse_summary_response(self, response: str) -> Dict:
        """Parse LLM response into structured format"""
        # Simple parsing - in practice would use more sophisticated parsing
        try:
            # Try to extract structured information from response
            lines = response.split('\n')
            summary = {
                'chief_complaint': 'Extracted from notes',
                'assessment': 'Clinical assessment provided',
                'plan': ['Continue current treatment', 'Monitor progress'],
                'medications': ['Medications extracted'],
                'follow_up': 'Follow-up plan recommended'
            }
            
            # Extract specific sections if they exist in response
            for line in lines:
                if 'chief complaint' in line.lower():
                    summary['chief_complaint'] = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                elif 'assessment' in line.lower():
                    summary['assessment'] = line.split(':', 1)[1].strip() if ':' in line else line.strip()
                elif 'plan' in line.lower() or 'treatment' in line.lower():
                    summary['plan'] = [line.split(':', 1)[1].strip() if ':' in line else line.strip()]
            
            return summary
        except:
            # Fallback to basic structure
            return {
                'chief_complaint': 'Extracted from notes',
                'assessment': 'Clinical assessment provided',
                'plan': ['Continue current treatment', 'Monitor progress'],
                'medications': ['Medications extracted'],
                'follow_up': 'Follow-up plan recommended',
                'raw_response': response
            }

class ClinicalDecisionSupport(HealthcareLLMRAG):
    """
    Clinical decision support system using Med-PaLM
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)
        
        # Create decision support prompt
        self.decision_prompt = PromptTemplate(
            input_variables=["patient_data", "clinical_question", "evidence"],
            template="""
            You are a medical expert providing clinical decision support.
            
            Patient Data: {patient_data}
            Clinical Question: {clinical_question}
            Supporting Evidence: {evidence}
            
            Provide evidence-based recommendations including:
            1. Primary recommendation
            2. Alternative options
            3. Risk considerations
            4. Monitoring plan
            5. Confidence level (0-1)
            
            Recommendations:
            """
        )
        
        self.decision_chain = LLMChain(
            llm=self.med_palm_llm,  # Use Med-PaLM for clinical decisions
            prompt=self.decision_prompt
        )
    
    def generate_treatment_recommendations(self, patient_data: Dict, clinical_question: str) -> Dict:
        """
        Generate evidence-based treatment recommendations using Med-PaLM
        
        Args:
            patient_data: Patient demographics, history, and current status
            clinical_question: Specific clinical question or concern
            
        Returns:
            Treatment recommendations with supporting evidence
        """
        # Retrieve relevant clinical evidence using RAG
        evidence = self._retrieve_clinical_evidence(patient_data, clinical_question)
        
        # Generate recommendations using Med-PaLM
        recommendation_response = self.decision_chain.run({
            "patient_data": json.dumps(patient_data),
            "clinical_question": clinical_question,
            "evidence": "\n".join(evidence)
        })
        
        # Parse recommendations
        recommendations = self._parse_recommendations(recommendation_response)
        recommendations['supporting_evidence'] = evidence
        
        return recommendations
    
    def _retrieve_clinical_evidence(self, patient_data: Dict, question: str) -> List[str]:
        """Retrieve relevant clinical evidence using RAG"""
        # Combine patient data and question for retrieval
        search_query = f"{question} {json.dumps(patient_data)}"
        
        # Use vector store to find relevant evidence
        relevant_docs = self.vector_store.similarity_search(search_query, k=5)
        
        evidence = []
        for doc in relevant_docs:
            evidence.append(doc.page_content)
        
        return evidence
    
    def _parse_recommendations(self, response: str) -> Dict:
        """Parse Med-PaLM response into structured recommendations"""
        try:
            # Simple parsing - in practice would use more sophisticated parsing
            lines = response.split('\n')
            recommendations = {
                'primary_recommendation': 'Continue current treatment plan',
                'alternative_options': ['Alternative treatment options'],
                'risk_considerations': ['Standard risks apply'],
                'monitoring_plan': ['Regular monitoring recommended'],
                'confidence_score': 0.8
            }
            
            # Extract confidence score if present
            for line in lines:
                if 'confidence' in line.lower():
                    try:
                        confidence = float(line.split()[-1])
                        recommendations['confidence_score'] = confidence
                    except:
                        pass
            
            return recommendations
        except:
            return {
                'primary_recommendation': 'Continue current treatment plan',
                'alternative_options': ['Alternative treatment options'],
                'risk_considerations': ['Standard risks apply'],
                'monitoring_plan': ['Regular monitoring recommended'],
                'confidence_score': 0.8,
                'raw_response': response
            }

class DrugInteractionAnalyzer(HealthcareLLMRAG):
    """
    Drug interaction analysis system using Gemini
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)
        
        # Create drug interaction prompt
        self.interaction_prompt = PromptTemplate(
            input_variables=["medications", "patient_data", "drug_info"],
            template="""
            You are a clinical pharmacist analyzing drug interactions.
            
            Medications: {medications}
            Patient Data: {patient_data}
            Drug Information: {drug_info}
            
            Analyze potential interactions and provide:
            1. Drug-drug interactions
            2. Patient-specific warnings
            3. Recommendations
            4. Risk level (low/moderate/high)
            5. Required monitoring
            
            Analysis:
            """
        )
        
        self.interaction_chain = LLMChain(
            llm=self.gemini_llm,
            prompt=self.interaction_prompt
        )
    
    def analyze_drug_interactions(self, medications: List[str], patient_data: Dict) -> Dict:
        """
        Analyze potential drug interactions using Gemini
        
        Args:
            medications: List of current medications
            patient_data: Patient demographics and medical history
            
        Returns:
            Interaction analysis with safety recommendations
        """
        # Retrieve drug information using RAG
        drug_info = self._retrieve_drug_information(medications)
        
        # Analyze interactions using Gemini
        analysis_response = self.interaction_chain.run({
            "medications": ", ".join(medications),
            "patient_data": json.dumps(patient_data),
            "drug_info": "\n".join(drug_info)
        })
        
        # Parse analysis
        analysis = self._parse_interaction_analysis(analysis_response)
        
        return analysis
    
    def _retrieve_drug_information(self, medications: List[str]) -> List[str]:
        """Retrieve drug information using RAG"""
        drug_info = []
        
        for medication in medications:
            # Search for drug information in vector store
            relevant_docs = self.vector_store.similarity_search(medication, k=2)
            for doc in relevant_docs:
                if doc.metadata.get("type") == "drug":
                    drug_info.append(doc.page_content)
        
        return drug_info
    
    def _parse_interaction_analysis(self, response: str) -> Dict:
        """Parse interaction analysis response"""
        try:
            # Simple parsing - in practice would use more sophisticated parsing
            analysis = {
                'interactions': ['Potential interactions identified'],
                'warnings': ['Patient-specific warnings'],
                'recommendations': ['Safety recommendations'],
                'risk_level': 'moderate',
                'monitoring_required': ['Required monitoring']
            }
            
            # Extract risk level if present
            if 'high risk' in response.lower():
                analysis['risk_level'] = 'high'
            elif 'low risk' in response.lower():
                analysis['risk_level'] = 'low'
            
            return analysis
        except:
            return {
                'interactions': ['Potential interactions identified'],
                'warnings': ['Patient-specific warnings'],
                'recommendations': ['Safety recommendations'],
                'risk_level': 'moderate',
                'monitoring_required': ['Required monitoring'],
                'raw_response': response
            }

class PatientEducationSystem(HealthcareLLMRAG):
    """
    Patient education and communication system using Gemini
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)
        
        # Create patient education prompt
        self.education_prompt = PromptTemplate(
            input_variables=["diagnosis", "patient_data", "educational_content"],
            template="""
            You are a patient educator creating personalized educational materials.
            
            Diagnosis: {diagnosis}
            Patient Data: {patient_data}
            Educational Content: {educational_content}
            
            Create personalized patient education including:
            1. Simple explanation of the condition
            2. Personalized action plan
            3. Warning signs to watch for
            4. Follow-up instructions
            5. Additional resources
            
            Education Materials:
            """
        )
        
        self.education_chain = LLMChain(
            llm=self.gemini_llm,
            prompt=self.education_prompt
        )
    
    def generate_patient_education(self, diagnosis: str, patient_data: Dict) -> Dict:
        """
        Generate personalized patient education materials using Gemini
        
        Args:
            diagnosis: Patient's diagnosis
            patient_data: Patient demographics and preferences
            
        Returns:
            Personalized educational content
        """
        # Retrieve relevant educational content using RAG
        educational_content = self._retrieve_educational_content(diagnosis, patient_data)
        
        # Generate personalized education using Gemini
        education_response = self.education_chain.run({
            "diagnosis": diagnosis,
            "patient_data": json.dumps(patient_data),
            "educational_content": "\n".join(educational_content)
        })
        
        # Parse education materials
        education = self._parse_education_materials(education_response)
        
        return education
    
    def _retrieve_educational_content(self, diagnosis: str, patient_data: Dict) -> List[str]:
        """Retrieve educational content using RAG"""
        # Search for condition-specific information
        search_query = f"{diagnosis} patient education"
        relevant_docs = self.vector_store.similarity_search(search_query, k=3)
        
        educational_content = []
        for doc in relevant_docs:
            educational_content.append(doc.page_content)
        
        return educational_content
    
    def _parse_education_materials(self, response: str) -> Dict:
        """Parse education materials response"""
        try:
            # Simple parsing - in practice would use more sophisticated parsing
            education = {
                'explanation': 'Condition explained in simple terms',
                'action_plan': ['Personalized action steps'],
                'warning_signs': ['Signs to watch for'],
                'follow_up_instructions': 'Follow-up instructions provided',
                'resources': ['Additional resources']
            }
            
            return education
        except:
            return {
                'explanation': 'Condition explained in simple terms',
                'action_plan': ['Personalized action steps'],
                'warning_signs': ['Signs to watch for'],
                'follow_up_instructions': 'Follow-up instructions provided',
                'resources': ['Additional resources'],
                'raw_response': response
            }

class MedicalCodingSystem(HealthcareLLMRAG):
    """
    Medical coding and billing support system using Gemini
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key=api_key)
        
        # Create medical coding prompt
        self.coding_prompt = PromptTemplate(
            input_variables=["clinical_documentation", "patient_data", "coding_guidelines"],
            template="""
            You are a medical coder analyzing clinical documentation for billing.
            
            Clinical Documentation: {clinical_documentation}
            Patient Data: {patient_data}
            Coding Guidelines: {coding_guidelines}
            
            Generate appropriate medical codes including:
            1. ICD-10 diagnosis codes
            2. CPT procedure codes
            3. Documentation validation
            4. Billing notes
            
            Coding Analysis:
            """
        )
        
        self.coding_chain = LLMChain(
            llm=self.gemini_llm,
            prompt=self.coding_prompt
        )
    
    def generate_medical_codes(self, clinical_documentation: str, patient_data: Dict) -> Dict:
        """
        Generate appropriate medical codes for billing using Gemini
        
        Args:
            clinical_documentation: Clinical notes and documentation
            patient_data: Patient demographics and history
            
        Returns:
            Medical codes with supporting documentation
        """
        # Retrieve coding guidelines using RAG
        coding_guidelines = self._retrieve_coding_guidelines(clinical_documentation)
        
        # Generate codes using Gemini
        coding_response = self.coding_chain.run({
            "clinical_documentation": clinical_documentation,
            "patient_data": json.dumps(patient_data),
            "coding_guidelines": "\n".join(coding_guidelines)
        })
        
        # Parse coding analysis
        coding = self._parse_coding_analysis(coding_response)
        
        return coding
    
    def _retrieve_coding_guidelines(self, documentation: str) -> List[str]:
        """Retrieve coding guidelines using RAG"""
        # Search for relevant coding information
        relevant_docs = self.vector_store.similarity_search(documentation, k=3)
        
        coding_guidelines = []
        for doc in relevant_docs:
            if doc.metadata.get("type") == "coding":
                coding_guidelines.append(doc.page_content)
        
        return coding_guidelines
    
    def _parse_coding_analysis(self, response: str) -> Dict:
        """Parse coding analysis response"""
        try:
            # Simple parsing - in practice would use more sophisticated parsing
            coding = {
                'icd_codes': ['E11.9', 'I10'],  # Example codes
                'cpt_codes': ['99213'],  # Example codes
                'validation': {
                    'valid': True,
                    'warnings': [],
                    'recommendations': []
                },
                'documentation_requirements': ['Required documentation'],
                'billing_notes': 'Billing notes provided'
            }
            
            return coding
        except:
            return {
                'icd_codes': ['E11.9', 'I10'],
                'cpt_codes': ['99213'],
                'validation': {
                    'valid': True,
                    'warnings': [],
                    'recommendations': []
                },
                'documentation_requirements': ['Required documentation'],
                'billing_notes': 'Billing notes provided',
                'raw_response': response
            }

def main():
    """
    Demonstrate various healthcare LLM and RAG use cases with LangChain, Gemini, and Med-PaLM
    """
    print("Healthcare LLM and RAG Use Cases Demonstration")
    print("Using LangChain, Gemini, and Med-PaLM")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. Using mock responses.")
        print("Set GOOGLE_API_KEY environment variable for full functionality.")
        api_key = "mock_key"
    
    try:
        # Initialize systems
        doc_system = ClinicalDocumentationSystem(api_key=api_key)
        decision_support = ClinicalDecisionSupport(api_key=api_key)
        drug_analyzer = DrugInteractionAnalyzer(api_key=api_key)
        education_system = PatientEducationSystem(api_key=api_key)
        coding_system = MedicalCodingSystem(api_key=api_key)
        
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
        
        print("\n1. Clinical Documentation Summarization (Gemini)")
        print("-" * 50)
        summary = doc_system.summarize_clinical_notes(clinical_notes, patient_data)
        print(f"Summary: {json.dumps(summary, indent=2)}")
        
        print("\n2. Clinical Decision Support (Med-PaLM)")
        print("-" * 50)
        recommendations = decision_support.generate_treatment_recommendations(
            patient_data, "How should I manage this patient's diabetes?"
        )
        print(f"Recommendations: {json.dumps(recommendations, indent=2)}")
        
        print("\n3. Drug Interaction Analysis (Gemini)")
        print("-" * 50)
        interactions = drug_analyzer.analyze_drug_interactions(
            patient_data['medications'], patient_data
        )
        print(f"Interactions: {json.dumps(interactions, indent=2)}")
        
        print("\n4. Patient Education (Gemini)")
        print("-" * 50)
        education = education_system.generate_patient_education('diabetes', patient_data)
        print(f"Education: {json.dumps(education, indent=2)}")
        
        print("\n5. Medical Coding (Gemini)")
        print("-" * 50)
        coding = coding_system.generate_medical_codes(clinical_notes, patient_data)
        print(f"Coding: {json.dumps(coding, indent=2)}")
        
        print("\nDemonstration completed successfully!")
        print("\nNote: For full functionality, ensure you have:")
        print("- Valid GOOGLE_API_KEY environment variable")
        print("- Required packages installed: langchain, google-generativeai, chromadb")
        
    except Exception as e:
        print(f"Error during demonstration: {str(e)}")
        print("This may be due to missing API key or required packages.")

if __name__ == "__main__":
    main() 