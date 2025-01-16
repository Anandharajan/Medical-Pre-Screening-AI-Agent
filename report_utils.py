import streamlit as st
import json
from typing import Dict, List
from datetime import datetime

class MedicalReportGenerator:
    def __init__(self, conversation_history: List[Dict[str, str]], patient_info: Dict[str, str]):
        self.conversation_history = conversation_history
        self.patient_info = patient_info
        self.report_sections = {
            "patient_summary": {},
            "clinical_assessment": {},
            "risk_factors": {},
            "diagnostic_recommendations": {},
            "treatment_plan": {},
            "follow_up": {},
            "specialist_referral": {},
            "potential_diagnosis": {},
            "medication_info": {}
        }
        
    def extract_patient_summary(self) -> Dict[str, str]:
        """Extract and format patient summary information"""
        return {
            "name": self.patient_info.get("name", ""),
            "age": self.patient_info.get("age", 0),
            "gender": self.patient_info.get("gender", ""),
            "date_of_assessment": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "primary_complaint": self._extract_primary_complaint(),
            "medical_history": self._extract_medical_history()
        }
        
    def _extract_primary_complaint(self) -> Dict[str, str]:
        """Extract primary complaint details from conversation"""
        complaint = {}
        for message in self.conversation_history:
            if message["role"] == "user" and "symptom" in message["content"].lower():
                complaint["description"] = message["content"]
                break
        return complaint
        
    def _extract_medical_history(self) -> Dict[str, str]:
        """Extract medical history from conversation"""
        history = {
            "past_conditions": [],
            "medications": [],
            "allergies": [],
            "family_history": []
        }
        
        for message in self.conversation_history:
            content = message["content"].lower()
            if "medical history" in content:
                history["past_conditions"].append(content)
            elif "medication" in content:
                history["medications"].append(content)
            elif "allergy" in content:
                history["allergies"].append(content)
            elif "family history" in content:
                history["family_history"].append(content)
                
        return history
        
    def generate_clinical_assessment(self) -> Dict[str, str]:
        """Generate clinical assessment section"""
        assessment = {
            "symptoms": [],
            "physical_exam": {},
            "differential_diagnosis": []
        }
        
        # Extract symptoms
        for message in self.conversation_history:
            if message["role"] == "user" and any(word in message["content"].lower() 
                for word in ["pain", "ache", "discomfort", "symptom"]):
                assessment["symptoms"].append({
                    "description": message["content"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
                })
                
        return assessment
        
    def generate_risk_factors(self) -> Dict[str, str]:
        """Generate risk factors section"""
        risk_factors = {
            "lifestyle": [],
            "genetic": [],
            "environmental": []
        }
        
        for message in self.conversation_history:
            content = message["content"].lower()
            if any(word in content for word in ["smoke", "drink", "exercise", "diet"]):
                risk_factors["lifestyle"].append(content)
            elif "family history" in content:
                risk_factors["genetic"].append(content)
            elif any(word in content for word in ["work", "environment", "exposure"]):
                risk_factors["environmental"].append(content)
                
        return risk_factors
        
    def generate_diagnostic_recommendations(self) -> Dict[str, str]:
        """Generate diagnostic recommendations based on symptoms and history"""
        recommendations = {
            "laboratory_tests": [],
            "imaging_studies": [],
            "specialist_referrals": []
        }
        
        # Analyze symptoms for test recommendations
        symptoms = self.report_sections["clinical_assessment"]["symptoms"]
        for symptom in symptoms:
            desc = symptom["description"].lower()
            if any(word in desc for word in ["pain", "ache"]):
                recommendations["laboratory_tests"].append("Complete Blood Count")
                recommendations["imaging_studies"].append("X-Ray")
            if "chest" in desc:
                recommendations["imaging_studies"].append("Chest X-Ray")
            if "head" in desc:
                recommendations["imaging_studies"].append("CT Scan")
                
        # Add basic tests if no specific recommendations
        if not recommendations["laboratory_tests"]:
            recommendations["laboratory_tests"] = [
                "Complete Blood Count",
                "Basic Metabolic Panel"
            ]
            
        return recommendations
        
    def generate_treatment_plan(self) -> Dict[str, str]:
        """Generate treatment plan based on symptoms and history"""
        plan = {
            "immediate_interventions": [],
            "long_term_management": [],
            "lifestyle_modifications": [],
            "follow_up_schedule": []
        }
        
        # Add interventions based on symptoms
        symptoms = self.report_sections["clinical_assessment"]["symptoms"]
        for symptom in symptoms:
            desc = symptom["description"].lower()
            if any(word in desc for word in ["pain", "ache"]):
                plan["immediate_interventions"].append("Pain management")
            if "fever" in desc:
                plan["immediate_interventions"].append("Antipyretics")
            if "cough" in desc:
                plan["immediate_interventions"].append("Cough suppressant")
                
        # Add lifestyle modifications
        risk_factors = self.report_sections["risk_factors"]
        if risk_factors["lifestyle"]:
            plan["lifestyle_modifications"].append("Dietary changes")
            plan["lifestyle_modifications"].append("Exercise regimen")
            
        # Add follow-up schedule
        if symptoms:
            plan["follow_up_schedule"].append("1 week follow-up")
            plan["follow_up_schedule"].append("Monthly checkups for 3 months")
            
        return plan
        
    def generate_follow_up(self) -> Dict[str, str]:
        """Generate follow-up plan with monitoring parameters"""
        follow_up = {
            "monitoring_parameters": [],
            "warning_signs": [],
            "emergency_contacts": []
        }
        
        # Add monitoring based on symptoms
        symptoms = self.report_sections["clinical_assessment"]["symptoms"]
        for symptom in symptoms:
            desc = symptom["description"].lower()
            if "pain" in desc:
                follow_up["monitoring_parameters"].append("Pain level (1-10 scale)")
            if "fever" in desc:
                follow_up["monitoring_parameters"].append("Temperature 3x daily")
            if "cough" in desc:
                follow_up["monitoring_parameters"].append("Cough frequency")
                
        # Add warning signs
        if symptoms:
            follow_up["warning_signs"] = [
                "Increased pain",
                "High fever (>39°C)",
                "Difficulty breathing"
            ]
            
        # Add emergency contacts
        follow_up["emergency_contacts"] = [
            "Primary care physician",
            "Local emergency services"
        ]
        
        return follow_up
        
    def generate_report(self) -> Dict[str, Dict[str, str]]:
        """Generate complete medical report"""
        try:
            # Validate inputs
            if not self.conversation_history or not isinstance(self.conversation_history, list):
                raise ValueError("Invalid conversation history")
                
            if not self.patient_info or not isinstance(self.patient_info, dict):
                raise ValueError("Invalid patient information")
            
            # Generate report sections with error handling
            self.report_sections["patient_summary"] = self.extract_patient_summary() or {
                "name": "Unknown",
                "age": 0,
                "gender": "Unknown",
                "date_of_assessment": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "primary_complaint": {"description": "No primary complaint provided"},
                "medical_history": {}
            }
            
            self.report_sections["clinical_assessment"] = self.generate_clinical_assessment() or {
                "symptoms": [{"description": "No symptoms reported", "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}],
                "physical_exam": {},
                "differential_diagnosis": []
            }
            
            self.report_sections["risk_factors"] = self.generate_risk_factors() or {
                "lifestyle": ["No lifestyle risk factors identified"],
                "genetic": [],
                "environmental": []
            }
            
            self.report_sections["diagnostic_recommendations"] = self.generate_diagnostic_recommendations() or {
                "laboratory_tests": ["No specific tests recommended"],
                "imaging_studies": [],
                "specialist_referrals": []
            }
            
            self.report_sections["treatment_plan"] = self.generate_treatment_plan() or {
                "immediate_interventions": ["No immediate interventions required"],
                "long_term_management": [],
                "lifestyle_modifications": [],
                "follow_up_schedule": []
            }
            
            self.report_sections["follow_up"] = self.generate_follow_up() or {
                "monitoring_parameters": ["No specific monitoring required"],
                "warning_signs": [],
                "emergency_contacts": []
            }
            
            self.report_sections["specialist_referral"] = self.generate_specialist_referral() or {
                "recommended_specialists": ["No specialist referral required"],
                "urgency_level": "routine",
                "referral_notes": ""
            }
            
            self.report_sections["potential_diagnosis"] = self.generate_potential_diagnosis() or {
                "possible_conditions": ["No specific diagnosis identified"],
                "probability_estimates": {}
            }
            
            self.report_sections["medication_info"] = self.generate_medication_info() or {
                "common_medications": ["No specific medications recommended"],
                "dosage_ranges": {},
                "side_effects": [],
                "contraindications": []
            }
            
            return self.report_sections
            
        except Exception as e:
            # Log error and return default report
            print(f"Error generating report: {str(e)}")
            return {
                "error": str(e),
                "message": "Unable to generate complete report. Please check input data."
            }
        
    def generate_specialist_referral(self) -> Dict[str, str]:
        """Generate specialist referral section based on symptoms"""
        referral = {
            "recommended_specialists": [],
            "urgency_level": "routine",
            "referral_notes": ""
        }
        
        # Analyze symptoms for specialist recommendations
        symptoms = self.report_sections["clinical_assessment"]["symptoms"]
        for symptom in symptoms:
            desc = symptom["description"].lower()
            if any(word in desc for word in ["chest", "heart"]):
                referral["recommended_specialists"].append("Cardiologist")
                referral["urgency_level"] = "urgent"
            if any(word in desc for word in ["head", "brain", "neurological"]):
                referral["recommended_specialists"].append("Neurologist")
            if any(word in desc for word in ["joint", "bone", "muscle"]):
                referral["recommended_specialists"].append("Orthopedist")
            if any(word in desc for word in ["skin", "rash", "lesion"]):
                referral["recommended_specialists"].append("Dermatologist")
                
        # Add general practitioner if no specific specialists
        if not referral["recommended_specialists"]:
            referral["recommended_specialists"].append("General Practitioner")
            
        # Add referral notes based on symptoms
        if symptoms:
            referral["referral_notes"] = "Patient presents with: " + ", ".join(
                [s["description"] for s in symptoms]
            )
            
        return referral

    def generate_potential_diagnosis(self) -> Dict[str, str]:
        """Generate potential diagnosis based on symptoms and history"""
        diagnosis = {
            "possible_conditions": [],
            "probability_estimates": {}
        }
        
        # Analyze symptoms for potential conditions
        symptoms = self.report_sections["clinical_assessment"]["symptoms"]
        for symptom in symptoms:
            desc = symptom["description"].lower()
            if any(word in desc for word in ["chest", "heart"]):
                diagnosis["possible_conditions"].extend([
                    "Angina",
                    "Myocardial Infarction",
                    "Pericarditis"
                ])
            if any(word in desc for word in ["head", "brain"]):
                diagnosis["possible_conditions"].extend([
                    "Migraine",
                    "Tension Headache",
                    "Meningitis"
                ])
            if any(word in desc for word in ["joint", "bone"]):
                diagnosis["possible_conditions"].extend([
                    "Arthritis",
                    "Osteoporosis",
                    "Tendonitis"
                ])
                
        # Add probability estimates based on symptom count
        if symptoms:
            base_prob = 100 / (len(diagnosis["possible_conditions"]) or 1)
            for condition in diagnosis["possible_conditions"]:
                diagnosis["probability_estimates"][condition] = round(base_prob, 2)
                
        # Add common conditions if no specific matches
        if not diagnosis["possible_conditions"]:
            diagnosis["possible_conditions"] = [
                "Viral Infection",
                "Bacterial Infection",
                "Stress-related Condition"
            ]
            diagnosis["probability_estimates"] = {
                "Viral Infection": 60,
                "Bacterial Infection": 30,
                "Stress-related Condition": 10
            }
            
        return diagnosis

    def generate_medication_info(self) -> Dict[str, str]:
        """Generate medication information based on symptoms and history"""
        medications = {
            "common_medications": [],
            "dosage_ranges": {},
            "side_effects": [],
            "contraindications": []
        }
        
        # Analyze symptoms for medication recommendations
        symptoms = self.report_sections["clinical_assessment"]["symptoms"]
        for symptom in symptoms:
            desc = symptom["description"].lower()
            if any(word in desc for word in ["pain", "ache"]):
                medications["common_medications"].append("Acetaminophen")
                medications["dosage_ranges"]["Acetaminophen"] = "500-1000mg every 4-6 hours"
                medications["side_effects"].append("Liver damage with overdose")
            if any(word in desc for word in ["fever", "inflammation"]):
                medications["common_medications"].append("Ibuprofen")
                medications["dosage_ranges"]["Ibuprofen"] = "400-800mg every 6-8 hours"
                medications["side_effects"].append("Stomach irritation")
            if any(word in desc for word in ["cough", "cold"]):
                medications["common_medications"].append("Dextromethorphan")
                medications["dosage_ranges"]["Dextromethorphan"] = "10-20mg every 4-6 hours"
                medications["side_effects"].append("Drowsiness")
                
        # Add general recommendations if no specific matches
        if not medications["common_medications"]:
            medications["common_medications"] = ["Multivitamins"]
            medications["dosage_ranges"]["Multivitamins"] = "Once daily"
            medications["side_effects"] = ["Rare allergic reactions"]
            
        # Add contraindications based on medical history
        history = self.report_sections["patient_summary"]["medical_history"]
        if "liver disease" in str(history).lower():
            medications["contraindications"].append("Avoid acetaminophen")
        if "stomach ulcer" in str(history).lower():
            medications["contraindications"].append("Avoid NSAIDs")
            
        return medications

    def safe_format_report(self, report: Dict[str, Dict[str, str]]) -> str:
        """Safely format report with comprehensive error handling"""
        import io
        
        try:
            # Create a string buffer to capture the formatted report
            buffer = io.StringIO()
            
            # Report title
            buffer.write("# Medical Assessment Report\n")
            buffer.write("---\n")
            
            # Patient Summary
            buffer.write("## Patient Summary\n")
            buffer.write("### Basic Information\n")
            buffer.write(f"**Name:** {report.get('patient_summary', {}).get('name', 'Unknown')}\n")
            buffer.write(f"**Age:** {report.get('patient_summary', {}).get('age', 0)}\n")
            buffer.write(f"**Gender:** {report.get('patient_summary', {}).get('gender', 'Unknown')}\n")
            buffer.write(f"**Date:** {report.get('patient_summary', {}).get('date_of_assessment', 'Unknown')}\n")
            
            buffer.write("### Primary Complaint\n")
            primary_complaint = report.get('patient_summary', {}).get('primary_complaint', {})
            buffer.write(f"{primary_complaint.get('description', 'No primary complaint provided')}\n")
            
            # Clinical Assessment
            buffer.write("## Clinical Assessment\n")
            symptoms = report.get('clinical_assessment', {}).get('symptoms', [])
            if symptoms:
                buffer.write("### Reported Symptoms\n")
                for symptom in symptoms:
                    buffer.write(f"- {symptom.get('description', 'Unspecified symptom')}\n")
            else:
                buffer.write("### No symptoms reported\n")
            
            # Risk Factors
            buffer.write("## Risk Factors\n")
            lifestyle_factors = report.get('risk_factors', {}).get('lifestyle', [])
            if lifestyle_factors:
                buffer.write("### Lifestyle Factors\n")
                for factor in lifestyle_factors:
                    buffer.write(f"- {factor}\n")
            else:
                buffer.write("### No significant lifestyle risk factors identified\n")
            
            # Diagnostic Recommendations
            buffer.write("## Diagnostic Recommendations\n")
            lab_tests = report.get('diagnostic_recommendations', {}).get('laboratory_tests', [])
            if lab_tests:
                buffer.write("### Laboratory Tests\n")
                for test in lab_tests:
                    buffer.write(f"- {test}\n")
            
            imaging_studies = report.get('diagnostic_recommendations', {}).get('imaging_studies', [])
            if imaging_studies:
                buffer.write("### Imaging Studies\n")
                for study in imaging_studies:
                    buffer.write(f"- {study}\n")
            
            # Treatment Plan
            buffer.write("## Treatment Plan\n")
            interventions = report.get('treatment_plan', {}).get('immediate_interventions', [])
            if interventions:
                buffer.write("### Immediate Interventions\n")
                for intervention in interventions:
                    buffer.write(f"- {intervention}\n")
            
            # Follow-up Plan
            buffer.write("## Follow-up Plan\n")
            monitoring = report.get('follow_up', {}).get('monitoring_parameters', [])
            if monitoring:
                buffer.write("### Monitoring Parameters\n")
                for param in monitoring:
                    buffer.write(f"- {param}\n")
            
            # Specialist Referral
            buffer.write("## Specialist Referral\n")
            specialists = report.get('specialist_referral', {}).get('recommended_specialists', [])
            if specialists:
                buffer.write("### Recommended Specialists\n")
                for specialist in specialists:
                    buffer.write(f"- {specialist}\n")
            
            # Potential Diagnosis
            buffer.write("## Potential Diagnosis\n")
            conditions = report.get('potential_diagnosis', {}).get('possible_conditions', [])
            if conditions:
                buffer.write("### Possible Conditions\n")
                for condition in conditions:
                    buffer.write(f"- {condition}\n")
            
            # Medication Information
            buffer.write("## Medication Information\n")
            medications = report.get('medication_info', {}).get('common_medications', [])
            if medications:
                buffer.write("### Recommended Medications\n")
                for med in medications:
                    buffer.write(f"- {med}\n")
            
            # Final note
            buffer.write("---\n")
            buffer.write("**Please review this report carefully and consult with your healthcare provider.**\n")
            
            return buffer.getvalue()
            
        except Exception as e:
            return f"# Error Generating Report\n\nAn error occurred while generating the report: {str(e)}\n\nPlease contact support."

    def format_report(self, report: Dict[str, Dict[str, str]]) -> None:
        """Format report for display in Streamlit with clean, responsive layout"""
        
        # Report title
        st.title("Medical Assessment Report")
        st.markdown("---")
        
        # Patient Summary
        st.header("Patient Summary")
        with st.container():
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Basic Information")
                st.markdown(f"**Name:** {report['patient_summary']['name']}")
                st.markdown(f"**Age:** {report['patient_summary']['age']}")
                st.markdown(f"**Gender:** {report['patient_summary']['gender']}")
                st.markdown(f"**Date:** {report['patient_summary']['date_of_assessment']}")
            
            with col2:
                st.subheader("Primary Complaint")
                primary_complaint = report['patient_summary'].get('primary_complaint', {})
                st.markdown(primary_complaint.get('description', 'Not specified'))
        
        st.markdown("---")
        
        # Clinical Assessment
        st.header("Clinical Assessment")
        if report['clinical_assessment']['symptoms']:
            st.subheader("Reported Symptoms")
            for symptom in report['clinical_assessment']['symptoms']:
                st.markdown(f"- {symptom['description']}")
        else:
            st.info("No symptoms reported")
        
        st.markdown("---")
        
        # Risk Factors
        st.header("Risk Factors")
        if report['risk_factors']['lifestyle']:
            st.subheader("Lifestyle Factors")
            for factor in report['risk_factors']['lifestyle']:
                st.markdown(f"- {factor}")
        else:
            st.info("No significant lifestyle risk factors identified")
        
        st.markdown("---")
        
        # Diagnostic Recommendations
        st.header("Diagnostic Recommendations")
        if report['diagnostic_recommendations']['laboratory_tests']:
            st.subheader("Laboratory Tests")
            for test in report['diagnostic_recommendations']['laboratory_tests']:
                st.markdown(f"- {test}")
        
        if report['diagnostic_recommendations']['imaging_studies']:
            st.subheader("Imaging Studies")
            for study in report['diagnostic_recommendations']['imaging_studies']:
                st.markdown(f"- {study}")
        
        st.markdown("---")
        
        # Treatment Plan
        st.header("Treatment Plan")
        if report['treatment_plan']['immediate_interventions']:
            st.subheader("Immediate Interventions")
            for intervention in report['treatment_plan']['immediate_interventions']:
                st.markdown(f"- {intervention}")
        
        if report['treatment_plan']['long_term_management']:
            st.subheader("Long-term Management")
            for management in report['treatment_plan']['long_term_management']:
                st.markdown(f"- {management}")
        
        st.markdown("---")
        
        # Follow-up Plan
        st.header("Follow-up Plan")
        if report['follow_up']['monitoring_parameters']:
            st.subheader("Monitoring Parameters")
            for parameter in report['follow_up']['monitoring_parameters']:
                st.markdown(f"- {parameter}")
        
        if report['follow_up']['warning_signs']:
            st.subheader("Warning Signs")
            for sign in report['follow_up']['warning_signs']:
                st.error(f"- {sign}")
        
        st.markdown("---")
        
        # Specialist Referral
        st.header("Specialist Referral")
        if report['specialist_referral']['recommended_specialists']:
            st.subheader("Recommended Specialists")
            for specialist in report['specialist_referral']['recommended_specialists']:
                st.success(f"- {specialist}")
            
            st.subheader("Urgency Level")
            st.markdown(f"- {report['specialist_referral']['urgency_level'].upper()}")
            
            if report['specialist_referral']['referral_notes']:
                st.subheader("Referral Notes")
                st.markdown(f"- {report['specialist_referral']['referral_notes']}")
        
        st.markdown("---")
        
        # Potential Diagnosis
        st.header("Potential Diagnosis")
        if report['potential_diagnosis']['possible_conditions']:
            st.subheader("Possible Conditions")
            for condition in report['potential_diagnosis']['possible_conditions']:
                st.warning(f"- {condition}")
            
            st.subheader("Probability Estimates")
            for condition, probability in report['potential_diagnosis']['probability_estimates'].items():
                st.markdown(f"- {condition}: {probability}%")
        
        st.markdown("---")
        
        # Medication Information
        st.header("Medication Information")
        if report['medication_info']['common_medications']:
            st.subheader("Recommended Medications")
            for med in report['medication_info']['common_medications']:
                st.success(f"- {med}")
            
            st.subheader("Dosage Information")
            for med, dosage in report['medication_info']['dosage_ranges'].items():
                st.markdown(f"- {med}: {dosage}")
            
            if report['medication_info']['side_effects']:
                st.subheader("Potential Side Effects")
                for side_effect in report['medication_info']['side_effects']:
                    st.warning(f"- {side_effect}")
            
            if report['medication_info']['contraindications']:
                st.subheader("Contraindications")
                for contraindication in report['medication_info']['contraindications']:
                    st.error(f"- {contraindication}")
        
        # Final note
        st.markdown("---")
        st.info("NOTE: This report is generated based on the information provided and should be reviewed by a qualified medical professional.")