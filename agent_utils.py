import os
import random
import time
from dotenv import load_dotenv
from functools import lru_cache
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.tools import tool
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import AIMessage, HumanMessage

load_dotenv()

# Persistent conversation memory with size limits
class PersistentMemory:
    def __init__(self, max_messages=100):
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
        self.previous_questions = []
        self.max_messages = max_messages
        self.message_count = 0
    
    def add_interaction(self, question: str, response: str):
        # Clean up old messages if we're over limit
        if self.message_count >= self.max_messages:
            # Remove oldest messages
            self.memory.chat_memory.messages = self.memory.chat_memory.messages[2:]
            self.previous_questions = self.previous_questions[1:]
            self.message_count -= 1
        
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)
        self.previous_questions.append(question)
        self.message_count += 1
    
    def get_memory(self):
        return self.memory
    
    def get_previous_questions(self):
        return self.previous_questions
    
    def clear_memory(self):
        self.memory.clear()
        self.previous_questions = []
        self.message_count = 0

# Singleton memory instance with 100 message limit
memory = PersistentMemory(max_messages=100)

def validate_api_key(api_key: str) -> bool:
    """Validate the Google API key"""
    try:
        test_model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )
        test_response = test_model.invoke("Test")
        return bool(test_response.content)
    except Exception:
        return False

def get_model(api_key: str = None):
    """Get model instance with enhanced rate limiting and monitoring"""
    if not api_key:
        raise ValueError("Google API key is required")
    
    if not validate_api_key(api_key):
        raise ValueError("Invalid Google API key")
    
    # Enhanced rate limiting with jitter and longer backoff
    max_retries = 5
    base_delay = 2.0  # seconds
    max_delay = 30.0  # seconds
    
    for attempt in range(max_retries):
        try:
            model = ChatGoogleGenerativeAI(
                model="gemini-pro",
                google_api_key=api_key,
                temperature=0.7,
                max_output_tokens=1024,  # Increased for better responses
                convert_system_message_to_human=True,
                request_timeout=10.0,  # Increased timeout
                max_retries=0  # Let our custom logic handle retries
            )
            
            # Test the model with a small prompt
            test_response = model.invoke("Test")
            if not test_response.content:
                raise ValueError("Empty response from model")
                
            return model
                
        except Exception as e:
            if "ResourceExhausted" in str(e):
                # Exponential backoff with jitter
                delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                time.sleep(delay)
                continue
            elif attempt == max_retries - 1:
                raise RuntimeError(f"API request failed after {max_retries} attempts: {str(e)}")
            else:
                raise

    # Enhanced API usage tracking
    if not hasattr(get_model, 'api_usage'):
        get_model.api_usage = {
            'total_calls': 0,
            'failed_calls': 0,
            'last_call': None,
            'rate_limit': 60,  # calls per minute
            'window_start': time.time(),
            'call_count': 0
        }
    
    # Reset rate limit window if expired
    if time.time() - get_model.api_usage['window_start'] > 60:
        get_model.api_usage['window_start'] = time.time()
        get_model.api_usage['call_count'] = 0
    
    # Enforce rate limiting
    get_model.api_usage['call_count'] += 1
    if get_model.api_usage['call_count'] > get_model.api_usage['rate_limit']:
        wait_time = 60 - (time.time() - get_model.api_usage['window_start'])
        time.sleep(wait_time)
        get_model.api_usage['window_start'] = time.time()
        get_model.api_usage['call_count'] = 1
    
    get_model.api_usage['total_calls'] += 1
    get_model.api_usage['last_call'] = time.time()
    
    if attempt > 0:
        get_model.api_usage['failed_calls'] += 1
        
    # Dynamic rate limiting based on failures
    if get_model.api_usage['failed_calls'] > 5:
        # Reduce rate limit if too many failures
        get_model.api_usage['rate_limit'] = max(30, get_model.api_usage['rate_limit'] - 10)
        print(f"Reduced rate limit to {get_model.api_usage['rate_limit']} calls/minute due to failures")
    
    # Log usage every 10 calls
    if get_model.api_usage['total_calls'] % 10 == 0:
        print(f"API usage: {get_model.api_usage['total_calls']} total calls, {get_model.api_usage['failed_calls']} failures")

def create_tools() -> List[Dict[str, Any]]:
    """Create and return a list of tools for medical pre-screening"""
    @tool
    def get_symptom_details(symptom: str) -> str:
        """Get detailed information about a specific symptom"""
        model = get_model()
        prompt = f"""
        You are a medical assistant. Provide detailed information about:
        {symptom}
        
        Include:
        - Common causes
        - Associated symptoms
        - When to seek medical attention
        - First aid recommendations
        """
        return model.invoke(prompt).content

    @tool
    def assess_urgency(response: str) -> str:
        """Assess the urgency of a patient's condition"""
        model = get_model()
        prompt = f"""
        Based on this patient response, assess the urgency:
        {response}
        
        Return one of:
        - Emergency: Requires immediate medical attention
        - Urgent: Should be seen within 24 hours
        - Routine: Can wait for regular appointment
        """
        return model.invoke(prompt).content

    return [get_symptom_details, assess_urgency]

def create_conversation_chain(api_key: str = None) -> ConversationChain:
    """Create a conversation chain with medical-specific prompts"""
    # Get the persistent memory instance
    memory_instance = memory.get_memory()
    if not memory_instance:
        memory_instance = ConversationBufferMemory(memory_key="chat_history", input_key="input")
    system_prompt = SystemMessagePromptTemplate.from_template("""
    You are a medical assistant conducting a structured pre-screening interview. Follow this protocol:

    1. Start with chief complaint:
       - Ask about primary symptoms
       - Establish timeline (onset, duration, frequency)
       - Characterize symptoms (quality, severity, location)
       - Identify aggravating/alleviating factors

    2. Systematic review:
       - Cardiovascular: Chest pain, palpitations, edema
       - Respiratory: Shortness of breath, cough, wheezing
       - Gastrointestinal: Nausea, vomiting, diarrhea, abdominal pain
       - Neurological: Headache, dizziness, weakness, numbness
       - Other: Fever, weight changes, fatigue

    3. Medical history:
       - Past medical conditions
       - Surgeries and hospitalizations
       - Medications and allergies
       - Family history of major diseases

    4. Red flags assessment:
       - Sudden onset/severe symptoms
       - Neurological deficits
       - Chest pain with exertion
       - Unintentional weight loss
       - Persistent fever

    Maintain this structure while:
    - Using clear, specific questions
    - Keeping professional yet empathetic tone
    - Avoiding repetition
    - Prioritizing urgent symptoms
    - Maintaining conversation context

    Current conversation history:
    {chat_history}
    """)
    
    human_prompt = HumanMessagePromptTemplate.from_template("{input}")
    
    prompt_template = ChatPromptTemplate.from_messages([
        system_prompt,
        human_prompt
    ])
    
    return ConversationChain(
        llm=get_model(api_key),
        prompt=prompt_template,
        memory=memory_instance,
        verbose=True
    )

def handle_llm_interaction(prompt: str, tools: List[Dict[str, Any]], api_key: str = None) -> str:
    """Handle LLM interaction with simplified flow"""
    try:
        conversation = create_conversation_chain(api_key)
        
        # Add tools context to the prompt
        tool_context = "\n".join([
            f"Available tool: {tool.name} - {tool.description}"
            for tool in tools
        ])
        
        # Get previous questions for context
        previous_questions = memory.get_previous_questions()
        
        full_prompt = f"""
        {prompt}
        
        Previous Questions:
        {previous_questions}
        
        Available Tools:
        {tool_context}
        """
        
        # Single attempt with timeout
        response = conversation.predict(input=full_prompt)
        memory.add_interaction(prompt, response)
        return response
        
    except Exception as e:
        if "ResourceExhausted" in str(e):
            # Single retry with short delay
            time.sleep(1.5)
            try:
                response = conversation.predict(input=full_prompt)
                memory.add_interaction(prompt, response)
                return response
            except:
                return "I'm currently experiencing high demand. Please try again shortly."
        return "I'm currently experiencing high demand. Please try again shortly."

def reflect(previous_question: str, user_response: str, reflection_prompt: str, api_key: str = None) -> str:
    """Reflect on the user's response and assess phase completeness"""
    conversation = create_conversation_chain(api_key)
    
    # Add conversation history to persistent memory
    memory.add_interaction(previous_question, user_response)
    
    # Determine current phase based on previous questions
    current_phase = "chief_complaint"
    if any(q.lower() in ["chest pain", "palpitations", "edema"] for q in memory.get_previous_questions()):
        current_phase = "cardiovascular"
    elif any(q.lower() in ["shortness of breath", "cough", "wheezing"] for q in memory.get_previous_questions()):
        current_phase = "respiratory"
    elif any(q.lower() in ["nausea", "vomiting", "diarrhea", "abdominal pain"] for q in memory.get_previous_questions()):
        current_phase = "gastrointestinal"
    elif any(q.lower() in ["headache", "dizziness", "weakness", "numbness"] for q in memory.get_previous_questions()):
        current_phase = "neurological"
    elif any(q.lower() in ["fever", "weight changes", "fatigue"] for q in memory.get_previous_questions()):
        current_phase = "other_symptoms"
    elif any(q.lower() in ["past medical", "surgeries", "medications", "allergies"] for q in memory.get_previous_questions()):
        current_phase = "medical_history"
    
    phase_completeness = {
        "chief_complaint": {
            "required": ["main symptoms", "timeline", "characteristics", "aggravating/alleviating factors"],
            "covered": []
        },
        "cardiovascular": {
            "required": ["chest pain", "palpitations", "edema"],
            "covered": []
        },
        "respiratory": {
            "required": ["shortness of breath", "cough", "wheezing"],
            "covered": []
        },
        "gastrointestinal": {
            "required": ["nausea/vomiting", "diarrhea/constipation", "abdominal pain"],
            "covered": []
        },
        "neurological": {
            "required": ["headache", "dizziness", "weakness/numbness"],
            "covered": []
        },
        "other_symptoms": {
            "required": ["fever", "weight changes", "fatigue"],
            "covered": []
        },
        "medical_history": {
            "required": ["past medical", "surgeries", "medications", "allergies", "family history"],
            "covered": []
        }
    }
    
    # Analyze which aspects have been covered
    for aspect in phase_completeness[current_phase]["required"]:
        if any(aspect in q.lower() for q in memory.get_previous_questions()):
            phase_completeness[current_phase]["covered"].append(aspect)
    
    # Check for urgent symptoms
    urgent_keywords = ["chest pain", "shortness of breath", "severe headache", "unconscious", "bleeding"]
    is_urgent = any(keyword in user_response.lower() for keyword in urgent_keywords)
    
    prompt = f"""
    Current phase: {current_phase}
    Phase completeness: {len(phase_completeness[current_phase]["covered"])}/{len(phase_completeness[current_phase]["required"])}
    Urgent symptoms detected: {is_urgent}
    
    {reflection_prompt}
    
    Provide reflection that:
    1. Assesses completeness of current phase
    2. Identifies remaining aspects to cover
    3. Flags any urgent symptoms
    4. Determines if phase transition is needed
    5. Maintains clinical context
    """
    
    try:
        response = conversation.predict(input=prompt)
        # Store the reflection in memory
        memory.add_interaction(prompt, response)
        return response
    except Exception as e:
        print(f"Error generating reflection: {str(e)}")
        return "Can you please provide more details about your symptoms and medical history?"

# Cached common medical questions
COMMON_QUESTIONS = {
    "chief_complaint": [
        "Can you describe your main symptoms?",
        "When did these symptoms first appear?",
        "How would you describe the severity of your symptoms?",
        "What makes your symptoms better or worse?"
    ],
    "cardiovascular": [
        "Do you experience any chest pain? If so, can you describe it?",
        "Have you noticed any irregular heartbeats or palpitations?",
        "Do you have any swelling in your legs or feet?"
    ],
    "respiratory": [
        "Do you experience shortness of breath? When does it occur?",
        "Do you have a cough? If so, is it productive?",
        "Have you noticed any wheezing or difficulty breathing?"
    ],
    "gastrointestinal": [
        "Have you experienced any nausea or vomiting?",
        "Have you noticed any changes in your bowel movements?",
        "Do you have any abdominal pain? If so, where is it located?"
    ],
    "neurological": [
        "Do you experience headaches? If so, how would you describe them?",
        "Have you felt dizzy or lightheaded?",
        "Have you experienced any weakness or numbness?"
    ],
    "other_symptoms": [
        "Have you had a fever? If so, how high and for how long?",
        "Have you noticed any recent weight changes?",
        "Have you been feeling unusually tired or fatigued?"
    ],
    "medical_history": [
        "Do you have any existing medical conditions?",
        "Have you had any surgeries or hospitalizations?",
        "Are you currently taking any medications?",
        "Do you have any known allergies?"
    ]
}

def generate_next_question(user_response: str, reflection: str, previous_questions: List[str], api_key: str = None) -> str:
    """Generate the next question using cached questions when possible"""
    # Determine current phase based on previous questions
    current_phase = "chief_complaint"
    if any(q.lower() in ["chest pain", "palpitations", "edema"] for q in previous_questions):
        current_phase = "cardiovascular"
    elif any(q.lower() in ["shortness of breath", "cough", "wheezing"] for q in previous_questions):
        current_phase = "respiratory"
    elif any(q.lower() in ["nausea", "vomiting", "diarrhea", "abdominal pain"] for q in previous_questions):
        current_phase = "gastrointestinal"
    elif any(q.lower() in ["headache", "dizziness", "weakness", "numbness"] for q in previous_questions):
        current_phase = "neurological"
    elif any(q.lower() in ["fever", "weight changes", "fatigue"] for q in previous_questions):
        current_phase = "other_symptoms"
    elif any(q.lower() in ["past medical", "surgeries", "medications", "allergies"] for q in previous_questions):
        current_phase = "medical_history"
    
    # Get cached questions for this phase
    phase_questions = COMMON_QUESTIONS[current_phase]
    
    # Find first question not already asked
    for question in phase_questions:
        if not any(q.lower() in question.lower() for q in previous_questions):
            memory.add_interaction(question, user_response)
            return question
    
    # If all cached questions have been asked, use LangChain for complex follow-up
    if "urgent" in reflection.lower() or "critical" in reflection.lower():
        conversation = create_conversation_chain(api_key)
        try:
            prompt = f"""
            Based on this response and reflection, generate a critical follow-up question:
            Patient response: {user_response}
            Reflection: {reflection}
            Previous questions: {previous_questions}
            """
            response = conversation.predict(input=prompt)
            memory.add_interaction(response, user_response)
            return response
        except Exception as e:
            print(f"Error generating critical question: {str(e)}")
    
    # Fallback to simple question
    return "Can you please provide more details about that?"

def initial_question(api_key: str = None) -> str:
    """Generate the initial question using LangChain"""
    conversation = create_conversation_chain(api_key)
    
    try:
        response = conversation.predict(input="""
        Based on the medical history, what is an essential and open-ended question to ask a patient for pre-screening?
        Ensure the question is clear and specific.
        """)
        # Store the initial question in memory
        memory.add_interaction(response, "")
        return response
    except Exception as e:
        print(f"Error generating initial question: {str(e)}")
        return "Can you please describe your current symptoms and when they started?"

def generate_reflection_prompt(user_response: str, previous_questions: List[str], api_key: str = None) -> str:
    """Generate a reflection prompt using LangChain"""
    conversation = create_conversation_chain(api_key)
    
    prompt = f"""
    Patient's response: {user_response}
    Previous questions: {previous_questions}
    Generate a reflection prompt that can help assess if we are collecting relevant information from the user.
    """
    
    try:
        response = conversation.predict(input=prompt)
        return response
    except Exception as e:
        print(f"Error generating reflection prompt: {str(e)}")
        return "Please provide more details about your symptoms and medical history."
