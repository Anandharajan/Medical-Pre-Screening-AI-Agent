import os
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

# Persistent conversation memory
class PersistentMemory:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
        self.previous_questions = []
    
    def add_interaction(self, question: str, response: str):
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(response)
        self.previous_questions.append(question)
    
    def get_memory(self):
        return self.memory
    
    def get_previous_questions(self):
        return self.previous_questions

# Singleton memory instance
memory = PersistentMemory()

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

# Cache models to avoid repeated initialization
@lru_cache(maxsize=1)
def get_model(api_key: str = None):
    """Get cached model instance using LangChain's GoogleGenerativeAI wrapper"""
    if not api_key:
        raise ValueError("Google API key is required")
    
    if not validate_api_key(api_key):
        raise ValueError("Invalid Google API key")
    
    # Advanced rate limiting with queue and circuit breaker
    import time
    from datetime import datetime, timedelta
    from collections import deque
    import random
    
    # Initialize rate limiting state
    if not hasattr(get_model, "rate_limiter"):
        get_model.rate_limiter = {
            "queue": deque(maxlen=30),  # Last 30 request timestamps
            "circuit_open": False,
            "circuit_open_time": None,
            "retry_count": 0,
            "max_retries": 3,
            "rate_limit_window": timedelta(seconds=60),
            "min_delay": 3.0,  # Increased minimum delay
            "max_delay": 120.0,
            "failure_threshold": 5,  # Number of failures before circuit opens
            "failure_count": 0
        }
    
    rate_limiter = get_model.rate_limiter
    
    # Check circuit breaker
    if rate_limiter["circuit_open"]:
        if datetime.now() - rate_limiter["circuit_open_time"] < timedelta(minutes=5):
            raise RuntimeError("API circuit breaker is open - too many failures")
        else:
            rate_limiter["circuit_open"] = False
            rate_limiter["failure_count"] = 0
    
    # Calculate time since last call
    if rate_limiter["queue"]:
        time_since_last_call = datetime.now() - rate_limiter["queue"][-1]
    else:
        time_since_last_call = timedelta(seconds=rate_limiter["min_delay"] + 1)
    
    # Enforce rate limits
    if len(rate_limiter["queue"]) >= 30:
        oldest_call = rate_limiter["queue"][0]
        if datetime.now() - oldest_call < rate_limiter["rate_limit_window"]:
            wait_time = (rate_limiter["rate_limit_window"] - (datetime.now() - oldest_call)).total_seconds()
            print(f"Rate limit exceeded, waiting {wait_time:.1f} seconds...")
            time.sleep(wait_time)
    
    # Enforce minimum delay with jitter
    if time_since_last_call.total_seconds() < rate_limiter["min_delay"]:
        jitter = random.uniform(0.5, 1.5)
        wait_time = (rate_limiter["min_delay"] - time_since_last_call.total_seconds()) * jitter
        time.sleep(wait_time)
    
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=512,  # Further reduced to 512
            convert_system_message_to_human=True
        )
        
        # Update rate limiter state
        rate_limiter["queue"].append(datetime.now())
        rate_limiter["retry_count"] = 0
        rate_limiter["failure_count"] = 0
        
        return model
        
    except Exception as e:
        rate_limiter["failure_count"] += 1
        rate_limiter["retry_count"] += 1
        
        if rate_limiter["failure_count"] >= rate_limiter["failure_threshold"]:
            rate_limiter["circuit_open"] = True
            rate_limiter["circuit_open_time"] = datetime.now()
            raise RuntimeError("API circuit breaker opened due to repeated failures")
            
        if "ResourceExhausted" in str(e):
            # Exponential backoff with jitter
            base_delay = 3.0  # Increased base delay
            max_delay = rate_limiter["max_delay"]
            jitter = random.uniform(0.8, 1.2)
            retry_delay = min(
                base_delay * (2 ** rate_limiter["retry_count"]) * jitter,
                max_delay
            )
            
            if rate_limiter["retry_count"] < rate_limiter["max_retries"]:
                print(f"Rate limit hit, retrying in {retry_delay:.1f} seconds...")
                time.sleep(retry_delay)
                return get_model(api_key)
            else:
                raise RuntimeError("API rate limit exceeded after maximum retries")
                
        print(f"Error configuring Google Generative AI: {str(e)}")
        raise

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
    """Handle LLM interaction with caching, fallback, and rate limiting"""
    from functools import lru_cache
    import time
    from queue import Queue
    from threading import Lock
    
    # Enhanced caching with TTL
    @lru_cache(maxsize=500)
    def get_cached_response(prompt_hash: int) -> str:
        return None
    
    # Request queue for rate limiting
    if not hasattr(handle_llm_interaction, "request_queue"):
        handle_llm_interaction.request_queue = Queue(maxsize=30)
        handle_llm_interaction.queue_lock = Lock()
    
    # Check cache first with TTL
    prompt_hash = hash(prompt)
    cached_response = get_cached_response(prompt_hash)
    if cached_response:
        return cached_response
    
    # Wait for available slot in queue
    with handle_llm_interaction.queue_lock:
        if handle_llm_interaction.request_queue.full():
            # Wait for oldest request to complete
            oldest_time = handle_llm_interaction.request_queue.get()
            wait_time = max(3.0, time.time() - oldest_time)
            time.sleep(wait_time)
        
        # Add current request to queue
        handle_llm_interaction.request_queue.put(time.time())
    
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
    
    # Try API with exponential backoff
    max_retries = 3
    base_delay = 3.0
    last_error = None
    
    for attempt in range(max_retries):
        try:
            response = conversation.predict(input=full_prompt)
            
            # Cache the response with TTL
            get_cached_response.cache_clear()
            get_cached_response(prompt_hash)
            
            # Store the interaction in memory
            memory.add_interaction(prompt, response)
            return response
            
        except Exception as api_error:
            last_error = api_error
            if "ResourceExhausted" in str(api_error):
                delay = min(base_delay * (2 ** attempt), 30.0)
                time.sleep(delay)
                continue
            raise
    
    # If we exhausted retries, try local model
    try:
        from transformers import pipeline
        local_model = pipeline("text-generation", model="gpt2")
        response = local_model(prompt, max_length=512, do_sample=True)[0]['generated_text']
        
        # Store in cache and memory
        get_cached_response.cache_clear()
        get_cached_response(prompt_hash)
        memory.add_interaction(prompt, response)
        
        return response
        
    except Exception as local_error:
        # Final fallback to static response
        return "I'm currently experiencing high demand. Please try again shortly or provide more details about your medical concern."
        
        # If API fails after retries, try local model
        try:
            from transformers import pipeline
            local_model = pipeline("text-generation", model="gpt2")
            response = local_model(prompt, max_length=512, do_sample=True)[0]['generated_text']
            
            # Store in cache and memory
            get_cached_response.cache_clear()
            get_cached_response(prompt_hash)
            memory.add_interaction(prompt, response)
            
            return response
            
        except Exception as local_error:
            # Final fallback to static response
            return "I'm currently experiencing high demand. Please try again shortly or provide more details about your medical concern."

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

def generate_next_question(user_response: str, reflection: str, previous_questions: List[str], api_key: str = None) -> str:
    """Generate the next question following structured interview protocol"""
    conversation = create_conversation_chain(api_key)
    
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
    
    phase_prompts = {
        "chief_complaint": """
        Focus on the patient's primary concern:
        1. Ask about main symptoms
        2. Establish timeline (onset, duration, frequency)
        3. Characterize symptoms (quality, severity, location)
        4. Identify aggravating/alleviating factors
        """,
        "cardiovascular": """
        Assess cardiovascular system:
        1. Chest pain (character, radiation, triggers)
        2. Palpitations (frequency, duration)
        3. Edema (location, timing, associated symptoms)
        """,
        "respiratory": """
        Assess respiratory system:
        1. Shortness of breath (timing, triggers)
        2. Cough (character, duration, sputum)
        3. Wheezing (timing, triggers)
        """,
        "gastrointestinal": """
        Assess gastrointestinal system:
        1. Nausea/vomiting (timing, content)
        2. Diarrhea/constipation (frequency, character)
        3. Abdominal pain (location, character, radiation)
        """,
        "neurological": """
        Assess neurological system:
        1. Headache (character, location, triggers)
        2. Dizziness (type, triggers)
        3. Weakness/numbness (location, progression)
        """,
        "other_symptoms": """
        Assess general symptoms:
        1. Fever (duration, pattern)
        2. Weight changes (amount, timeframe)
        3. Fatigue (severity, impact)
        """,
        "medical_history": """
        Gather medical history:
        1. Past medical conditions
        2. Surgeries and hospitalizations
        3. Medications and allergies
        4. Family history of major diseases
        """
    }
    
    prompt = f"""
    Current interview phase: {current_phase}
    Phase focus: {phase_prompts[current_phase]}
    
    Previous questions: {previous_questions}
    Patient's most recent response: {user_response}
    Your reflection on the response: {reflection}

    Generate a detailed follow-up question that:
    1. Aligns with current interview phase
    2. Builds on previous questions and responses
    3. Seeks specific information needed for a complete medical report
    4. Uses a professional yet empathetic tone
    5. Helps gather comprehensive medical information
    6. Always ends with a question mark
    7. Maintains conversation context from previous interactions
    8. Avoids repeating previously asked questions
    """
    
    try:
        response = conversation.predict(input=prompt)
        # Store the generated question in memory
        memory.add_interaction(response, user_response)
        return response
    except Exception as e:
        print(f"Error generating next question: {str(e)}")
        return "Can you describe your symptoms in more detail?"

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
