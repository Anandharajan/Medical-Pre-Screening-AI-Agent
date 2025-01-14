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
    
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=api_key,
            temperature=0.7,
            max_output_tokens=2048,
            convert_system_message_to_human=True
        )
    except Exception as e:
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
    You are a medical assistant conducting a pre-screening interview. Your role is to:
    1. Ask clear, specific questions to assess the patient's condition
    2. Maintain a professional yet empathetic tone
    3. Gather comprehensive medical information
    4. Identify potential urgent conditions
    5. Provide appropriate follow-up recommendations
    
    Always consider:
    - The patient's medical history
    - The context of previous questions
    - The need for accurate and complete information

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
    """Handle LLM interaction using LangChain's conversation chain"""
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
    
    try:
        response = conversation.predict(input=full_prompt)
        # Store the interaction in memory
        memory.add_interaction(prompt, response)
        return response
    except Exception as e:
        return f"An error occurred while processing your request: {str(e)}. Please try again."

def reflect(previous_question: str, user_response: str, reflection_prompt: str, api_key: str = None) -> str:
    """Reflect on the user's response using LangChain's memory"""
    conversation = create_conversation_chain(api_key)
    
    # Add conversation history to persistent memory
    memory.add_interaction(previous_question, user_response)
    
    try:
        response = conversation.predict(input=reflection_prompt)
        # Store the reflection in memory
        memory.add_interaction(reflection_prompt, response)
        return response
    except Exception as e:
        print(f"Error generating reflection: {str(e)}")
        return "Can you please provide more details about your symptoms and medical history?"

def generate_next_question(user_response: str, reflection: str, previous_questions: List[str], api_key: str = None) -> str:
    """Generate the next question using LangChain's chain"""
    conversation = create_conversation_chain(api_key)
    
    prompt = f"""
    Previous questions: {previous_questions}
    Patient's most recent response: {user_response}
    Your reflection on the response: {reflection}

    Generate a detailed follow-up question that:
    1. Builds on previous questions and responses
    2. Seeks specific information needed for a complete medical report
    3. Uses a professional yet empathetic tone
    4. Helps gather comprehensive medical information
    5. Always ends with a question mark
    6. Maintains conversation context from previous interactions
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