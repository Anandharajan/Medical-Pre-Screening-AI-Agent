import streamlit as st
from agent_utils import (
    reflect,
    generate_next_question,
    initial_question,
    generate_reflection_prompt,
    create_tools,
    handle_llm_interaction,
    get_model,
    validate_api_key
)
import json
import os

# Configure Streamlit page
st.set_page_config(page_title="Medical Pre-Screening", page_icon="🏥", layout="wide")

# Initialize session state with size limits
MAX_CONVERSATION_LENGTH = 50  # Maximum number of messages to store

if 'conversation' not in st.session_state:
    st.session_state.conversation = []
    st.session_state.conversation_size = 0
    
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
    
if 'patient_info' not in st.session_state:
    st.session_state.patient_info = {}
    
if 'tools' not in st.session_state:
    st.session_state.tools = create_tools()
    
if 'info_complete' not in st.session_state:
    st.session_state.info_complete = False

def add_to_conversation(role: str, content: str):
    """Add message to conversation with size limits"""
    if len(st.session_state.conversation) >= MAX_CONVERSATION_LENGTH:
        # Remove oldest messages
        st.session_state.conversation = st.session_state.conversation[1:]
        
    st.session_state.conversation.append({
        "role": role,
        "content": content
    })
    st.session_state.conversation_size = len(st.session_state.conversation)

# Initialize API key in session state
if 'google_api_key' not in st.session_state:
    st.session_state.google_api_key = None

# Check if API key is set and valid
if not st.session_state.google_api_key:
    with st.chat_message("assistant"):
        st.error("⚠️ API Key Required")
        st.write("""
        Welcome to the Medical Pre-Screening Assistant!
        
        To get started, you'll need a Google API key. Please:
        1. Go to [Google AI Studio](https://aistudio.google.com/)
        2. Create an API key
        3. Enter it below to begin
        
        Your API key will only be used for this session and won't be stored.
        """)
        
        st.warning("Without a valid API key, the assistant cannot generate responses.")
    
    if api_key := st.chat_input("Enter your Google API Key here..."):
        try:
            # Validate the API key
            if validate_api_key(api_key):
                st.session_state.google_api_key = api_key
                os.environ["GOOGLE_API_KEY"] = api_key
                st.rerun()
            else:
                st.error("Invalid API key - please check and try again")
        except Exception as e:
            st.error(f"Error validating API key: {str(e)}")
            st.session_state.google_api_key = None
    st.stop()

# Main application
st.title("Medical Pre-Screening Assistant")
st.write("Please answer the following questions to help us assess your condition.")

# Initialize patient info form state
if 'form_patient_info' not in st.session_state:
    st.session_state.form_patient_info = {
        "name": "",
        "age": 0,
        "gender": "Male"
    }

# Patient information form
if not st.session_state.patient_info:
    with st.form("patient_info_form"):
        st.subheader("Patient Information")
        name = st.text_input("Full Name", value=st.session_state.form_patient_info["name"], key="form_name")
        age = st.number_input("Age", min_value=0, max_value=120, value=st.session_state.form_patient_info["age"], key="form_age")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(st.session_state.form_patient_info["gender"]), key="form_gender")
        
        if st.form_submit_button("Submit"):
            st.session_state.form_patient_info = {
                "name": st.session_state.form_name,
                "age": st.session_state.form_age,
                "gender": st.session_state.form_gender
            }
            st.session_state.patient_info = st.session_state.form_patient_info
            
            # Initialize conversation with AI introduction
            with st.chat_message("assistant", avatar="🤖"):
                st.write(f"Hello {st.session_state.patient_info['name']}, I'm an Agentic AI Medical Assistant.")
                st.write("I'll be conducting a medical pre-screening interview to assess your condition.")
                st.write("I need to ask you about 6-8 questions regarding your symptoms and medical history before I can provide a proper assessment.")
                st.write("Let's begin with your primary concern - what has brought you in today?")
            
            # Get initial question using LangChain
            initial_response = handle_llm_interaction(
                f"""You are an Agentic AI Medical Assistant conducting a focused pre-screening interview.
                The patient is {st.session_state.patient_info['name']},
                a {st.session_state.patient_info['age']} year old {st.session_state.patient_info['gender'].lower()}.
                You already have this basic information, so do not ask for it again.
                Begin by asking relevant medical questions to assess their condition.
                Start with questions about their primary symptoms and medical history.""",
                st.session_state.tools,
                api_key=st.session_state.google_api_key
            )
            
            # Extract just the question text from JSON response
            try:
                if isinstance(initial_response, str) and initial_response.startswith("[{"):
                    question_data = json.loads(initial_response)
                    if isinstance(question_data, list) and len(question_data) > 0:
                        question_text = question_data[0].get("question", initial_response)
                    else:
                        question_text = initial_response
                else:
                    question_text = initial_response
            except json.JSONDecodeError:
                question_text = initial_response
                
            add_to_conversation("assistant", question_text)
            st.session_state.current_question = question_text
            
            st.rerun()
    st.stop()  # Stop execution until patient info is collected
else:
    # Initialize conversation if empty
    if not st.session_state.conversation:
        # Use agent for initial medical assessment
        initial_prompt = f"""
        You are a medical assistant conducting a focused pre-screening interview.
        The patient is {st.session_state.patient_info['name']},
        a {st.session_state.patient_info['age']} year old {st.session_state.patient_info['gender'].lower()}.
        You already have this basic information, so do not ask for it again.
        Begin by asking relevant medical questions to assess their condition.
        Start with questions about their primary symptoms and medical history.
        """
        initial_response = handle_llm_interaction(
            initial_prompt,
            st.session_state.tools,
            api_key=st.session_state.google_api_key
        )
        add_to_conversation("assistant", initial_response)
        st.session_state.current_question = initial_response

# Custom icons
DOCTOR_ICON = "👨‍⚕️"
PATIENT_ICON = "👤"

def format_agent_response(response: str) -> str:
    """Format agent response with detailed logs and collapsible sections"""
    # If this is a simple greeting message, return it as-is
    if "welcome to your medical pre-screening" in response.lower():
        return response
        
    # Handle final answers
    if "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[1].strip()
        return f"**{final_answer}**"
        
    # Format all assistant questions in bold
    if any(phrase in response.lower() for phrase in ["question:", "can you", "do you", "have you", "when did"]):
        # Split into lines and format each question line
        lines = response.split("\n")
        formatted_lines = []
        
        for line in lines:
            if any(phrase in line.lower() for phrase in ["question:", "can you", "do you", "have you", "when did"]):
                # Remove any existing markdown formatting
                clean_line = line.replace("**", "").strip()
                # Format the entire question line in bold
                formatted_lines.append(f"**{clean_line}**")
            else:
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)
    
    # Handle tool responses
    if "[Tool Response]" in response:
        tool_response = response.split("[Tool Response]")[1].strip()
        with st.expander("Detailed Analysis"):
            st.markdown(tool_response)
        return response.split("[Tool Response]")[0].strip()
    
    return response

# Display conversation history
for message in st.session_state.conversation:
    icon = DOCTOR_ICON if message["role"] == "assistant" else PATIENT_ICON
    with st.chat_message(message["role"], avatar=icon):
        formatted_content = format_agent_response(message["content"])
        st.write(formatted_content)

# Get user response
user_input = None
if user_input := st.chat_input("Type your response here..."):
    # Add user response to conversation
    add_to_conversation("user", user_input)
    
    # Check if we have collected enough information - require at least 6 complete Q&A pairs
    if len(st.session_state.conversation) >= 12 and not st.session_state.info_complete:
        # Verify we have at least 6 assistant questions and 6 user responses
        assistant_messages = [msg for msg in st.session_state.conversation if msg["role"] == "assistant"]
        user_messages = [msg for msg in st.session_state.conversation if msg["role"] == "user"]
        
        if len(assistant_messages) >= 6 and len(user_messages) >= 6:
            st.session_state.info_complete = True
    
    # Use agent to handle the conversation
    conversation_history = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in st.session_state.conversation
    )
    
    # Get previous questions and responses for context
    previous_questions = [msg['content'] for msg in st.session_state.conversation if msg['role'] == 'assistant']
    previous_responses = [msg['content'] for msg in st.session_state.conversation if msg['role'] == 'user']
    
    # Generate reflection on previous response
    reflection = reflect(
        previous_question=previous_questions[-1] if previous_questions else None,
        user_response=previous_responses[-1] if previous_responses else None,
        reflection_prompt=generate_reflection_prompt(
            user_response=previous_responses[-1] if previous_responses else None,
            previous_questions=previous_questions,
            api_key=st.session_state.google_api_key
        ),
        api_key=st.session_state.google_api_key
    )
    
    # Generate next question or analysis based on conversation state
    if not st.session_state.info_complete:
        # Generate next question dynamically
        agent_response = generate_next_question(
            user_response=previous_responses[-1] if previous_responses else None,
            reflection=reflection,
            previous_questions=previous_questions,
            api_key=st.session_state.google_api_key
        )
    else:
        # Generate structured report using report generator
        from report_utils import MedicalReportGenerator
        
        report_generator = MedicalReportGenerator(
            conversation_history=st.session_state.conversation,
            patient_info=st.session_state.patient_info
        )
        
        # Generate and safely format the report
        report = report_generator.generate_report()
        formatted_report = report_generator.safe_format_report(report)
        
        # Add report to conversation
        agent_response = f"""
        ## Medical Pre-Screening Report
        
        {formatted_report}
        
        **Please review this report carefully and consult with your healthcare provider.**
        """
    
    # Format and add agent response to conversation
    formatted_response = agent_response
    
    st.session_state.conversation.append({
        "role": "assistant",
        "content": formatted_response
    })
    st.session_state.current_question = agent_response
    
    # Rerun to update the conversation display
    st.rerun()