# Medical Pre-Screening AI Agent

medical-pre-screening-app is a Python-based repository for developers and healthcare professionals to understand and implement AI-powered medical pre-screening systems. It demonstrates how to build a medical assistant agent using Streamlit and Google's Generative AI.

## Important Warning

**This application is for educational and demonstration purposes only.**

- **Not a Medical Device**: This application is not a certified medical device and should not be used for actual medical diagnosis or treatment.
- **No Medical Advice**: The information provided by this application should not be considered as professional medical advice.
- **Accuracy Limitations**: The AI model may provide inaccurate or incomplete information. Always consult a qualified healthcare professional for medical concerns.
- **Data Privacy**: While we implement security measures, sensitive health information should not be entered into this application.
- **Emergency Situations**: In case of medical emergencies, contact your local emergency services immediately.

By using this application, you acknowledge and accept these limitations and agree to use it at your own risk.

[![Watch the video](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID)

## Whom is this for?
This is for healthcare developers and AI engineers who want to understand how to build AI-powered medical pre-screening systems. This is not a full-fledged medical system but rather a demonstration of how to implement such a system using AI agents.

## What can you do with this?
The repository contains a medical pre-screening agent implementation that can:
- Conduct interactive medical interviews
- Generate relevant medical questions
- Analyze patient responses
- Provide follow-up questions based on symptoms
- Maintain patient history

## Getting Started
1. Clone the repository:
```bash
git clone https://github.com/yourusername/medical-pre-screening.git
cd medical-pre-screening
```

2. Set up the environment:
```bash
conda env create -f environment.yml
conda activate medical-pre-screening
```

3. Create a `.env` file by copying `.env.template`:
```bash
cp .env.template .env
```

4. Add your Google API key to the `.env` file

5. Install requirements:
```bash
pip install -r requirements.txt
```

6. Run the application:
```bash
streamlit run app.py
```

## Architecture
The medical pre-screening agent follows this architecture:

<img src="assets/medical_agent_architecture.png" alt="Medical Agent Architecture" width="50%"/>

The system consists of:
1. **Streamlit Interface**: Handles user interaction and data collection
2. **Medical Agent**: Core AI logic for interview management
3. **Google Generative AI**: Provides the language model capabilities
4. **Medical Knowledge Base**: Contains medical guidelines and protocols

## Agent Implementation
The main agent logic is implemented in `agent_utils.py`:

```python
class MedicalAgent:
    def __init__(self, model, tools):
        self.model = model
        self.tools = tools
        
    def conduct_interview(self, patient_info):
        # Implements medical interview logic
        
    def analyze_responses(self, responses):
        # Analyzes patient responses
        
    def generate_followup(self, analysis):
        # Generates follow-up questions
        
    def run(self):
        # Runs the complete interview process
```

## Example Usage
The main application in `app.py` demonstrates how to use the medical agent:

```python
from agent_utils import MedicalAgent
import streamlit as st

# Initialize agent
agent = MedicalAgent(model="gemini-pro", tools=["medical_knowledge"])

# Streamlit interface
st.title("Medical Pre-Screening Assistant")
patient_info = st.text_input("Enter patient information")
if st.button("Start Interview"):
    agent.conduct_interview(patient_info)
```

## Going Deeper
1. [AI in Healthcare: Challenges and Opportunities](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6616181/)
2. [Generative AI for Medical Applications](https://arxiv.org/abs/2302.07377)
3. [Medical AI Ethics and Best Practices](https://www.who.int/publications/i/item/9789240029200)

## Contributing
We welcome contributions! Please focus on:
1. Additional medical screening scenarios
2. Improved patient data handling
3. Enhanced medical knowledge integration
4. Better error handling and validation

Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Commit your changes
4. Open a pull request

## License
MIT License - See [LICENSE](LICENSE) for details