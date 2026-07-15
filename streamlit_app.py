import uuid

import streamlit as st
from openai import OpenAI 


#------ PAGE CONFIGURATION -----------

st.set_page_config(
    page_title="Programming for Data Analytics Course Assistant",
    page_icon=";material/icon_code:",
    layout="centered",
)


#------ APPLICATION SETTINGS -----------

CHATBOT_CONDITION = st.secrets.get(
    "CHATBOT_CONDITION",
    "non_metacognitive",
)

MODEL_NAME = st.secrets.get(
    "MODEL_NAME",
    "gpt-4.1-mini",
)

# ------ OPENAI CLIENT -------------

@st.cache_resource
def get_openai_client() -> OpenAI:
    """Create and cache one OpenAI client."""
    return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

client = get_openai_client()


# ------- SYSTEM PROMPT ----------------

CONTROL_SYSTEM_PROMPT = """
You are a supportive learning assistant for students taking a data analytics course that uses Python.

Your responsibilities are to:

1. Explain Python and data analytics concepts clearly.
2. Help students understand error messages.
3. Assist students with debugging their own code.
4. Explain code one step at a time.
5. Provide small and relevant code examples. 
6. Help students interpret data-analysis results.
7. Encourage students to check that the response matches their assignment requirements.

Use language appropriate for students who may be beginners or intermediats
in Python.

Do not claim to have accessed course materials unless those materials
have actually been provided in the conversation or retrieved by the
application. 

Do not invent assignment instructions, dataset details, grading rules,
or course requirements.

When information is missing, clearly say what information is needed.

Do not use systematic metacognitive scaffolding. Do not routinely ask students 
to reflect on their thinking, evaluate their confidence, develop a plan, or
explain their reasoning before receiving assistance. 

Keep responses focuses and usually under 400 words unless the student
requests more detail.
"""
