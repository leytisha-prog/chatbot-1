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

def get_system_prompt () -> str:
    """
    Return the prompt for the active study condition.

    The metacognitive prompt will be added after the control chatbot
    and shared infrastructur are working reliably.

    """
    if CHATBOT_CONDITION == "non_metacognitive":
        return CONTROL_SYSTEM_PROMPT
    
    # Temporary safeguard until the metacognitive prompt is added.
    return CONTROL_SYSTEM_PROMPT



# ------- SESSION STATE -------------------

def initialize_session_state() -> None:
    """Initialize values that must persist during the browser
    session."""

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "participant_id" not in st.session_state:
        st.session_state.participant_id = ""

    if "selected_assignment" not in st.session_state:
        st.session_state.selected_assignment = "General course support"

initialize_session_state()



# ------- HELPER FUNCTIONS --------------------

def reset_conversation() -> None:
    """Clear messages and create a new anonymous session ID."""
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

def prepare_conversation_input() -> list[dict]:
    """
    Convert Streamlit conversation history into the format expected
    by the OpenAI Response API.
    """

    conversation= []

    assignment_context = (
        f"The student selected this assignment of context: "
        f"{st.session_state.selected_assignment}"
    )

    conversation.append(
        {
            "role": "user",
            "content": assignment_context,
        }
    )

    for message in st.session_state.messages:
        conversation.append(
            {
                "role": message["role"],
                "content": message["content"], 
            }
        )
    
    return conversation


def stream_assistant_response():
    """Generate and stream an assistant response."""

    conversation_input = prepare_conversation_input()

    with client.responses.stream(
        model=MODEL_NAME,
        instructions=get_system_prompt(),
        input=conversation_input,
    ) as stream:
        
        for event in stream:
            if event.type == "response.output_text.delta":
                yield event.delta


# ----- USER INTERFACE --------------------

st.title("Programming for Data Analytics Learning Assistant")

st.caption(
    "Ask questions about Python, data analysis, errors, code,"
    "and your course assignments."
)

with st.sidebar:
    st.header("Session Information")

    participant_id = st.text_input(
        "Participant ID",
        value=st.session_state.participant_id,
        placeholder="Example: P001",
        help=(
            "Enter the anonymous participant code assigned by"
            "the researcher. Do not enter your name."
        ),
    )

    st.session_state.participant_id = participant_id.strip()

    assignment_options = [
        "General course support",
        "Assignment 1",
        "Assignment 2",
        "Assignment 3",
        "Assignment 4",
        "Assignment 5",
        "Final Project",
    ]

    selected_assignment = st.selectbox(
        "What are you working on?",
        assignment_options,
        index=assignment_options.index(
            st.session_state.selected_assignment
        ),
    )

    st.session_state.selected_assignment = selected_assignment

    st.divider()

    st.write(
        f"**Session:**"
        f"`{st.session_state.session_id[:8]}`"
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        reset_conversation()
        st.rerun()

    st.divider()

    st.info(
        "Do not enter your name, student ID, email address," 
        "or other personal information in the chatbot."
    )


# ------ PARTICIPATION VALIDATION --------------------

if not st.session_state.participant_id:
    st.warning(
        "Enter your assigned participant ID in the sidebar"
        "before beginning."
    )
    st.stop()



#----- DISPLAY EXISTING CHAT HISTORY ------------------

if not st.session_state.messages:
    with st.chat_message("assistant"):
       st.markdown(
            """
Hello! I can help you with:

- understanding Python concepts,
- interpreting error messages,
- debugging code,
- selecting data-analysis methods, and 
- interpeting analytical results.

What are you working on?
"""
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])



#----- RECEIVE A NEW STUDENT MESSAGE -------------------

student_prompt = st.chat_input(
    "Ask a question about Python or data analytics"
)

if student_prompt:

    clean_prompt = student_prompt.strip()

    if not clean_prompt:
        st.stop()

    # Save and display the student's message.
    st.session_state.messages.append(
        {
            "role": "user",
            "content": clean_prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(clean_prompt)

    # Generate and display the assistant's response.
    with st.chat_message("assistant"):
        try:
            assistant_response = st.write_stream(
                stream_assistant_response()
            )
        except Exception as error:
            assistant_response = (
                "I could not generate a response. Please try again."
                "If the problem continues, contact the researcher."
            )

            st.error(assistant_response)

            # During development, this appears only in the termina.
            print(f"OpenAI error: {error}")


        # Save the completed assistant response in session history.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )