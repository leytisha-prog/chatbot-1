import time 
import uuid

import streamlit as st
from openai import OpenAI 
from supabase import Client, create_client 


#------ PAGE CONFIGURATION -----------

st.set_page_config(
    page_title="Programming for Data Analytics Course Assistant",
    page_icon=";material/icon_code:",
    layout="centered",
)

EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_COUNT = 5
RETRIEVAL_THRESHOLD = 0.30



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

@st.cache_resource
def get_supabase_client() -> Client:
    """Create and cache one Supabase client."""

    supabase_settings = st.secrets["connections"]["supabase"]

    return create_client(
        supabase_settings["SUPABASE_URL"],
        supabase_settings["SUPABASE_SECRET_KEY"],
    )

supabase = get_supabase_client() 

def create_question_embedding(
    question: str,
    openai_client: OpenAI,    
) -> list[float]:
    """
    Convert the student's question into a numerical embedding.
    """

    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
    )

    return response.data[0].embedding


def retrieve_course_chunks(
    question: str,
    openai_client: OpenAI,
    supabase_client: Client,
) -> list[dict]:
    """
    Retrieve course chunks that are semantically related
    to the student's question. 
    """

    question_embedding = create_question_embedding(
        question=question,
        openai_client=openai_client,
    )

    response = supabase_client.rpc(
        "match_course_chunks",
        {
            "query_embedding": question_embedding,
            "match_count": RETRIEVAL_COUNT,
            "match_threshold": RETRIEVAL_THRESHOLD,
        },
    ).execute()

    return response.data or []



#---Format the Retrieved Material#

def format_course_context(chunks: list[dict]) -> str:
    """
    Format retrieved database chunks for the chatbot prompt.
    """

    if not chunks:
        return "No relevant course material was retrieved."
    
    context_parts = []

    for number, chunk in enumerate(chunks, start=1):
        source_title = chunk.get("source_title") or "Course material"
        section_heading = chunk.get("section_heading") or "Unspecified section"
        content = chunk.get("content") or ""

        context_parts.append(
            f"""
COURSE SOURCE {number}
Title: {source_title}
Section: {section_heading}
Similarity: {chunk.get("similarity", 0):.3f}

Content:
{content}
""".strip()
        )
    
    return "\n\n---\n\n".join(context_parts)



#--- Create a source list for loggins
def prepare_retrieved_sources(chunks: list[dict]) -> list[dict]:
    """
    Create a concise version of the retrieval results
    for Supabase logging.
    """

    sources = []

    for chunk in chunks:
        sources.append(
            {
                "chunk_id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "source_title": chunk.get("source_title"),
                "source_url": chunk.get("source_url"),
                "section_heading": chunk.get("section_heading"),
                "similarity": chunk.get("similarity"),
            }
        )

    return sources 


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

Use language appropriate for students who may be beginners or intermediates
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

Keep responses focused and usually under 400 words unless the student
requests more detail.
"""

def get_system_prompt () -> str:
    """
    Return the prompt for the active study condition.

    The metacognitive prompt will be added after the control chatbot
    and shared infrastructure are working reliably.

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

def save_chat_log(
    user_message,
    assistant_message,
    turn_number,
    response_time_ms,
    retrieved_sources,
) -> bool:
    """Save one completed student-assistant exchange to Supabase."""

    log_record = {
        "participant_id": st.session_state.participant_id,
        "condition": CHATBOT_CONDITION,
        "session_id": st.session_state.session_id,
        "assignment_name": st.session_state.selected_assignment,
        "turn_number": turn_number,
        "user_message": user_message,
        "assistant_message": assistant_message,
        "model_name": MODEL_NAME,
        "response_time_ms": response_time_ms,
        "retrieved_sources": retrieved_sources,
    }

    try:
        (
            supabase
            .table("chat_logs")
            .insert(log_record)
            .execute()
        )

        return True
    
    except Exception as error:
        print(f"Supabase logging error: {error}")
        return False

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


def stream_assistant_response(course_context: str):
    """Generate and stream an assistant response."""

    rag_instruction = f"""
Use the retrieved IST 356 course material below when answering
the student.

Rules:
1. Prioritize the retrieved course material.
2. Do not invent course materials, deadlines, or policies.
3. If the material does not fully answer the question, say so.
4. Explain the answer at an introductory Python and data analytics level.
5. For practice exercises, guide the student through the
   reasoning instead of immediately giving the complete answer.

RETRIEVED COURSE MATERIAL:

{course_context}
"""

    combined_instructions = (
        get_system_prompt()
        + "\n\n"
        + rag_instruction
    )
    
    conversation_input = prepare_conversation_input()

    with client.responses.stream(
        model=MODEL_NAME,
        instructions=combined_instructions,
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
- interpreting analytical results.

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

    # -------- RETRIEVE RELEVANT COURSE MATERIAL --------------

    try:
        retrieved_chunks = retrieve_course_chunks(
            question=clean_prompt,
            openai_client=client,
            supabase_client=supabase,
        )

        course_context = format_course_context(
            retrieved_chunks
        )

        retrieved_sources = prepare_retrieved_sources(
            retrieved_chunks
        )

    except Exception as error:
        retrieved_chunks = []
        retrieved_sources = []

        course_context = (
            "Course retrieval was temporarily unavailable."
        )

        print(f"Course retrieval error: {error}")


    # Generate and display the assistant's response.
    with st.chat_message("assistant"):

        response_start_time = time.perf_counter()

        try:
            assistant_response = st.write_stream(
                stream_assistant_response(
                    course_context=course_context
                )
            )
        except Exception as error:
            assistant_response = (
                "I could not generate a response. Please try again."
                "If the problem continues, contact the researcher."
            )

            st.error(assistant_response)

            # During development, this appears only in the termina.
            print(f"OpenAI error: {error}")

        response_end_time = time.perf_counter()

        response_time_ms = int(
            (response_end_time - response_start_time) * 1000
        )


        # Save the completed assistant response in session history.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_response,
            }
        )
    
        turn_number = sum(
            1
            for message in st.session_state.messages
            if message["role"] == "user"
        )

        log_saved = save_chat_log(
            user_message=clean_prompt,
            assistant_message=assistant_response,
            turn_number=turn_number,
            response_time_ms=response_time_ms,
            retrieved_sources=retrieved_sources,
        )

        if not log_saved:
            st.warning(
                "Your answer was generated, but the interaction"
                "could not be saved. Please notify the researcher."
            )

        # Display the retrieved sources during development.
        if retrieved_chunks:
            with st.expander("Course sources used"):

                for chunk in retrieved_chunks:

                    source_title = (
                        chunk.get("source_title")
                        or "Course material"
                    )

                    section_heading = chunk.get(
                        "section_heading"
                    )

                    source_url = chunk.get("source_url")

                    similarity = chunk.get(
                        "similarity",
                        0,     
                    )

                    st.markdown(f"**{source_title}**")

                    if section_heading:
                        st.caption(
                            f"Section: {section_heading}"
                        )

                    st.caption(
                        f"Similarity score: {similarity:.3f}"
                    )

                    if source_url:
                        st.markdown(
                            f"[Open source page]({source_url})"
                        )

                