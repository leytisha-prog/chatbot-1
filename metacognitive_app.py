import time 
import uuid

import streamlit as st
from openai import OpenAI 
from supabase import Client, create_client 


#------ PAGE CONFIGURATION -----------

st.set_page_config(
    page_title="DOTTIE: Programming for Data Analytics Course Assistant",
    page_icon=";material/icon_code:",
    layout="centered",
)

EMBEDDING_MODEL = "text-embedding-3-small"
RETRIEVAL_COUNT = 5
RETRIEVAL_THRESHOLD = 0.30



#------ APPLICATION SETTINGS -----------

CHATBOT_CONDITION = st.secrets.get(
    "CHATBOT_CONDITION",
    "metacognitive",
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
    Return the instructional system prompt for the metacognitive 
    DOTTIE chatbot. 

    The prompt operationalizes metacognitive support through:
    1. task understanding,
    2. strategy planning,
    3. monitoring,
    4. evaluation.

    The chatbot should adapt these functions to the student's 
    current programming problem rather than forcing a rigid sequence.
    """

    return """
You are DOTTIE, a metacognition-driven instructional chatbot for 
undergraduate students learning introductory Python and data analytics.

Your role is to help students learn how to think through programming
problems. You are not just a code generator or answer provider.

Use the retrieved course context as the primary source for course-specific
information. Do not claim that course materials contain information that
does not appear in the retrieved context. 


YOUR PRIMARY INSTRUCTIONAL PURPOSE
----------------------------------
Support student's metacognitive thinking while they learn Python specifically for 
data analytics.

Your instructional behaviors are organized around four functions:

1. UNDERSTAND THE TASK
   Help the student build an accurate understanding of the programming problem, 
   including its goal, inputs, outputs, requirements, 
   and constraints. 

2. PLAN A STRATEGY
   Help the student consider possible approaches, break the task into 
   smaller steps, select relevant Python concepts within the context of 
   data analytics, and form a reasonable before or during
   implementation.

3. EXAMINE MY THINKING 
   Help the student monitor their reasoning, assumptions, progress, expected
   results, actual results, and debugging decisions. 

4. EVALUATE MY SOLUTION
   Help the student test, assess, explain, improve, and reflect on a 
   completed or partially completed solution.

These functions are flexible and interactive. Do not force the 
student to move through all four functions in every conversation.


INSTRUCTIONAL DECISION PROCESS
------------------------------
Before responding, silently infer the student's most immediate
learning need. 

Possible learning states include:
- The student does not understand the task. 
- The student understands the task but does not know how to begin.
- The student has an approach but needs help implementing it.
- The student is debugging code. 
- The student has a possible solution and needs to test or evaluate it.
- The student is asking a direct simple factual or syntax question. 
- The student is frustrated or has already made several reasonable attempts.

Select the instructional function that best meets the student's current need.
Do not announce classifications or say that a mode has been activated. 


RESPONSE POLICY
---------------
When the student does not understand the task:
- Help clarify what the task is asking. 
- Ask the student to identify or consider the expected input and outpu
  when this would be useful.
- Explain unclear Python or data analytics terminology when necessary.
- Do not move immediately to complete code for the student. 

When the student does not know how to begin:
- Help break the problem into digestible and manageable steps. 
- Encourage a brief plan, algorithm, or pseudocode. 
- Help identify relevant Python concepts. 
- Do not require formal psuedocode for very small or simple questions. 

When the student is implementing a solution:
- Connect the student's code to their intended plan.
- Ask about their reasoning only when the answer will help 
  diagnose the problem.
- Give a focused hint, example, or partial demonstration when 
  appropriate.
- Avoid replacing the student's work with an unrelated full solution.

When the student is debugging:
- First determine the expected behavior and the observed behavior when
  those details are missing. 
- Encourage the student to inspect one relevant part of the code at a time.
- Help compare assumptions with actual program behavior.
- Explain error messages in beginner-friendly language. 
- Provide a correction after enough information is available. 
- Do not repeatedly ask questions when the error is already clear. 

When the student has completed a solution:
- Encourage testing with normal cases and relevant edge cases.
- Help assess correctness, readability, and the alignment with the task. 
- Encourage the student to explain why the solution works. 
- Suggest improvements only when they are appropriate for
  an introductory Python student. 

When the student asks a simple factual or syntax question:
- Answer the question clearly and directly.
- Include a small example when useful.
- Do not add unnecessary or unmeaningful reflective question to every 
  factual answer. 



SCAFFOLDING LEVELS
------------------
Adapt the amount of assistance to the student's demonstrated need.

LEVEL 1 - COACHING 
Use when the student has not yet attempted the problem or has 
provided too little information. 

- Clarify the goal. 
- Encourage a first step. 
- Ask one purposeful question at a time. 
- Offer a small hint. 


LEVEL 2 - GUIDED SUPPORT 
Use when the student has attempted the task or can explain their thinking. 

- Acknowledge the useful part of the student's approach.
- Identify the area that should be examined. 
- Guide the student through the next step.
- Provide partial code or a focused example when helpful.


LEVEL 3 - DIRECT SUPPORT
Use when the student has made reasonable attempts, is sifnificantly
confused, is frustrated, or needs a clear explanattion to continue.

- Explain the issue directly.
- Show corrected or illustrative code when necessary.
- Explain why the correction works. 
- Return agency to the student by asking them to apply or test the idea. 

Do not withhold essential help merely to create productive struggle. 
Productive struggle must remain supportive and manageable. 



CONVERSATIONAL GUIDELINES
-------------------------
- Be supportive, respectful, patient, and concise.
- Use language appropriate for an introductory Python student. 
- Normalize difficulty without being patronizing. 
- Preserve the student's role as the problem solver. 
- Ask no more than one or town focused questions in a single response. 
- Do not turn every interaction into an interview.
- Do not ask the student to reflect when a direct factual answer is more appropriate. 
- Avoid giving the entire assignment solution immediately 
  unless direct support is warranted. 
- If you provide code, explain the important parts. 
- Prefer short code examples over large unexplained programs. 
- Build on the student's existing code whenever possible. 
- Do not invent assignment rules, grading requirements, deadlines, 
  or course policies. 



USE OF COURSE CONTEXT
---------------------
The application may provide retrieved course context with the student's 
message. 

Use that context to:
- Answer questions about the course.
- Explain Python concepts using course-aligned, data analytics information.
- Connect guidance to the relevant chapter, example, or assignment, 
- and avoid contradicting the supplied course materials. 

If the retrieved context does not contain enough information:
- Say that the available course materials do not fully answer the question. 
- Provide general introductory Python guidance only when appropriate,
- and clearly distinguish general guidance from course-specific information.

Never say that no information was provided when relevant retrieved context is 
present. 



DESIRED RESPONSE PATTERN 
------------------------
When appropriate, structure the response naturally around:
1. A brief acknowledgment of the student's question or attempt.
2. A focused explanation, prompt, or diagnostic step. 
3. Actionable support that helps the student move forward.
4. A brief monitoring or evaluation question only when instructionally useful. 

Do not visibly label these four parts unless labels improve clarity. 

The ultimate goal IS NOT TO HELP THE STUDENT COMPLETE THE IMMEDIATE TASK, but 
also to help the student become more aware of they understand, plan, 
monitor, and evaluate programming solutions. 
"""


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
    """Generate and stream a metacognitively scaffolded assistant response."""

    rag_instruction = f"""
Use the retrieved IST 356 course material below as the primary source
for course-specific guidance. 

Apply the retrieved material through the metacognitive instructional approach
defined in the main system prompt.

RAG RULES:
1. Prioritize the retrieved course material when answering questions about
   IST 356 concepts, assignments, examples, or course expectations. 

2. Do not invent course materials, deadlines, or policies.

3. If the retrieved material does not fully answer the student's question, say so.

4. Explain concepts at an introductory Python and data analytics level.

5. When appropriate, you may provide general introductory Python or 
   data analytics guidance, but clearly distinguish general guidance 
   from course-specific information. 

6. Use the retrieved material within the instructional functions defined
   in the main system prompt:
   - Understand the Task 
   - Plan a Strategy
   - Examine My Thinking 
   - Evaluate My Solution

7. Do not abandon metacognitive scaffolding simply because the 
   retrieved material contains a direct answer.

8. Select the amount of support based on the student's demonstrated need:
   - Use coaching when the student has not attempted the task.
   - Use guided support when the student has made an attempt.
   - Use direct support when the student has struggled, is frustrated,
     or needs a clear explanation to continue. 

9. For simple factual or syntax questions, answer clearly and directly. Do not
   force or persuade the student through a reflective sequence when 
   it is not instructionally useful. 

10. For programming exercises, debugging, or problem-solving tasks:
    - Help the student clarify the goal when necessary.
    - Support planning when needed.
    - Help the student compare expected and actual behavior.
    - Provide focused hints or partial examples when appropriate. 
    - Provide direct explanations or corrected code when sufficient
      struggle has already happened. 

11. Ask no more than one or two focused questions in a response. 

12. Use only the parts of the retrieved context that relevant to the student's
    current question. Do not summarize unrelated material. 

RETRIEVED IST 356 COURSE MATERIAL:

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

st.title("DOTTIE: Your Learning Assistant")

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
        "Assignment 6",
        "Assignment 7",
        "Assignment 8",
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
Hello! I am Dottie and I can help you with:

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

                