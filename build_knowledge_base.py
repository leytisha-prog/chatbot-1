import re
import sys
import time
import tomllib
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from openai import OpenAI 
from supabase import Client, create_client


# ------ CONFIGURATION ----------------------------
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
REQUEST_TIMEOUT = 30

@dataclass
class CourseSource:
    title: str
    url: str
    source_type: str
    topic: str
    contains_solution: bool = False 

APPROVED_SOURCES = [
    CourseSource(
        title="IST 356 Syllabus",
        url="https://mafudge.github.io/ist356/syllabus.html",
        source_type="syllabus",
        topic="Course policies and requirements",
    ),
    CourseSource(
        title="Python Programming for Data Science: Python Basics",
        url=(
            "https://www.tomasbeuzen.com/"
            "python-programming-for-data-science/"
            "chapters/chapter1-basics.html"
        ),
        source_type="textbook_chapter",
        topic="Python basics",
    ),
    CourseSource(
        title="Python Basics Practice Exercises",
        url=(
            "https://www.tomasbeuzen.com/"
            "python-programming-for-data-science/"
            "practice-exercises/chapter1-basics-practice.html"
        ),
        source_type="textbook_exercise",
        topic="Python basics practice",
        contains_solution=True,
    ),
]


# ------- CLIENTS --------------------------------------

def load_secrets() -> dict:
    """Load the same secrets used by the Streamlit application."""

    with open(".streamlit/secrets.toml", "rb") as file:
        return tomllib.load(file)


def create_clients() -> tuple[OpenAI, Client]:
    secrets = load_secrets ()

    openai_client = OpenAI (
        api_key=secrets["OPENAI_API_KEY"]
    )

    supabase_client = create_client(
        secrets["SUPABASE_URL"],
        secrets["SUPABASE_SECRET_KEY"],
    )
    
    return openai_client, supabase_client


# ----- WEB CONTENT EXTRACTION ----------------------------

def download_page(url: str) -> str:
    """Download one approved webpage."""

    headers = {
        "User-Agent": (
            "IST356-Educational-RAG/1.0"
            "(research course-content ingestion)"
        )
    }

    response = requests.get (
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
    )

    response.raise_for_status()
    return response.text

def extract_page_sections(html: str) -> list[dict]:
    """
    Extract headings, paragraphs, lists and code blocks.
    
    Each returned item contains a section heading and its text.
    """

    soup = BeautifulSoup(html, "html.parser")

    for unwanted in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "form",
            "noscript",
        ]

    ):
        unwanted.decompose()

    main_content = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_="content")
        or soup.body
    )

    if main_content is None:
        return []
    
    sections = []
    current_heading = "General"
    current_parts = []

    useful_tags = main_content.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "p",
            "li",
            "pre",
            "code",
            "table",
        ]
    )

    for element in useful_tags:
        tag_name = element.name

        if tag_name in {"h1", "h2", "h3", "h4"}:
            if current_parts:
                sections.append(
                    {
                        "heading": current_heading,
                        "text": "\n".join(current_parts),
                    }
                )
            
            current_heading = element.get_text(
                " ",
                strip=True
            )
            current_parts = []
            continue

        text = element.get_text(
            "\n" if tag_name in {"pre", "code"} else " ",
        )

        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if not text:
            continue

        if tag_name in {"pre", "code"}:
            text = f"CODE EXAMPLE:\n{text}"

        current_parts.append(text)
    
    if current_parts:
        sections.append(
            {
                "heading": current_heading,
                "text": "\n".join(current_parts),
            }
        )
    return sections




# ----- CHUNKING --------------------------------------
def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping character-based chunks."""

    cleaned_text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not cleaned_text:
        return []
    
    chunks = []
    start = 0

    while start < len(cleaned_text):
        end = min(start + chunk_size, len(cleaned_text))

        if end < len(cleaned_text):
            paragraph_break = cleaned_text.rfind(
                ". ",
                start, 
                end,
            )

            sentence_break = cleaned_text.rfind(
                ". ",
                start,
                end,
            )

            preferred_break = max(
                paragraph_break,
                sentence_break,
            )

            if preferred_break > start + int(chunk_size * 0.60):
                end = preferred_break + 1

        chunk = cleaned_text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)

        if end >= len(cleaned_text):
            break

        start = max(end - overlap, start + 1)
    
    return chunks


# ---- EMBEDDINGS -----------------------------------------
def create_embedding(
    client: OpenAI,
    text: str,  
) -> list[float]:
    """Create one embedding vector."""

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding



# ------- DATABASE FUNCTIONS ---------------------------------#

def source_already_exists(
        supabase: Client,
        source_url: str,        
) -> bool:
    """Check whether a source has already been ingested."""

    response = (
        supabase
        .table("course_documents")
        .select("id")
        .eq("source_url", sorted)
        .limit(1)
        .execute()
    )
 
    return bool(response.data)

def insert_document (
    supabase: Client,
    source: CourseSource,
) -> int:
    """Create a document record and return its database ID."""

    record = {
        "filename": source.title,
        "title": source.title,
        "source_type": source.source_type,
        "source_url": source.url,
        "category": source.topic,
        "topic": source.topic,
    }

    response = (
        supabase
        .table("course_documents")
        .insert(record)
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            f"Document record was not created: {source.title}"
        )
    return response.data[0]["id"]

def insert_chunk(
    supabase: Client,
    document_id: int,
    chunk_number: int,
    chunk_text: str,
    embedding: list[float],
    source: CourseSource,
    section_heading: str,
) -> None:
    """Insert one course-material chunk."""
    
    record = {
        "document_id": document_id,
        "chunk_number": chunk_number,
        "chunk_text": chunk_text,
        "embedding": embedding,
        "source_title": source.title,
        "source_url": source.url,
        "source_type": source.source_type,
        "topic": source.topic,
        "section_heading": section_heading,
        "contains_solution": source.contains_solution,
    }

    (
        supabase
        .table("course_chunks")
        .insert(record)
        .execute()
    )


# ------- INGESTION --------------------------------------------#

def ingest_source(
    openai_client: OpenAI,
    supabase: Client,
    source: CourseSource,
) -> None:
    """Download, chunk, embed and store one approved source."""

    print(f"\nProcessing: {source.title}")
    print(f"URL: {source.url}")

    if source_already_exists(supabase, source.url):
        print("Skipped: source already exists.")
        return

    html = download_page(source.url)
    sections = extract_page_sections(html)

    if not sections:
        raise RuntimeError(
            f"No usable content found for {source.title}"
        )
    
    prepared_chunks = []

    for section in sections:
        section_chunks = split_text(section["text"])

        for chunk in section_chunks:
            prepared_chunks.append(
                {
                    "heading": section["heading"],
                    "text": chunk,
                }
            )
    
    if not prepared_chunks:
        raise RuntimeError(
            f"No chunks were created for {source.title}"
        )
    
    document_id = insert_document(
        supabase,
        source,
    )

    for index, chunk in enumerate(prepared_chunks, start=1):
        embedding_input = (
            f"Course: IST 356\n"
            f"Source: {source.title}\n"
            f"Topic: {source.topic}\n"
            f"Section: {chunk['heading']}\n\n"
            f"{chunk['text']}"
        )

        embedding = create_embedding(
            openai_client,
            embedding_input,
        )

        insert_chunk(
            supabase=supabase,
            document_id=document_id,
            chunk_number=index,
            chunk_text=chunk["text"],
            embedding=embedding,
            source=source,
            section_heading=chunk["heading"],
        )

        print(
            f"  Saved chunk {index} of "
            f"{len(prepared_chunks)}"
        )

        # Small pause to keep ingestion gentle.
        time.sleep(0.05)

    print(
        f"Completed: {len(prepared_chunks)} chunks saved."
    )

def main() -> None:
    print("Building IST 356 knowledge base...")

    openai_client, supabase = create_clients()

    failed_sources = []

    for source in APPROVED_SOURCES:
        try:
            ingest_source(
                openai_client, 
                supabase,
                source,
            )
        
        except Exception as error:
            failed_sources.append(source.title)
            print(f"ERROR: {source.title}")
            print(repr(error))

    print("\nKnowledge-base build finished.")

    if failed_sources:
        print("Sources that failed:")
        for title in failed_sources:
            print(f" - {title}")

            sys.exit(1)

        print("All approved sources were processed successfully.")

if __name__ == "__main__":
    main()

