
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from kb_core import load_markdown_documents, chunk_documents
from retrieval_core import make_chunk_items
from organizer_api import OrganizerAPI
from hybrid_retrieval import HybridRetriever
from answer_generation import GroundedAnswerGenerator

from database import (
    SessionLocal,
    ChatSession,
    ChatMessage,
    CustomerLead,
    init_db,
    utc_now,
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]

KNOWLEDGE_BASE_PATH = (
    PROJECT_DIR / "local_data" / "knowledge-base.zip"
)

STATIC_DIR = Path(__file__).parent / "static"


# --------------------------------------------------
# AI ENGINE
# --------------------------------------------------

api: OrganizerAPI | None = None
retriever: HybridRetriever | None = None
answer_generator: GroundedAnswerGenerator | None = None


def initialize_ai_engine() -> None:
    global api, retriever, answer_generator

    print("Loading MTS knowledge base...")

    if not KNOWLEDGE_BASE_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found: {KNOWLEDGE_BASE_PATH}"
        )

    documents = load_markdown_documents(KNOWLEDGE_BASE_PATH)
    chunks = chunk_documents(documents)
    items = make_chunk_items(chunks)

    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")

    print("Connecting to local Ollama...")

    api = OrganizerAPI()

    print("Initializing hybrid retrieval...")

    retriever = HybridRetriever(
        items=items,
        api=api,
    )

    print("Initializing answer generator...")

    answer_generator = GroundedAnswerGenerator(api=api)

    print("MTS AI engine is ready!")


# --------------------------------------------------
# APPLICATION LIFESPAN
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing SQLite database...")
    init_db()

    initialize_ai_engine()

    yield

    print("MTS Energytool Consultant shutting down...")


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="MTS Energytool AI Consultant",
    description="Local AI-powered MTS Energytool customer consultant.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# REQUEST AND RESPONSE MODELS
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    history: list[dict[str, str]] = Field(
        default_factory=list
    )

    session_id: int | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    response_time_ms: int
    session_id: int


class SessionResponse(BaseModel):
    id: int
    title: str
    status: str
    created_at: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    created_at: str


class ConversationResponse(BaseModel):
    id: int
    title: str
    messages: list[MessageResponse]


class LeadRequest(BaseModel):
    session_id: int

    full_name: str = Field(
        min_length=1,
        max_length=255,
    )

    phone: str = Field(
        min_length=1,
        max_length=50,
    )

    company: str = Field(
        min_length=1,
        max_length=255,
    )

    summary: str = Field(
        default="",
        max_length=5000,
    )


# --------------------------------------------------
# DATABASE HELPERS
# --------------------------------------------------

def serialize_datetime(value):
    if value is None:
        return ""

    return value.isoformat()


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/api/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ready" if retriever is not None else "starting",
        "ai_engine": retriever is not None,
        "database": True,
        "chat_model": (
            api.llm_model if api is not None else None
        ),
        "embedding_model": (
            api.embedding_model if api is not None else None
        ),
    }


# --------------------------------------------------
# CREATE A NEW CONVERSATION
# --------------------------------------------------

@app.post("/api/sessions", response_model=SessionResponse)
def create_session():
    db: Session = SessionLocal()

    try:
        new_session = ChatSession(
            title="Новый диалог",
            status="active",
        )

        db.add(new_session)
        db.commit()
        db.refresh(new_session)

        return SessionResponse(
            id=new_session.id,
            title=new_session.title,
            status=new_session.status,
            created_at=serialize_datetime(
                new_session.created_at
            ),
        )

    finally:
        db.close()


# --------------------------------------------------
# LIST PREVIOUS CONVERSATIONS
# --------------------------------------------------

@app.get("/api/sessions", response_model=list[SessionResponse])
def list_sessions():
    db: Session = SessionLocal()

    try:
        sessions = (
            db.query(ChatSession)
            .order_by(ChatSession.updated_at.desc())
            .all()
        )

        return [
            SessionResponse(
                id=item.id,
                title=item.title,
                status=item.status,
                created_at=serialize_datetime(
                    item.created_at
                ),
            )
            for item in sessions
        ]

    finally:
        db.close()


# --------------------------------------------------
# RETRIEVE A CONVERSATION AND ITS MESSAGES
# --------------------------------------------------

@app.get(
    "/api/sessions/{session_id}",
    response_model=ConversationResponse,
)
def get_session(session_id: int):
    db: Session = SessionLocal()

    try:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.id == session_id)
            .first()
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        return ConversationResponse(
            id=session.id,
            title=session.title,
            messages=[
                MessageResponse(
                    id=message.id,
                    role=message.role,
                    content=message.content,
                    created_at=serialize_datetime(
                        message.created_at
                    ),
                )
                for message in session.messages
            ],
        )

    finally:
        db.close()


# --------------------------------------------------
# CHAT ENDPOINT
# --------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if retriever is None or answer_generator is None:
        raise HTTPException(
            status_code=503,
            detail="The MTS AI engine is not ready.",
        )

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    start_time = time.perf_counter()

    db: Session = SessionLocal()

    try:
        # Find an existing conversation or create one.
        if request.session_id is not None:
            current_session = (
                db.query(ChatSession)
                .filter(
                    ChatSession.id == request.session_id
                )
                .first()
            )

            if current_session is None:
                raise HTTPException(
                    status_code=404,
                    detail="Conversation not found.",
                )

        else:
            current_session = ChatSession(
                title=question[:60],
                status="active",
            )

            db.add(current_session)
            db.commit()
            db.refresh(current_session)

        # Search the MTS knowledge base.
        results = retriever.search(
            question,
            top_k=5,
        )

        # Generate an evidence-based answer.
        grounded_answer = answer_generator.generate(
            question=question,
            results=results,
            history=request.history,
        )

        # Save the customer's question.
        user_message = ChatMessage(
            session_id=current_session.id,
            role="user",
            content=question,
        )

        # Save the AI response.
        assistant_message = ChatMessage(
            session_id=current_session.id,
            role="assistant",
            content=grounded_answer.answer,
        )

        db.add_all([
            user_message,
            assistant_message,
        ])

        # Update conversation title and activity.
        if current_session.title == "Новый диалог":
            current_session.title = question[:60]

        current_session.updated_at = utc_now()

        db.commit()

        elapsed_ms = int(
            (time.perf_counter() - start_time) * 1000
        )

        return ChatResponse(
            answer=grounded_answer.answer,
            sources=grounded_answer.sources,
            response_time_ms=elapsed_ms,
            session_id=current_session.id,
        )

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        print(f"Chat processing error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to process your question. Please try again.",
        ) from exc

    finally:
        db.close()


# --------------------------------------------------
# CUSTOMER LEAD COLLECTION
# --------------------------------------------------

@app.post("/api/leads")
def create_lead(request: LeadRequest):
    db: Session = SessionLocal()

    try:
        # Check that the conversation exists.
        current_session = (
            db.query(ChatSession)
            .filter(
                ChatSession.id == request.session_id
            )
            .first()
        )

        if current_session is None:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )

        # Clean the submitted information.
        full_name = request.full_name.strip()
        phone = request.phone.strip()
        company = request.company.strip()
        summary = request.summary.strip()

        if not full_name or not phone or not company:
            raise HTTPException(
                status_code=400,
                detail="Name, phone number and company are required.",
            )

        # Save the lead in SQLite.
        lead = CustomerLead(
            session_id=request.session_id,
            full_name=full_name,
            phone=phone,
            company=company,
            summary=summary,
        )

        db.add(lead)
        db.commit()
        db.refresh(lead)

        return {
            "success": True,
            "message": "Заявка успешно сохранена.",
            "lead_id": lead.id,
        }

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        print(f"Lead submission error: {exc}")

        raise HTTPException(
            status_code=500,
            detail="Unable to save the enquiry. Please try again.",
        ) from exc

    finally:
        db.close()


# --------------------------------------------------
# SERVE THE WEBSITE
# --------------------------------------------------

if not STATIC_DIR.exists():
    raise RuntimeError(
        f"Static directory not found: {STATIC_DIR}"
    )

app.mount(
    "/",
    StaticFiles(
        directory=str(STATIC_DIR),
        html=True,
    ),
    name="static",
)
