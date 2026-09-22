"""FastAPI realtime WebSocket gateway."""

import asyncio
import json
from contextlib import asynccontextmanager
from uuid import uuid4
from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from packages.config.settings import get_settings
from packages.database.database import db, get_db
from packages.providers.factory import provider_factory
from packages.conversation.engine import ConversationEngine
from packages.knowledge.retrieval import RetrievalService
from packages.conversation.handoff import HandoffService
from packages.schemas.schemas import LanguageCode


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    db.initialize()
    await provider_factory.get_cache_provider()
    yield
    await db.close()
    await provider_factory.close_all()


app = FastAPI(title="SVKM Voice Assistant Realtime", lifespan=lifespan)
settings = get_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors.origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active sessions
active_sessions: Dict[str, ConversationEngine] = {}


@app.get("/health")
async def health_check():
    """Health check."""
    return {
        "status": "healthy",
        "service": "realtime",
        "active_sessions": len(active_sessions)
    }


@app.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """Voice session WebSocket endpoint."""
    await websocket.accept()

    session_id = uuid4()

    # Send ready event
    await websocket.send_json({
        "type": "ready",
        "session_id": str(session_id),
        "supported_languages": ["en-IN", "hi-IN", "mr-IN"]
    })

    try:
        # Create conversation engine
        async with db.session() as db_session:
            speech_provider = provider_factory.get_speech_provider()
            llm_provider = provider_factory.get_llm_provider()
            embedding_provider = provider_factory.get_embedding_provider()

            retrieval_service = RetrievalService(db_session, embedding_provider)
            handoff_service = HandoffService(db_session)

            engine = ConversationEngine(
                session_id=session_id,
                speech_provider=speech_provider,
                llm_provider=llm_provider,
                retrieval_service=retrieval_service,
                handoff_service=handoff_service
            )

            active_sessions[str(session_id)] = engine

            # Handle messages
            while True:
                data = await websocket.receive_json()
                event_type = data.get("type")

                if event_type == "text":
                    # Text input (for testing)
                    text = data.get("text", "")
                    language = LanguageCode(data.get("language", "en-IN"))

                    # Send thinking event
                    await websocket.send_json({"type": "thinking"})

                    # Process input
                    response = await engine.process_text_input(text, language)

                    # Send response
                    await websocket.send_json({
                        "type": "response",
                        "text": response["text"],
                        "language": response["language"].value,
                        "decision_type": response["decision_type"].value,
                        "citations": [
                            {
                                "source_id": c.source_id,
                                "chunk_id": c.chunk_id,
                                "text": c.text,
                                "confidence": c.confidence
                            }
                            for c in response["citations"]
                        ]
                    })

                elif event_type == "interrupt":
                    await engine.cancel_generation()

                elif event_type == "end":
                    result = await engine.end_conversation()
                    await websocket.send_json({
                        "type": "end",
                        "reason": result["reason"]
                    })
                    break

    except WebSocketDisconnect:
        pass
    finally:
        active_sessions.pop(str(session_id), None)
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.realtime.host,
        port=settings.realtime.port,
        reload=False
    )
