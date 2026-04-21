import os
import json
import uuid
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq

app = FastAPI(title="WM Studio Chatbot API")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://wmstudio.io",
        "https://www.wmstudio.io",
        "https://chat-widget.fly.dev",
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Groq client ───────────────────────────────────────────────────────────────
client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])

GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful and friendly support assistant for WM Studio. "
    "Be concise, clear, and professional.",
)

# In-memory session history (resets on restart)
sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    if session_id not in sessions:
        sessions[session_id] = []
    return sessions[session_id]


async def stream_ai_response(session_id: str, user_message: str):
    """Async generator that yields text chunks from Groq."""
    history = get_history(session_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]

    full_response = ""
    stream = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
        stream=True,
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            full_response += delta
            yield delta

    history.append({"role": "assistant", "content": full_response})


async def get_ai_response(session_id: str, user_message: str) -> str:
    """Non-streaming response for REST fallback."""
    history = get_history(session_id)
    history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]

    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    bot_reply = response.choices[0].message.content or ""
    history.append({"role": "assistant", "content": bot_reply})
    return bot_reply


# ── WebSocket endpoint ────────────────────────────────────────────────────────
@app.websocket("/api/v1/ws/chat")
async def websocket_chat(
    websocket: WebSocket,
    session_id: str = "",
    tenant_id: Optional[str] = None,
):
    await websocket.accept()
    await websocket.send_text(json.dumps({"type": "connected", "session_id": session_id}))

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type", "")

            if msg_type == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if msg_type == "disconnect":
                break

            if msg_type == "chat_message":
                user_text = data.get("text", "").strip()
                sid = data.get("session_id", session_id) or session_id
                if not user_text:
                    continue

                message_id = str(uuid.uuid4())
                try:
                    async for chunk_text in stream_ai_response(sid, user_text):
                        await websocket.send_text(json.dumps({
                            "type": "chunk",
                            "message_id": message_id,
                            "text": chunk_text,
                        }))

                    await websocket.send_text(json.dumps({
                        "type": "chunk_done",
                        "message_id": message_id,
                    }))

                except Exception as e:
                    print(f"[WS] Groq error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Sorry, I couldn't process that. Please try again.",
                    }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": "Connection error."}))
        except Exception:
            pass


# ── REST fallback endpoint ────────────────────────────────────────────────────
class ConversationRequest(BaseModel):
    session_id: Optional[str] = ""
    tenant_id: Optional[str] = None
    user_message: str


@app.post("/api/v1/conversations")
async def conversation(req: ConversationRequest):
    if not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    try:
        bot_reply = await get_ai_response(req.session_id or "rest", req.user_message)
        return {"bot_response": bot_reply, "session_id": req.session_id}
    except Exception as e:
        print(f"[REST] Groq error: {e}")
        raise HTTPException(status_code=500, detail="AI service error")


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "WM Studio Chatbot API", "status": "running"}
