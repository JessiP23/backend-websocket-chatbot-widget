import os
import json
import uuid
import httpx
from typing import Optional, List

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

# Supabase config (optional — insight storage only)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

# Browser owns conversation history — no server-side session storage needed.

async def stream_ai_response(history: list[dict], user_message: str):
    """Async generator that yields text chunks from Groq. History comes from the browser."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]
    messages.append({"role": "user", "content": user_message})

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


async def get_ai_response(history: list[dict], user_message: str) -> str:
    """Non-streaming response for REST fallback."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history[-20:]
    messages.append({"role": "user", "content": user_message})

    response = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""


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
                history   = data.get("history", [])  # full history sent by browser
                if not user_text:
                    continue

                message_id = str(uuid.uuid4())
                try:
                    async for chunk_text in stream_ai_response(history, user_text):
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
    history: List[dict] = []


@app.post("/api/v1/conversations")
async def conversation(req: ConversationRequest):
    if not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")

    try:
        bot_reply = await get_ai_response(req.history, req.user_message)
        return {"bot_response": bot_reply, "session_id": req.session_id}
    except Exception as e:
        print(f"[REST] Groq error: {e}")
        raise HTTPException(status_code=500, detail="AI service error")


# ── Session insight endpoint ───────────────────────────────────────────────────
class InsightRequest(BaseModel):
    session_id: str
    tenant_id: Optional[str] = None
    messages: List[dict]  # [{ role: "user"|"assistant", content: "..." }]


@app.post("/api/v1/sessions/insight")
async def session_insight(req: InsightRequest):
    print(f"[Insight] Received for session={req.session_id} messages={len(req.messages)}")

    if len(req.messages) < 2:
        print(f"[Insight] Skipped — too short ({len(req.messages)} messages)")
        return {"status": "skipped", "reason": "too_short"}

    # Ask Groq for a 2-sentence summary
    summary_prompt = (
        "Summarise this support conversation in exactly 2 sentences. "
        "Include: main topic, user intent, and whether it was resolved. "
        "Be factual and concise. No filler phrases."
    )
    try:
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": summary_prompt},
                *req.messages[-20:],
                {"role": "user", "content": "Summarise the above conversation now."},
            ],
            temperature=0.3,
            max_tokens=120,
        )
        insight = resp.choices[0].message.content or ""
        print(f"[Insight] Generated: {insight[:80]}...")
    except Exception as e:
        print(f"[Insight] Groq error: {e}")
        raise HTTPException(status_code=500, detail="AI summary failed")

    # Store in Supabase — log response status and body on failure
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient() as http:
                sb_resp = await http.post(
                    f"{SUPABASE_URL}/rest/v1/session_insights",
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                    json={
                        "session_id": req.session_id,
                        "tenant_id": req.tenant_id,
                        "insight": insight,
                        "message_count": len(req.messages),
                    },
                    timeout=5,
                )
            if sb_resp.status_code in (200, 201):
                print(f"[Insight] Saved to Supabase ✓ (session={req.session_id})")
            else:
                # Log full response so we can see exactly why it failed
                print(f"[Insight] Supabase error {sb_resp.status_code}: {sb_resp.text}")
        except Exception as e:
            print(f"[Insight] Supabase request failed: {e}")
    else:
        print("[Insight] Supabase not configured — skipping storage (set SUPABASE_URL + SUPABASE_KEY secrets)")

    return {"status": "ok", "insight": insight}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"service": "WM Studio Chatbot API", "status": "running"}
