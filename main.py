import os
import json
import uuid
import httpx
import time
from typing import Optional, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import AsyncGroq

app = FastAPI(title="WM Studio Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

groq_client  = AsyncGroq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful support agent for WM Studio, a premium AI image and video generation platform. "
    "Be concise, friendly, and professional.",
)

# session_id -> { ws, history:[{role,text,ts}], created_at, tenant_id }
customer_sessions: dict[str, dict] = {}
agent_connections: set[WebSocket]  = set()


def ts() -> int:
    return int(time.time() * 1000)


def session_summary(sid: str) -> dict:
    s = customer_sessions.get(sid, {})
    history = s.get("history", [])
    last = history[-1] if history else None
    return {
        "session_id":    sid,
        "created_at":    s.get("created_at", 0),
        "message_count": len(history),
        "last_text":     last["text"][:80] if last else "",
        "last_role":     last["role"] if last else "",
        "last_ts":       last["ts"] if last else 0,
        "online":        s.get("ws") is not None,
    }


async def broadcast_to_agents(msg: dict):
    dead = set()
    text = json.dumps(msg)
    for aw in agent_connections:
        try:
            await aw.send_text(text)
        except Exception:
            dead.add(aw)
    agent_connections.difference_update(dead)


# ── Customer WebSocket ────────────────────────────────────────────────────────
@app.websocket("/api/v1/ws/chat")
async def customer_ws(
    websocket: WebSocket,
    session_id: str = Query(""),
    tenant_id: Optional[str] = Query(None),
):
    await websocket.accept()
    sid = session_id or str(uuid.uuid4())

    if sid not in customer_sessions:
        customer_sessions[sid] = {"history": [], "created_at": ts(), "tenant_id": tenant_id}
    customer_sessions[sid]["ws"] = websocket

    await websocket.send_text(json.dumps({"type": "connected", "session_id": sid}))
    await broadcast_to_agents({"type": "session_update", "session": session_summary(sid)})
    print(f"[Chat] Customer connected session={sid}")

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            mtype = data.get("type", "")

            if mtype == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if mtype == "disconnect":
                break

            if mtype == "chat_message":
                text = data.get("text", "").strip()
                if not text:
                    continue

                entry = {"role": "user", "text": text, "ts": ts()}
                customer_sessions[sid]["history"].append(entry)
                print(f"[Chat] session={sid} user: {text[:60]}")

                await broadcast_to_agents({
                    "type":       "customer_message",
                    "session_id": sid,
                    "text":       text,
                    "ts":         entry["ts"],
                    "session":    session_summary(sid),
                })

                if not agent_connections:
                    await _auto_reply(sid, websocket)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Chat] Error session={sid}: {e}")
    finally:
        if sid in customer_sessions:
            customer_sessions[sid]["ws"] = None
        await broadcast_to_agents({"type": "session_update", "session": session_summary(sid)})
        print(f"[Chat] Customer disconnected session={sid}")


async def _auto_reply(sid: str, ws: WebSocket):
    """Fallback when no agent is online — Groq if available, else canned message."""
    if groq_client:
        history = customer_sessions[sid].get("history", [])
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history[-20:]:
            msgs.append({"role": "user" if h["role"] == "user" else "assistant", "content": h["text"]})
        try:
            resp = await groq_client.chat.completions.create(
                model=GROQ_MODEL, messages=msgs, temperature=0.7, max_tokens=512
            )
            reply_text = resp.choices[0].message.content or "I'll look into that for you."
        except Exception as e:
            print(f"[AutoReply] Groq error: {e}")
            reply_text = "Our team will get back to you shortly."
    else:
        reply_text = "Thanks for your message — a team member will respond shortly."

    entry = {"role": "assistant", "text": reply_text, "ts": ts()}
    customer_sessions[sid]["history"].append(entry)
    await ws.send_text(json.dumps({"type": "bot_response", "text": reply_text}))


# ── Agent WebSocket ───────────────────────────────────────────────────────────
@app.websocket("/api/v1/ws/agent")
async def agent_ws(
    websocket: WebSocket,
):
    await websocket.accept()
    agent_connections.add(websocket)
    print(f"[Agent] Agent connected. Total agents: {len(agent_connections)}")

    # Send all current sessions + full histories on connect
    await websocket.send_text(json.dumps({
        "type":     "init",
        "sessions": [session_summary(sid) for sid in customer_sessions],
        "history":  {sid: customer_sessions[sid].get("history", []) for sid in customer_sessions},
    }))

    try:
        while True:
            raw  = await websocket.receive_text()
            data = json.loads(raw)
            mtype = data.get("type", "")

            if mtype == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
                continue

            if mtype == "agent_reply":
                sid  = data.get("session_id", "")
                text = data.get("text", "").strip()
                if not sid or not text:
                    continue

                entry = {"role": "assistant", "text": text, "ts": ts()}
                if sid in customer_sessions:
                    customer_sessions[sid]["history"].append(entry)

                cust_ws = customer_sessions.get(sid, {}).get("ws")
                if cust_ws:
                    try:
                        await cust_ws.send_text(json.dumps({"type": "bot_response", "text": text}))
                        print(f"[Agent] Replied to session={sid}: {text[:60]}")
                    except Exception as e:
                        print(f"[Agent] Delivery failed session={sid}: {e}")
                else:
                    print(f"[Agent] session={sid} offline — stored only")

                await broadcast_to_agents({
                    "type":       "agent_reply_echo",
                    "session_id": sid,
                    "text":       text,
                    "ts":         entry["ts"],
                })

            if mtype == "typing":
                sid = data.get("session_id", "")
                cust_ws = customer_sessions.get(sid, {}).get("ws")
                if cust_ws:
                    try:
                        await cust_ws.send_text(json.dumps({"type": "agent_typing"}))
                    except Exception:
                        pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Agent] Error: {e}")
    finally:
        agent_connections.discard(websocket)
        print(f"[Agent] Agent disconnected. Total agents: {len(agent_connections)}")


# ── Session insight ───────────────────────────────────────────────────────────
class InsightRequest(BaseModel):
    session_id: str
    tenant_id:  Optional[str] = None
    messages:   List[dict]


@app.post("/api/v1/sessions/insight")
async def session_insight(req: InsightRequest):
    print(f"[Insight] session={req.session_id} messages={len(req.messages)}")
    if len(req.messages) < 2:
        return {"status": "skipped", "reason": "too_short"}

    insight = ""
    if groq_client:
        try:
            resp = await groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content":
                     "Summarise this support conversation in 2 sentences. "
                     "Include: main topic, user intent, resolution status."},
                    *[{"role": m.get("role","user"), "content": m.get("content", m.get("text",""))}
                      for m in req.messages[-20:]],
                    {"role": "user", "content": "Summarise now."},
                ],
                temperature=0.3, max_tokens=120,
            )
            insight = resp.choices[0].message.content or ""
            print(f"[Insight] Generated: {insight[:80]}...")
        except Exception as e:
            print(f"[Insight] Groq error: {e}")

    if SUPABASE_URL and SUPABASE_KEY:
        try:
            async with httpx.AsyncClient() as http:
                r = await http.post(
                    f"{SUPABASE_URL}/rest/v1/session_insights",
                    headers={
                        "apikey": SUPABASE_KEY,
                        "Authorization": f"Bearer {SUPABASE_KEY}",
                        "Content-Type": "application/json",
                        "Prefer": "return=minimal",
                    },
                    json={"session_id": req.session_id, "tenant_id": req.tenant_id,
                          "insight": insight, "message_count": len(req.messages)},
                    timeout=5,
                )
            if r.status_code in (200, 201):
                print(f"[Insight] Saved to Supabase ✓")
            else:
                print(f"[Insight] Supabase error {r.status_code}: {r.text}")
        except Exception as e:
            print(f"[Insight] Supabase failed: {e}")

    return {"status": "ok", "insight": insight}


# ── REST fallback ─────────────────────────────────────────────────────────────
class ConversationRequest(BaseModel):
    session_id:   Optional[str] = ""
    tenant_id:    Optional[str] = None
    user_message: str
    history:      List[dict] = []


@app.post("/api/v1/conversations")
async def conversation(req: ConversationRequest):
    if not req.user_message.strip():
        raise HTTPException(status_code=400, detail="user_message is required")
    if not groq_client:
        return {"bot_response": "A team member will respond shortly.", "session_id": req.session_id}
    try:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in req.history[-20:]:
            msgs.append({"role": h.get("role","user"), "content": h.get("content", h.get("text",""))})
        msgs.append({"role": "user", "content": req.user_message})
        resp = await groq_client.chat.completions.create(model=GROQ_MODEL, messages=msgs, temperature=0.7, max_tokens=512)
        return {"bot_response": resp.choices[0].message.content or "", "session_id": req.session_id}
    except Exception as e:
        print(f"[REST] error: {e}")
        raise HTTPException(status_code=500, detail="Service error")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(customer_sessions), "agents_online": len(agent_connections)}

@app.get("/")
def root():
    return {"service": "WM Studio Chatbot API", "status": "running"}
