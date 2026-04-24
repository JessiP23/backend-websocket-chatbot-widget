import os
import json
import uuid
import httpx
import time
import re
from typing import Optional, List
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="WM Studio Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
# Strip trailing /rest/v1/ from SUPABASE_URL if user included it
_raw_supa_url = os.environ.get("SUPABASE_URL", "")
SUPABASE_URL = re.sub(r"/rest/v1/?$", "", _raw_supa_url.rstrip("/"))
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

OFFLINE_MESSAGE = os.environ.get(
    "OFFLINE_MESSAGE",
    "Thanks for your message — a team member will respond shortly.",
)

# ── In-memory state ──────────────────────────────────────────────────────────
# session_id -> { ws, history:[{role,text,ts}], created_at, tenant_id, status }
customer_sessions: dict[str, dict] = {}
agent_connections: set[WebSocket]  = set()


def now_ms() -> int:
    return int(time.time() * 1000)


# ── Supabase persistence ─────────────────────────────────────────────────────
def _supa_headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _supa_ok() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


async def db_upsert_session(sid: str, created_at: int, tenant_id: Optional[str] = None, status: str = "active"):
    if not _supa_ok():
        return
    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                f"{SUPABASE_URL}/rest/v1/chat_sessions",
                headers={**_supa_headers(), "Prefer": "resolution=merge-duplicates,return=minimal"},
                json={"session_id": sid, "created_at": created_at, "tenant_id": tenant_id, "status": status},
                timeout=5,
            )
        print(f"[DB] Session upserted: {sid} status={status}")
    except Exception as e:
        print(f"[DB] upsert session failed: {e}")


async def db_update_session_status(sid: str, status: str):
    if not _supa_ok():
        return
    try:
        async with httpx.AsyncClient() as http:
            await http.patch(
                f"{SUPABASE_URL}/rest/v1/chat_sessions?session_id=eq.{sid}",
                headers=_supa_headers(),
                json={"status": status},
                timeout=5,
            )
        print(f"[DB] Session status updated: {sid} -> {status}")
    except Exception as e:
        print(f"[DB] update session status failed: {e}")


async def db_insert_message(sid: str, role: str, text: str, ts: int):
    if not _supa_ok():
        return
    try:
        async with httpx.AsyncClient() as http:
            await http.post(
                f"{SUPABASE_URL}/rest/v1/chat_messages",
                headers=_supa_headers(),
                json={"session_id": sid, "role": role, "text": text, "ts": ts},
                timeout=5,
            )
    except Exception as e:
        print(f"[DB] insert message failed: {e}")


async def db_load_all_sessions() -> list[dict]:
    if not _supa_ok():
        return []
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                f"{SUPABASE_URL}/rest/v1/chat_sessions?order=created_at.desc&limit=200",
                headers={**_supa_headers(), "Prefer": ""},
                timeout=10,
            )
            if r.status_code == 200:
                rows = r.json()
                print(f"[DB] Loaded {len(rows)} sessions from DB")
                return rows
            else:
                print(f"[DB] Load sessions HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[DB] load sessions failed: {e}")
    return []


async def db_load_all_histories() -> dict[str, list[dict]]:
    if not _supa_ok():
        return {}
    try:
        async with httpx.AsyncClient() as http:
            r = await http.get(
                f"{SUPABASE_URL}/rest/v1/chat_messages?order=ts.asc&limit=5000",
                headers={**_supa_headers(), "Prefer": ""},
                timeout=15,
            )
            if r.status_code == 200:
                histories: dict[str, list[dict]] = {}
                for msg in r.json():
                    sid = msg["session_id"]
                    if sid not in histories:
                        histories[sid] = []
                    histories[sid].append({"role": msg["role"], "text": msg["text"], "ts": msg["ts"]})
                print(f"[DB] Loaded histories for {len(histories)} sessions")
                return histories
            else:
                print(f"[DB] Load histories HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[DB] load histories failed: {e}")
    return {}


# ── Session summary builders ─────────────────────────────────────────────────
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
        "status":        s.get("status", "active"),
    }


def session_summary_from_db(row: dict, messages: list[dict]) -> dict:
    last = messages[-1] if messages else None
    sid = row["session_id"]
    is_online = (sid in customer_sessions and customer_sessions[sid].get("ws") is not None)
    return {
        "session_id":    sid,
        "created_at":    row.get("created_at", 0),
        "message_count": len(messages),
        "last_text":     last["text"][:80] if last else "",
        "last_role":     last["role"] if last else "",
        "last_ts":       last["ts"] if last else 0,
        "online":        is_online,
        "status":        row.get("status", "closed"),
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
    created = now_ms()

    if sid not in customer_sessions:
        customer_sessions[sid] = {"history": [], "created_at": created, "tenant_id": tenant_id, "status": "active"}
        await db_upsert_session(sid, created, tenant_id, "active")
    customer_sessions[sid]["ws"] = websocket
    customer_sessions[sid]["status"] = "active"

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

                msg_ts = now_ms()
                entry = {"role": "user", "text": text, "ts": msg_ts}
                customer_sessions[sid]["history"].append(entry)
                print(f"[Chat] session={sid} user: {text[:60]}")

                await db_insert_message(sid, "user", text, msg_ts)

                await broadcast_to_agents({
                    "type":       "customer_message",
                    "session_id": sid,
                    "text":       text,
                    "ts":         msg_ts,
                    "session":    session_summary(sid),
                })

                # Human-only — no auto-reply. Message is stored and
                # forwarded to the agent dashboard; user waits for a
                # real human response.

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[Chat] Error session={sid}: {e}")
    finally:
        if sid in customer_sessions:
            customer_sessions[sid]["ws"] = None
            customer_sessions[sid]["status"] = "closed"
        # Mark session as closed in DB
        await db_update_session_status(sid, "closed")
        await broadcast_to_agents({"type": "session_update", "session": session_summary(sid)})
        print(f"[Chat] Customer disconnected session={sid}")


# ── Agent WebSocket ───────────────────────────────────────────────────────────
@app.websocket("/api/v1/ws/agent")
async def agent_ws(
    websocket: WebSocket,
):
    await websocket.accept()
    agent_connections.add(websocket)
    print(f"[Agent] Agent connected. Total agents: {len(agent_connections)}")

    # Build init: merge DB + in-memory
    db_sessions = await db_load_all_sessions()
    db_histories = await db_load_all_histories()

    all_sids = set()
    merged_sessions = []
    merged_histories: dict[str, list[dict]] = {}

    for row in db_sessions:
        sid = row["session_id"]
        all_sids.add(sid)
        msgs = db_histories.get(sid, [])
        if sid in customer_sessions:
            mem = customer_sessions[sid].get("history", [])
            if len(mem) > len(msgs):
                msgs = mem
        merged_sessions.append(session_summary_from_db(row, msgs))
        merged_histories[sid] = msgs

    for sid, sdata in customer_sessions.items():
        if sid not in all_sids:
            merged_sessions.append(session_summary(sid))
            merged_histories[sid] = sdata.get("history", [])

    await websocket.send_text(json.dumps({
        "type":     "init",
        "sessions": merged_sessions,
        "history":  merged_histories,
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

                msg_ts = now_ms()
                entry = {"role": "assistant", "text": text, "ts": msg_ts}
                if sid in customer_sessions:
                    customer_sessions[sid]["history"].append(entry)

                await db_insert_message(sid, "assistant", text, msg_ts)

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
                    "ts":         msg_ts,
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
    return {"bot_response": "Your message has been received.", "session_id": req.session_id}


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "active_sessions": len([s for s in customer_sessions.values() if s.get("ws")]),
        "total_sessions": len(customer_sessions),
        "agents_online": len(agent_connections),
        "supabase": _supa_ok(),
    }

@app.get("/")
def root():
    return {"service": "WM Studio Chatbot API", "status": "running"}
