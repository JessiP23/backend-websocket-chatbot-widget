-- Chat Sessions & Messages tables for WM Studio Chatbot
-- Run this in your Supabase SQL Editor

-- Sessions table
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    created_at BIGINT NOT NULL,
    tenant_id  TEXT,
    inserted_at TIMESTAMPTZ DEFAULT NOW()
);

-- Messages table
CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    text        TEXT NOT NULL,
    ts          BIGINT NOT NULL,
    inserted_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast message loading by session
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_ts
    ON chat_messages(session_id, ts ASC);

-- Index for session listing
CREATE INDEX IF NOT EXISTS idx_chat_sessions_created
    ON chat_sessions(created_at DESC);

-- Enable Row Level Security (open for now — tighten with policies later)
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY;

-- Allow all operations via service key (anon key won't work by default)
CREATE POLICY "service_full_access_sessions" ON chat_sessions
    FOR ALL USING (true) WITH CHECK (true);

CREATE POLICY "service_full_access_messages" ON chat_messages
    FOR ALL USING (true) WITH CHECK (true);
