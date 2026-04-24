-- Add status column to chat_sessions if it doesn't exist
ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'closed';
