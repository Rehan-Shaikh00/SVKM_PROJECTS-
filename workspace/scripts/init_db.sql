-- Initialize pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create enum types
CREATE TYPE call_status AS ENUM ('active', 'completed', 'failed', 'transferred');
CREATE TYPE turn_role AS ENUM ('user', 'assistant', 'system');
CREATE TYPE decision_type AS ENUM ('answer', 'clarify', 'abstain', 'handoff');
CREATE TYPE ingestion_status AS ENUM ('pending', 'processing', 'completed', 'failed');
CREATE TYPE publication_status AS ENUM ('draft', 'staged', 'published', 'archived');

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE svkm_voice_assistant TO svkm_user;
