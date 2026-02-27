-- ============================================================
-- ScalpIQ — PostgreSQL Database Schema
-- Run: psql -U postgres -d scalpiq -f schema.sql
-- ============================================================

-- Create database (run separately as superuser if needed)
-- CREATE DATABASE scalpiq;
-- \c scalpiq

-- ── Extensions ───────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Users table ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name     VARCHAR(120) NOT NULL,
    email         VARCHAR(255) NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_verified   BOOLEAN DEFAULT FALSE,
    is_active     BOOLEAN DEFAULT TRUE,
    avatar_url    TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    updated_at    TIMESTAMPTZ DEFAULT NOW(),
    last_login    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ── Sessions table ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id       UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token         TEXT NOT NULL UNIQUE,
    ip_address    INET,
    user_agent    TEXT,
    expires_at    TIMESTAMPTZ NOT NULL,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_token   ON sessions(token);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

-- ── Analysis history table ───────────────────────────────────────
CREATE TABLE IF NOT EXISTS analyses (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    age                 INT,
    gender              VARCHAR(20),
    oil_frequency       VARCHAR(30),
    conditioner_use     VARCHAR(30),
    water_intake        VARCHAR(20),
    diet_type           VARCHAR(30),
    stress_level        VARCHAR(20),
    sleep_hours         VARCHAR(20),
    prediction          VARCHAR(20),
    confidence          FLOAT,
    prob_low            FLOAT,
    prob_medium         FLOAT,
    prob_high           FLOAT,
    hair_care_index     FLOAT,
    diet_quality_score  FLOAT,
    stress_proxy        FLOAT,
    sleep_score         FLOAT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);

-- ── Password reset tokens ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS password_resets (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token      TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ NOT NULL,
    used       BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Auto-update updated_at trigger ──────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_users_updated_at ON users;
CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ── Sample admin user (password: Admin@1234) ─────────────────────
-- INSERT INTO users (full_name, email, password_hash, is_verified)
-- VALUES ('Admin User', 'admin@scalpiq.com',
--   crypt('Admin@1234', gen_salt('bf', 12)), TRUE);