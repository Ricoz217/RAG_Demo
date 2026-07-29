CREATE TABLE IF NOT EXISTS documents (
    id bigserial PRIMARY KEY,
    source_repo text NOT NULL,
    source_commit text NOT NULL,
    source_path text NOT NULL,
    language text NOT NULL,
    title text,
    content_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),

    UNIQUE (source_repo, source_commit, source_path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    document_id bigint NOT NULL
        REFERENCES documents(id) ON DELETE CASCADE,

    chunk_index integer NOT NULL,
    chunker_version text NOT NULL,
    heading_path text[] NOT NULL DEFAULT '{}',
    content_raw text NOT NULL,
    retrieval_text text NOT NULL,
    char_count integer NOT NULL,
    content_hash text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}',
    embedding_model text NOT NULL,
    embedding_dimensions integer NOT NULL,
    embedding vector(1024) NOT NULL,

    CONSTRAINT chunks_embedding_dimensions_check
        CHECK (embedding_dimensions = 1024),
    UNIQUE (document_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS ingestion_requests (
    idempotency_key text PRIMARY KEY,
    request_hash text NOT NULL,
    status text NOT NULL,
    result jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
ON chunks
USING hnsw (
    embedding vector_cosine_ops
)
WITH (
    m = 16,
    ef_construction = 64
);

CREATE INDEX IF NOT EXISTS chunks_document_id_idx
ON chunks (document_id);

CREATE INDEX IF NOT EXISTS chunks_metadata_gin
ON chunks
USING gin (metadata);

