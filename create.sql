CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE notices (
    notice_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    deadline TIMESTAMPTZ NOT NULL,
    pdf_bytes BYTEA NOT NULL,
    content_markdown TEXT NOT NULL,
    clean_markdown TEXT NOT NULL
);

CREATE TABLE notice_tables (
    table_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    notice_id UUID NOT NULL REFERENCES notices(notice_id) ON DELETE CASCADE,
    content TEXT NOT NULL
);

select * from notices;