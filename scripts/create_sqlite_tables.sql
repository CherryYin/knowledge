CREATE TABLE IF NOT EXISTS faq_feedback (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    markets TEXT DEFAULT ''
);