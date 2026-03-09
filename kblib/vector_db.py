import sqlite3
from pathlib import Path


class FAQVectorDB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS faq_feedback (
                    id INTEGER PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    markets TEXT DEFAULT ''
                )
                """
            )
            conn.commit()

    def upsert_feedback(self, faq_id: int, question: str, answer: str, markets: str = ""):
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO faq_feedback (id, question, answer, markets)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    question = excluded.question,
                    answer = excluded.answer,
                    markets = excluded.markets
                """,
                (faq_id, question, answer, markets),
            )
            conn.commit()

    def update_feedback(self, faq_id: int, question: str, answer: str, markets: str = ""):
        self.upsert_feedback(faq_id, question, answer, markets=markets)

    def delete_feedback(self, faq_id: int):
        with self._connect() as conn:
            conn.execute("DELETE FROM faq_feedback WHERE id = ?", (faq_id,))
            conn.commit()

    def clear_all(self):
        with self._connect() as conn:
            conn.execute("DELETE FROM faq_feedback")
            conn.commit()

    def reindex(self, items: list[dict]):
        self.clear_all()
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO faq_feedback (id, question, answer, markets)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        int(item["id"]),
                        item["question"],
                        item["answer"],
                        item.get("markets", ""),
                    )
                    for item in items
                ],
            )
            conn.commit()
