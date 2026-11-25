"""Persistent, atomic claims from opposite ends of the MATH-500 question list."""

from contextlib import closing
from pathlib import Path
import sqlite3


class SharedQuestionQueue:
    def __init__(self, path):
        self.path = Path(path)

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def initialize(self, question_ids):
        ids = list(question_ids)
        if ids != list(range(500)):
            raise ValueError("The shared E4 queue requires all 500 MATH-500 question IDs")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("CREATE TABLE questions (id INTEGER PRIMARY KEY)")
            db.execute("""CREATE TABLE claims (
                sequence INTEGER PRIMARY KEY,
                question_id INTEGER NOT NULL UNIQUE,
                lane INTEGER NOT NULL CHECK (lane IN (0, 1))
            )""")
            db.executemany("INSERT INTO questions VALUES (?)", [(question_id,) for question_id in ids])

    def claim(self, lane):
        if lane not in (0, 1):
            raise ValueError("Expected GPU lane 0 or 1")
        order = "ASC" if lane == 0 else "DESC"
        with closing(self.connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(f"""
                SELECT id FROM questions
                WHERE id NOT IN (SELECT question_id FROM claims)
                ORDER BY id {order} LIMIT 1
            """).fetchone()
            if row is None:
                return None
            question_id, = row
            db.execute("INSERT INTO claims (question_id, lane) VALUES (?, ?)", (question_id, lane))
            return question_id

    def audit_claims(self):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT question_id, lane FROM claims ORDER BY sequence").fetchall()
        if len(rows) != 500 or sorted(question_id for question_id, _ in rows) != list(range(500)):
            raise ValueError("Shared E4 queue did not claim all 500 questions exactly once")
        first, last = 0, 499
        for question_id, lane in rows:
            if lane == 0 and question_id == first:
                first += 1
            elif lane == 1 and question_id == last:
                last -= 1
            else:
                raise ValueError(f"Unexpected head/tail claim at question {question_id}")
        return [{"problem_id": question_id, "lane": lane} for question_id, lane in rows]
