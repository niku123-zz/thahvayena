"""SQLite storage for analysis history (stdlib only, one file on disk)."""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.getenv("SENTIMENT_DB", "./data/history.db")


@contextmanager
def connect():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init():
    with connect() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT NOT NULL,
                source       TEXT NOT NULL,      -- "single", "batch" or "csv"
                text         TEXT NOT NULL,
                sentiment    TEXT NOT NULL,
                confidence   REAL NOT NULL,
                score        REAL NOT NULL,      -- positive minus negative, -1..1
                needs_review INTEGER NOT NULL,
                aspects      TEXT NOT NULL       -- JSON {"acting": 0.9, ...}
            )
        """)


def save(results, source):
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with connect() as c:
        c.executemany(
            "INSERT INTO analyses (created_at, source, text, sentiment, confidence,"
            " score, needs_review, aspects) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(now, source, r["text"], r["sentiment"], r["confidence"], r["score"],
              int(r["needs_review"]), json.dumps(r["aspects"])) for r in results],
        )


def _row(r):
    d = dict(r)
    d["needs_review"] = bool(d["needs_review"])
    d["aspects"] = json.loads(d["aspects"])
    return d


def history(limit=50):
    with connect() as c:
        rows = c.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row(r) for r in rows]


def all_rows():
    with connect() as c:
        return [_row(r) for r in c.execute("SELECT * FROM analyses ORDER BY id")]


def stats():
    """Totals, per-aspect averages and a daily trend for the dashboard."""
    rows = all_rows()
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    aspects, daily = {}, {}
    for r in rows:
        counts[r["sentiment"]] += 1
        for a, s in r["aspects"].items():
            aspects.setdefault(a, []).append(s)
        day = daily.setdefault(r["created_at"][:10], {"n": 0, "score": 0.0})
        day["n"] += 1
        day["score"] += r["score"]

    return {
        "total": len(rows),
        "counts": counts,
        "needs_review": sum(r["needs_review"] for r in rows),
        "avg_score": round(sum(r["score"] for r in rows) / len(rows), 4) if rows else 0,
        "aspects": {a: {"avg": round(sum(v) / len(v), 4), "mentions": len(v)}
                    for a, v in aspects.items()},
        "trend": [{"date": d, "count": v["n"], "avg_score": round(v["score"] / v["n"], 4)}
                  for d, v in sorted(daily.items())],
    }


def clear():
    with connect() as c:
        c.execute("DELETE FROM analyses")
