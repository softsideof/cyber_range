"""
PlaybookStore — Persistent Analyst Skill Library.

SQLite-backed store for saving and retrieving successful SOC investigation
strategies (playbooks) across episodes.

Inspired by: Deepmind Ace (Finalist, OpenEnv SF Hackathon 2026).

Agents that save successful playbooks and search them at episode start
improve performance across episodes — institutional memory for the AI.

Usage:
    store = PlaybookStore.get_instance()

    # Save a successful strategy
    store.save(
        name="ransomware_triage",
        description="Effective for ransomware with 1 false positive alert",
        steps=["investigate_alert ALT-0001", "dismiss_alert ALT-0001", "block_ip 185.x.x.x", "isolate_host ws-01"],
        scenario_id="ransomware_outbreak",
    )

    # Retrieve relevant strategies
    matches = store.search("ransomware outbreak workstation", top_k=3)
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional


DB_PATH = Path(__file__).parent.parent / "outputs" / "playbooks.db"


class PlaybookStore:
    """
    Thread-safe singleton playbook store backed by SQLite.

    Each playbook entry stores:
    - Unique ID
    - Name and description (for search)
    - Ordered steps list
    - Source scenario ID
    - Score achieved (if known)
    - Usage count and success rate
    """

    _instance: Optional["PlaybookStore"] = None

    @classmethod
    def get_instance(cls) -> "PlaybookStore":
        """Get or create the singleton store."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        """Initialize the SQLite database."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(DB_PATH)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS playbooks (
                    id          TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    description TEXT NOT NULL,
                    steps       TEXT NOT NULL,
                    scenario_id TEXT NOT NULL DEFAULT '',
                    score       REAL NOT NULL DEFAULT 0.0,
                    use_count   INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL
                )
            """)
            # Full-text search virtual table for keyword matching
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS playbooks_fts
                USING fts5(id UNINDEXED, name, description, scenario_id)
            """)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save(
        self,
        name: str,
        description: str,
        steps: list,
        scenario_id: str = "",
        score: float = 0.0,
    ) -> str:
        """
        Save a playbook. Returns the generated playbook ID.

        If a playbook with the same name exists, updates it rather than creating duplicate.
        """
        # Sanitize name into slug
        playbook_id = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
        playbook_id = f"pb_{playbook_id}_{int(time.time())}"

        steps_json = json.dumps(steps, ensure_ascii=False)
        now = time.time()

        with self._connect() as conn:
            # Check for existing playbook with same name
            existing = conn.execute(
                "SELECT id FROM playbooks WHERE name = ?", (name,)
            ).fetchone()

            if existing:
                # Update existing
                conn.execute("""
                    UPDATE playbooks SET
                        description = ?,
                        steps = ?,
                        scenario_id = ?,
                        score = ?,
                        updated_at = ?
                    WHERE name = ?
                """, (description, steps_json, scenario_id, score, now, name))
                playbook_id = existing["id"]
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO playbooks
                        (id, name, description, steps, scenario_id, score, use_count, success_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                """, (playbook_id, name, description, steps_json, scenario_id, score, now, now))

                # Update FTS index
                conn.execute("""
                    INSERT INTO playbooks_fts (id, name, description, scenario_id)
                    VALUES (?, ?, ?, ?)
                """, (playbook_id, name, description, scenario_id))

        return playbook_id

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Search playbooks using keyword matching.

        Uses SQLite FTS5 for fast full-text search across name, description, and scenario_id.
        Falls back to LIKE search if FTS fails.

        Args:
            query: Natural language search string
            top_k: Maximum number of results to return

        Returns:
            List of playbook dicts with name, description, steps, and stats
        """
        if not query.strip():
            return self._get_recent(top_k)

        # Clean query for FTS5 (avoid syntax errors)
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip()
        words = [w for w in clean_query.split() if len(w) > 2]

        if not words:
            return self._get_recent(top_k)

        # Try FTS5 search first
        try:
            fts_query = " OR ".join(words)
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT p.id, p.name, p.description, p.steps, p.scenario_id,
                           p.score, p.use_count, p.success_count
                    FROM playbooks p
                    JOIN playbooks_fts fts ON p.id = fts.id
                    WHERE playbooks_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (fts_query, top_k)).fetchall()

                if rows:
                    return [self._row_to_dict(r) for r in rows]
        except sqlite3.OperationalError:
            pass

        # Fallback: LIKE search
        return self._like_search(words, top_k)

    def _like_search(self, words: list[str], top_k: int) -> list[dict]:
        """Fallback LIKE-based search."""
        with self._connect() as conn:
            conditions = " OR ".join(
                ["name LIKE ? OR description LIKE ? OR scenario_id LIKE ?"] * len(words)
            )
            params = []
            for w in words:
                like = f"%{w}%"
                params.extend([like, like, like])

            rows = conn.execute(
                f"SELECT id, name, description, steps, scenario_id, score, use_count, success_count "
                f"FROM playbooks WHERE ({conditions}) "
                f"ORDER BY score DESC, use_count DESC LIMIT ?",
                params + [top_k],
            ).fetchall()

            return [self._row_to_dict(r) for r in rows]

    def _get_recent(self, limit: int) -> list[dict]:
        """Get the most recently updated playbooks."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, steps, scenario_id, score, use_count, success_count "
                "FROM playbooks ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def record_usage(self, playbook_id: str, success: bool) -> None:
        """Track playbook usage outcome for success rate calculation."""
        with self._connect() as conn:
            if success:
                conn.execute(
                    "UPDATE playbooks SET use_count = use_count + 1, success_count = success_count + 1 WHERE id = ?",
                    (playbook_id,),
                )
            else:
                conn.execute(
                    "UPDATE playbooks SET use_count = use_count + 1 WHERE id = ?",
                    (playbook_id,),
                )

    def list_all(self) -> list[dict]:
        """Return all saved playbooks."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, description, steps, scenario_id, score, use_count, success_count "
                "FROM playbooks ORDER BY score DESC, use_count DESC"
            ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        """Return the total number of saved playbooks."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM playbooks").fetchone()
            return row["cnt"] if row else 0

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a DB row to a clean dict for API responses."""
        try:
            steps = json.loads(row["steps"])
        except (json.JSONDecodeError, TypeError):
            steps = []

        use_count = row["use_count"] or 0
        success_count = row["success_count"] or 0
        success_rate = (success_count / use_count) if use_count > 0 else None

        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "steps": steps,
            "scenario_id": row["scenario_id"],
            "score": round(row["score"], 3),
            "use_count": use_count,
            "success_rate": round(success_rate, 2) if success_rate is not None else "N/A",
        }
