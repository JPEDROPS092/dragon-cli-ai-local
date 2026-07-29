"""High-level conversation persistence API used by the chat layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .database import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Conversation:
    id: int
    title: str
    model_name: str | None
    created_at: str
    updated_at: str
    message_count: int = 0


@dataclass
class Message:
    role: str
    content: str
    created_at: str


class ConversationStore:
    """CRUD + search/export over the conversations/messages SQLite tables."""

    def __init__(self, db_path: Path):
        self.conn = connect(db_path)

    def create(self, title: str, model_name: str | None = None) -> int:
        now = _now()
        cur = self.conn.execute(
            "INSERT INTO conversations (title, model_name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, model_name, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def list(self) -> list[Conversation]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.title, c.model_name, c.created_at, c.updated_at,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            """
        ).fetchall()
        return [Conversation(**dict(r)) for r in rows]

    def get(self, conv_id: int) -> tuple[Conversation, list[Message]] | None:
        row = self.conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
        if row is None:
            return None
        msg_rows = self.conn.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conv_id,),
        ).fetchall()
        messages = [Message(**dict(m)) for m in msg_rows]
        conv = Conversation(
            id=row["id"], title=row["title"], model_name=row["model_name"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            message_count=len(messages),
        )
        return conv, messages

    def append_message(self, conv_id: int, role: str, content: str) -> None:
        now = _now()
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conv_id, role, content, now),
        )
        self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id))
        self.conn.commit()

    def rename(self, conv_id: int, title: str) -> bool:
        cur = self.conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title, _now(), conv_id)
        )
        self.conn.commit()
        return cur.rowcount > 0

    def delete(self, conv_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def search(self, text: str) -> list[tuple[int, str, str, str]]:
        rows = self.conn.execute(
            """
            SELECT c.id, c.title, m.role, m.content
            FROM messages m
            JOIN conversations c ON c.id = m.conversation_id
            WHERE m.content LIKE ?
            ORDER BY m.created_at DESC
            """,
            (f"%{text}%",),
        ).fetchall()
        return [(r["id"], r["title"], r["role"], r["content"]) for r in rows]

    def export(self, conv_id: int, path: Path, fmt: str = "md") -> Path | None:
        found = self.get(conv_id)
        if found is None:
            return None
        conv, messages = found
        if fmt == "json":
            payload = {
                "id": conv.id, "title": conv.title, "model_name": conv.model_name,
                "created_at": conv.created_at, "updated_at": conv.updated_at,
                "messages": [{"role": m.role, "content": m.content, "created_at": m.created_at} for m in messages],
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            lines = [f"# {conv.title}", f"_model: {conv.model_name or 'unknown'} · updated: {conv.updated_at}_", ""]
            for m in messages:
                if m.role == "system":
                    continue
                lines.append(f"**{m.role}:** {m.content}\n")
            path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def close(self) -> None:
        self.conn.close()
