from __future__ import annotations

import uuid

MAX_TURNS = 10


class SessionManager:
    """In-memory conversation session store with turn bounding."""

    def __init__(self) -> None:
        self._sessions: dict[str, list[dict[str, str]]] = {}

    def get_or_create(self, session_id: str | None) -> str:
        if session_id and session_id in self._sessions:
            return session_id
        new_id = session_id or str(uuid.uuid4())
        self._sessions[new_id] = []
        return new_id

    def get_history(self, session_id: str) -> list[dict[str, str]]:
        return self._sessions.get(session_id, [])

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        self._sessions[session_id] = self._sessions[session_id][-MAX_TURNS * 2 :]
