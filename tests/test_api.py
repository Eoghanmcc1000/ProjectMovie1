from __future__ import annotations

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_endpoint(client: AsyncClient) -> None:
    resp = await client.post("/chat", json={"message": "Tell me about Inception"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert "session_id" in data
    assert "metadata" in data
    assert "retrieval_strategy" in data["metadata"]


@pytest.mark.asyncio
async def test_chat_session_continuity(client: AsyncClient) -> None:
    resp1 = await client.post("/chat", json={"message": "Tell me about Inception"})
    session_id = resp1.json()["session_id"]

    resp2 = await client.post(
        "/chat",
        json={"message": "What about action movies?", "session_id": session_id},
    )
    assert resp2.status_code == 200
    assert resp2.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_chat_empty_message_rejected(client: AsyncClient) -> None:
    resp = await client.post("/chat", json={"message": ""})
    assert resp.status_code == 422
