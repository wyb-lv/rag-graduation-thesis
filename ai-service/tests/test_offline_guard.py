"""conftest.no_external_network: test thuong khong duoc goi mang that (router Gemini, Groq luon bat nen se tru han muc)."""
import asyncio
import socket

import httpx
import pytest


def test_external_hosts_cannot_be_resolved():
    with pytest.raises(socket.gaierror):
        socket.getaddrinfo("generativelanguage.googleapis.com", 443)


def test_localhost_still_resolves_for_qdrant_and_test_clients():
    assert socket.getaddrinfo("127.0.0.1", 6333)
    assert socket.getaddrinfo("localhost", 6333)


def test_an_async_http_client_cannot_reach_the_internet():
    """Duong ma router_llm dung (httpx async). Khong gui khoa nen neu chot chan hong thi cung khong tru han muc Gemini."""

    async def call():
        async with httpx.AsyncClient(timeout=5) as client:
            await client.get("https://generativelanguage.googleapis.com/")

    with pytest.raises(httpx.ConnectError):
        asyncio.run(call())
