"""Admission control for /redact, run against a stubbed model so no weights load."""

import asyncio
import threading

import httpx
import pytest

import server


class FakeEncoding:
    def encode(self, text, allowed_special=None):
        return text.split()


class FakeResult:
    def to_dict(self):
        return {
            "schema_version": 1,
            "summary": {},
            "text": "",
            "detected_spans": [],
            "redacted_text": "",
        }


class BlockingModel:
    """Holds the inference slot until released, like one long forward pass."""

    def __init__(self):
        self.release = threading.Event()
        self.started = threading.Event()

    def redact(self, text):
        self.started.set()
        self.release.wait(timeout=10)
        return FakeResult()


@pytest.fixture
def model(monkeypatch):
    """Swap in the stub without entering the app lifespan, so load_model never runs."""
    model = BlockingModel()
    monkeypatch.setattr(server, "_opf", model)
    monkeypatch.setattr(server, "_encoding", FakeEncoding())
    monkeypatch.setattr(server, "_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(server, "_queued", 0)
    monkeypatch.setattr(server, "MAX_INPUT_TOKENS", 4)
    monkeypatch.setattr(server, "MAX_QUEUE_DEPTH", 2)
    return model


def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    )


async def redact(c, text="short"):
    return await c.post("/redact", json={"text": text})


def test_rejects_input_over_token_limit_without_inference(model):
    async def run():
        async with client() as c:
            return await redact(c, "one two three four five")

    response = asyncio.run(run())
    assert response.status_code == 413
    assert "5 tokens" in response.json()["detail"]
    assert not model.started.is_set()


def test_health_fails_while_queue_is_full(model):
    async def run():
        async with client() as c:
            server._queued = server.MAX_QUEUE_DEPTH
            full = await c.get("/health")
            server._queued = 0
            return full, await c.get("/health")

    full, after = asyncio.run(run())
    assert full.status_code == 503
    assert after.status_code == 200


def test_sheds_load_once_queue_is_full(model):
    async def run():
        async with client() as c:
            first = asyncio.create_task(redact(c))
            await asyncio.to_thread(model.started.wait, 5)
            second = asyncio.create_task(redact(c))
            while server._queued < 2:
                await asyncio.sleep(0.01)
            # One running, one waiting: the queue is full, so this must not wait.
            shed = await redact(c)
            model.release.set()
            done = await asyncio.gather(first, second)
            after = await redact(c)
            return shed, done, after

    shed, done, after = asyncio.run(run())
    assert shed.status_code == 503
    assert [r.status_code for r in done] == [200, 200]
    assert server._queued == 0
    assert after.status_code == 200
