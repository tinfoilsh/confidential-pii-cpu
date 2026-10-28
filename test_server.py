"""Coordinator and forward-lane behaviour, run against a stubbed model so no
weights (and no torch) load."""

import asyncio
import struct
import threading
from array import array

import httpx
import pytest

import server
from chunking import plan_windows


class FakeEncoding:
    """Tokens are the whitespace-separated integers of the text."""

    def encode(self, text, allowed_special=None):
        return [int(tok) for tok in text.split()]


def text_of(n):
    return " ".join(str(i) for i in range(n))


CLASSES = 2


def fake_rows(tokens):
    """Row i is [token, position-in-window]: the second column makes a row
    depend on where the window was cut, like a truncated attention band."""
    return [[float(tok), float(i)] for i, tok in enumerate(tokens)]


def fake_forward(tokens):
    rows = fake_rows(tokens)
    return struct.pack(f"<{len(rows) * CLASSES}f", *(v for r in rows for v in r)), CLASSES


def rows_of(scores, classes):
    return memoryview(scores).cast("f", shape=(len(scores) // 4 // classes, classes)).tolist()


class Finisher:
    """Stands in for the decode stage; remembers what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, tokens, scores, classes):
        self.calls.append((text, tokens, rows_of(scores, classes) if tokens else []))
        return {
            "schema_version": 1,
            "summary": {},
            "text": text,
            "detected_spans": [
                {"label": "test", "start": 0, "end": 1, "text": "x", "placeholder": "<T>"}
            ]
            if tokens
            else [],
            "redacted_text": text,
        }


@pytest.fixture
def stubbed(monkeypatch):
    """Swap in the stubs without entering the app lifespan, so load_model never runs."""
    finisher = Finisher()
    monkeypatch.setattr(server, "_encoding", FakeEncoding())
    monkeypatch.setattr(server, "_semaphore", asyncio.Semaphore(1))
    monkeypatch.setattr(server, "forward_tokens", fake_forward)
    monkeypatch.setattr(server, "finish", finisher)
    monkeypatch.setattr(server, "WINDOW_TOKENS", 1024)
    monkeypatch.setattr(server, "WINDOW_OVERLAP", 128)
    monkeypatch.setattr(server, "FORWARD_URL", None)
    monkeypatch.setattr(server, "FORWARD_FANOUT", 4)
    monkeypatch.setattr(server, "_forward_client", None)
    return finisher


def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.app), base_url="http://test"
    )


def run(coro):
    return asyncio.run(coro)


def post_forward(c, tokens):
    return c.post(
        "/forward",
        content=array("i", tokens).tobytes(),
        headers={"Content-Type": "application/octet-stream"},
    )


def test_forward_scores_one_window_as_float32_bytes(stubbed):
    async def go():
        async with client() as c:
            return await post_forward(c, [5, 6, 7])

    response = run(go())
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers[server.SHAPE_HEADER] == f"3,{CLASSES}"
    assert response.content == fake_forward([5, 6, 7])[0]
    assert rows_of(response.content, CLASSES) == fake_rows([5, 6, 7])


def test_forward_rejects_empty_or_misaligned_window(stubbed):
    async def go():
        async with client() as c:
            empty = await post_forward(c, [])
            ragged = await c.post("/forward", content=b"abc")
            return empty.status_code, ragged.status_code

    assert run(go()) == (422, 422)


def test_forward_lane_is_serialized_by_the_semaphore(stubbed, monkeypatch):
    running = []
    peak = []
    release = threading.Event()

    def blocking_forward(tokens):
        running.append(1)
        peak.append(len(running))
        release.wait(timeout=5)
        running.pop()
        return fake_forward(tokens)

    monkeypatch.setattr(server, "forward_tokens", blocking_forward)

    async def go():
        async with client() as c:
            first = asyncio.create_task(post_forward(c, [1]))
            second = asyncio.create_task(post_forward(c, [2]))
            await asyncio.sleep(0.2)
            release.set()
            return await asyncio.gather(first, second)

    responses = run(go())
    assert [r.status_code for r in responses] == [200, 200]
    assert max(peak) == 1


def test_redact_in_process_stitches_windows_like_one_pass(stubbed):
    n = 2500
    plans = plan_windows(n, 1024, 128)
    assert len(plans) == 3

    async def go():
        async with client() as c:
            return await c.post("/redact", json={"text": text_of(n)})

    response = run(go())
    assert response.status_code == 200
    assert response.headers[server.INPUT_TOKENS_HEADER] == str(n)
    assert response.json()["detected_spans"][0]["label"] == "test"
    (text, tokens, rows), = stubbed.calls
    assert tokens == list(range(n))
    # The kept rows are the ones each token would get with a full-width
    # window on both sides; the position column is window-relative.
    expected = []
    for plan in plans:
        expected.extend(fake_rows(tokens[plan.start : plan.end])[plan.keep_start - plan.start : plan.keep_end - plan.start])
    assert rows == expected
    assert [r[0] for r in rows] == [float(t) for t in tokens]


def test_redact_empty_text_runs_no_windows(stubbed):
    async def go():
        async with client() as c:
            return await c.post("/redact", json={"text": ""})

    response = run(go())
    assert response.status_code == 200
    assert response.json()["detected_spans"] == []
    assert response.headers[server.INPUT_TOKENS_HEADER] == "0"
    assert stubbed.calls == [("", [], [])]


@pytest.fixture
def remote(stubbed, monkeypatch):
    """Route windows through a fake gateway /forward that records each call."""
    class Seen(list):
        fail_window = None
        in_flight = 0
        peak = 0

    seen = Seen()

    async def handler(request):
        tokens = array("i")
        tokens.frombytes(request.content)
        tokens = list(tokens)
        seen.append((request.headers.get(server.KEY_HASH_HEADER), tokens))
        ordinal = len(seen)
        seen.in_flight += 1
        seen.peak = max(seen.peak, seen.in_flight)
        await asyncio.sleep(0.01)  # let the other windows pile up
        seen.in_flight -= 1
        if ordinal == seen.fail_window:
            return httpx.Response(503, text="privacy filter is at capacity")
        payload, classes = fake_forward(tokens)
        return httpx.Response(
            200,
            content=payload,
            headers={
                "Content-Type": "application/octet-stream",
                server.SHAPE_HEADER: f"{len(tokens)},{classes}",
            },
        )

    monkeypatch.setattr(server, "FORWARD_URL", "http://gateway/forward")
    monkeypatch.setattr(
        server, "_forward_client", httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    return seen


def test_redact_fans_windows_out_through_the_gateway(stubbed, remote):
    n = 10 * 1024 + 7
    plans = plan_windows(n, 1024, 128)

    async def go():
        async with client() as c:
            return await c.post(
                "/redact",
                json={"text": text_of(n)},
                headers={server.KEY_HASH_HEADER: "caller-hash"},
            )

    response = run(go())
    assert response.status_code == 200
    assert len(remote) == len(plans)
    assert remote.peak == server.FORWARD_FANOUT  # bounded, but saturated
    assert {key for key, _ in remote} == {"caller-hash"}
    assert sorted(tokens[0] for _, tokens in remote) == [p.start for p in plans]
    (_, tokens, rows), = stubbed.calls
    assert [r[0] for r in rows] == [float(t) for t in tokens]


def test_redact_reports_capacity_when_the_gateway_refuses_a_window(stubbed, remote):
    remote.fail_window = 2

    async def go():
        async with client() as c:
            return await c.post("/redact", json={"text": text_of(5000)})

    response = run(go())
    assert response.status_code == 503
    assert stubbed.calls == []


def test_health(stubbed):
    async def go():
        async with client() as c:
            return await c.get("/health")

    assert run(go()).status_code == 200
