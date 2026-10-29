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


def fake_rows(tokens, position_offset=0):
    """Row i is [token, absolute position]: the second column records the
    position the window was run at, so a stitched result equals a single
    pass only if every window was given its true offset."""
    return [[float(tok), float(position_offset + i)] for i, tok in enumerate(tokens)]


def fake_forward(tokens, position_offset=0):
    rows = fake_rows(tokens, position_offset)
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
    monkeypatch.setattr(server, "QUEUE_URL", None)
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

    def blocking_forward(tokens, position_offset=0):
        running.append(1)
        peak.append(len(running))
        release.wait(timeout=5)
        running.pop()
        return fake_forward(tokens, position_offset)

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
            return await c.post("/chunk", json={"text": text_of(n)})

    response = run(go())
    assert response.status_code == 200
    assert response.headers[server.INPUT_TOKENS_HEADER] == str(n)
    assert response.json()["detected_spans"][0]["label"] == "test"
    (text, tokens, rows), = stubbed.calls
    assert tokens == list(range(n))
    # Every window ran at its absolute offset, so the stitched rows are
    # exactly what one pass over the whole sequence would produce.
    assert rows == fake_rows(tokens)
    assert [r[0] for r in rows] == [float(t) for t in tokens]


def test_redact_empty_text_runs_no_windows(stubbed):
    async def go():
        async with client() as c:
            return await c.post("/chunk", json={"text": ""})

    response = run(go())
    assert response.status_code == 200
    assert response.json()["detected_spans"] == []
    assert response.headers[server.INPUT_TOKENS_HEADER] == "0"
    assert stubbed.calls == [("", [], [])]


@pytest.fixture
def remote(stubbed, monkeypatch):
    """Route windows through a fake gateway /queue that records each call."""
    class Seen(list):
        fail_window = None
        # ordinal -> a reply to send instead of a good one, consumed once:
        # an httpx.Response, or an exception to raise (a transport error).
        faults = {}
        in_flight = 0
        peak = 0

    seen = Seen()
    seen.request_ids = set()

    async def handler(request):
        tokens = array("i")
        tokens.frombytes(request.content)
        tokens = list(tokens)
        seen.append(
            (
                request.headers.get(server.KEY_HASH_HEADER),
                tokens,
                request.headers.get(server.REQUEST_TOKENS_HEADER),
            )
        )
        seen.request_ids.add(request.headers.get(server.REQUEST_ID_HEADER))
        ordinal = len(seen)
        seen.in_flight += 1
        seen.peak = max(seen.peak, seen.in_flight)
        await asyncio.sleep(0.01)  # let the other windows pile up
        seen.in_flight -= 1
        if ordinal == seen.fail_window:
            return httpx.Response(503, text="privacy filter is at capacity")
        if ordinal in seen.faults:
            fault = seen.faults.pop(ordinal)
            if isinstance(fault, Exception):
                raise fault
            return fault
        # The gateway relays the window's absolute position to the worker;
        # the fake runs the window there, as the real one does.
        offset = int(request.headers.get(server.POSITION_OFFSET_HEADER, "0"))
        payload, classes = fake_forward(tokens, offset)
        return httpx.Response(
            200,
            content=payload,
            headers={
                "Content-Type": "application/octet-stream",
                server.SHAPE_HEADER: f"{len(tokens)},{classes}",
            },
        )

    monkeypatch.setattr(server, "QUEUE_URL", "http://gateway/queue")
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
                "/chunk",
                json={"text": text_of(n)},
                headers={server.KEY_HASH_HEADER: "caller-hash", server.REQUEST_ID_HEADER: "req-1"},
            )

    response = run(go())
    assert response.status_code == 200
    assert len(remote) == len(plans)
    # every window is posted at once: the gateway queue does the scheduling
    assert remote.peak == len(plans)
    assert remote.request_ids == {"req-1"}
    assert {key for key, _, _ in remote} == {"caller-hash"}
    assert {req_tokens for _, _, req_tokens in remote} == {str(n)}
    assert sorted(tokens[0] for _, tokens, _ in remote) == [p.start for p in plans]
    (_, tokens, rows), = stubbed.calls
    # Each window was posted with its absolute offset and the stitched rows
    # carry the true positions: exactly one pass over the whole sequence.
    assert rows == fake_rows(tokens)


def _chunk(n, **kw):
    async def go():
        async with client() as c:
            return await c.post("/chunk", json={"text": text_of(n)}, **kw)

    return run(go())


def test_window_is_retried_once_after_a_transport_error_or_malformed_reply(stubbed, remote):
    n = 5000
    plans = plan_windows(n, 1024, 128)
    assert len(plans) >= 3
    remote.faults = {
        1: httpx.ConnectError("worker connection reset"),
        2: httpx.Response(200, content=b"\x00" * 8, headers={server.SHAPE_HEADER: "1024,2"}),  # short body
        3: httpx.Response(200, content=b"\x00" * 8),  # no shape header
    }
    response = _chunk(n)
    assert response.status_code == 200
    assert len(remote) == len(plans) + 3  # three windows went twice
    assert not remote.faults
    (_, tokens, rows), = stubbed.calls
    assert rows == fake_rows(tokens)


def test_wedged_worker_window_is_requeued_once(stubbed, remote):
    n = 5000
    plans = plan_windows(n, 1024, 128)
    remote.faults = {2: httpx.Response(504, text="privacy filter forward timed out")}
    response = _chunk(n)
    assert response.status_code == 200
    assert len(remote) == len(plans) + 1
    (_, tokens, rows), = stubbed.calls
    assert rows == fake_rows(tokens)


def test_window_that_fails_twice_fails_the_request(stubbed, remote):
    # Every post fails (ordinals count posts, and with all windows posted at
    # once a retry's ordinal is not fixed): the first window to fail twice
    # gives up and the request is a 502, nothing partial.
    remote.faults = {k: httpx.ConnectError("reset") for k in range(1, 40)}
    response = _chunk(5000)
    assert response.status_code == 502
    assert stubbed.calls == []


def test_window_refused_as_no_longer_active_fails_fast(stubbed, remote):
    remote.faults = {1: httpx.Response(409, text="request no longer active")}
    response = _chunk(5000)
    assert response.status_code == 409
    assert stubbed.calls == []


def test_redact_reports_capacity_when_the_gateway_refuses_a_window(stubbed, remote):
    remote.fail_window = 2

    async def go():
        async with client() as c:
            return await c.post("/chunk", json={"text": text_of(5000)})

    response = run(go())
    assert response.status_code == 503
    assert stubbed.calls == []


def test_health(stubbed):
    async def go():
        async with client() as c:
            return await c.get("/health")

    assert run(go()).status_code == 200


def test_redact_refuses_inputs_over_the_token_cap(stubbed, monkeypatch):
    monkeypatch.setattr(server, "MAX_INPUT_TOKENS", 5000)

    async def go(n):
        async with client() as c:
            return await c.post("/chunk", json={"text": text_of(n)})

    assert run(go(5000)).status_code == 200
    over = run(go(5001))
    assert over.status_code == 413
    assert "5001 tokens" in over.json()["detail"] and "5000" in over.json()["detail"]
    assert len(stubbed.calls) == 1  # the refused input never reached the model


def test_rotary_rows_match_the_full_table():
    """The per-window rows must equal the corresponding rows of the module's
    own table: that is what makes a window at an offset exactly a slice of
    the single pass."""
    torch = pytest.importorskip("torch")

    class Rope:
        dtype = torch.float32

        def __init__(self):
            self.inv_freq = 1.0 / (10000 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
            t = torch.arange(300, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            self.cos_cache, self.sin_cache = freqs.cos() * 1.5, freqs.sin() * 1.5
            self._tinfoil_cos_default, self._tinfoil_sin_default = self.cos_cache, self.sin_cache

        def _compute_concentration_and_inv_freq(self, device=None):
            return 1.5, self.inv_freq

    rope = Rope()
    cos, sin = server._rotary_rows(rope, 100, 50)
    assert cos.shape == (50, 4)
    assert torch.equal(cos, rope._tinfoil_cos_default[100:150])
    assert torch.equal(sin, rope._tinfoil_sin_default[100:150])
    # beyond the module's own table, same formula
    cos2, _ = server._rotary_rows(rope, 1_000_000, 3)
    assert cos2.shape == (3, 4) and torch.isfinite(cos2).all()
