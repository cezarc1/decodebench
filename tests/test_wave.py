import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from fp4bench.wave import run_wave_async


def _check_payload_contract(body: dict) -> None:
    assert body["temperature"] == 0.0
    assert body["ignore_eos"] is True
    assert body["stream"] is False
    assert isinstance(body["prompt"], list) and all(type(t) is int for t in body["prompt"])


def _app(n_expected: int, token_delta: int, prompt_delta: int = 0) -> web.Application:
    state = {"arrived": 0, "all_in": asyncio.Event()}

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        _check_payload_contract(body)
        state["arrived"] += 1
        if state["arrived"] == n_expected:
            state["all_in"].set()
        await asyncio.wait_for(state["all_in"].wait(), timeout=5)
        return web.json_response(
            {
                "usage": {
                    "prompt_tokens": len(body["prompt"]) + prompt_delta,
                    "completion_tokens": body["max_tokens"] + token_delta,
                }
            }
        )

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    return app


def _fixed_reply_app(status: int, *, text: str | None = None, data: dict | None = None):
    async def handler(request: web.Request) -> web.Response:
        await request.json()
        if data is not None:
            return web.json_response(data, status=status)
        return web.Response(status=status, text=text)

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    return app


async def _run_against(app: web.Application, n: int = 1) -> float:
    server = TestServer(app)
    await server.start_server()
    try:
        base = str(server.make_url("")).rstrip("/")
        return await run_wave_async(base, "m", [[1, 2, 3]] * n, max_tokens=7)
    finally:
        await server.close()


async def _run(n: int, token_delta: int = 0, prompt_delta: int = 0) -> float:
    return await _run_against(_app(n, token_delta, prompt_delta), n)


def test_wave_keeps_all_requests_in_flight_at_once():
    assert asyncio.run(_run(120)) < 5.0


def test_wave_rejects_short_completions():
    with pytest.raises(RuntimeError, match="expected 7"):
        asyncio.run(_run(3, token_delta=-1))


def test_wave_rejects_prompt_token_mismatch():
    with pytest.raises(RuntimeError, match=r"prompt_tokens=4.*sent 3 prompt token ids"):
        asyncio.run(_run(2, prompt_delta=1))


def test_wave_surfaces_server_error_body():
    app = _fixed_reply_app(400, data={"error": {"message": "maximum context length is 4096"}})
    with pytest.raises(RuntimeError, match=r"400.*maximum context length is 4096"):
        asyncio.run(_run_against(app))


def test_wave_truncates_error_body_to_500_chars():
    app = _fixed_reply_app(500, text="A" * 500 + "TAIL")
    with pytest.raises(RuntimeError) as err:
        asyncio.run(_run_against(app))
    assert "A" * 500 in str(err.value) and "TAIL" not in str(err.value)


@pytest.mark.parametrize(
    "data,missing",
    [
        ({}, "usage"),
        ({"usage": None}, "usage"),
        ({"usage": {"completion_tokens": 7}}, "prompt_tokens"),
        ({"usage": {"prompt_tokens": 3}}, "completion_tokens"),
    ],
)
def test_wave_rejects_missing_usage(data, missing):
    with pytest.raises(RuntimeError, match=f"missing.*{missing}"):
        asyncio.run(_run_against(_fixed_reply_app(200, data=data)))
