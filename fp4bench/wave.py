# METHODOLOGY.md#m1
import asyncio
import time

import aiohttp

from fp4bench.core.types import Prompt, PromptSet

ERROR_BODY_CHARS = 500


def _usage_counts(body: object) -> tuple[int, int]:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        raise RuntimeError(f"response is missing 'usage': {str(body)[:ERROR_BODY_CHARS]}")
    missing = [k for k in ("prompt_tokens", "completion_tokens") if usage.get(k) is None]
    if missing:
        raise RuntimeError(f"response usage is missing {missing}: {usage}")
    return usage["prompt_tokens"], usage["completion_tokens"]


async def _one(
    session: aiohttp.ClientSession, url: str, model: str, prompt_ids: Prompt, max_tokens: int
) -> None:
    payload = {
        "model": model,
        "prompt": prompt_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": False,
    }
    async with session.post(url, json=payload) as resp:
        if not 200 <= resp.status < 300:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status} from {url}: {text[:ERROR_BODY_CHARS]}")
        body = await resp.json()
    prompt_tokens, got = _usage_counts(body)
    if prompt_tokens != len(prompt_ids):
        raise RuntimeError(
            f"server counted usage.prompt_tokens={prompt_tokens} but we sent "
            f"{len(prompt_ids)} prompt token ids"
        )
    if got != max_tokens:
        raise RuntimeError(f"expected {max_tokens} completion tokens, got {got}")


async def run_wave_async(
    base_url: str, model: str, prompts: PromptSet, max_tokens: int, timeout_s: float = 1800.0
) -> float:
    """Send all prompts at once; return seconds until the last response arrives."""
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        url = f"{base_url}/v1/completions"
        t0 = time.perf_counter()
        await asyncio.gather(*(_one(session, url, model, p, max_tokens) for p in prompts))
        return time.perf_counter() - t0


def run_wave(base_url: str, model: str, prompts: PromptSet, max_tokens: int) -> float:
    return asyncio.run(run_wave_async(base_url, model, prompts, max_tokens))
