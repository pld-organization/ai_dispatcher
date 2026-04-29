from __future__ import annotations

import json
import logging
import os

import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

REDIS_URL  = os.getenv("REDIS_URL",  "redis://localhost:6379")
QUEUE_NAME = os.getenv("QUEUE_NAME", "dispatch_queue")

_redis: aioredis.Redis | None = None


async def connect_redis() -> None:
    global _redis
    _redis = await aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    await _redis.ping()
    logger.info("[redis] connected → %s", REDIS_URL)


async def close_redis() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None
        logger.info("[redis] connection closed")


def _get_redis() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("Redis not initialised — call connect_redis() in lifespan")
    return _redis


async def enqueue_job(job: dict) -> None:
    r = _get_redis()
    await r.rpush(QUEUE_NAME, json.dumps(job))


async def dequeue_job() -> dict | None:
    r = _get_redis()
    raw = await r.lpop(QUEUE_NAME)
    if raw is None:
        return None
    return json.loads(raw)


async def queue_length() -> int:
    r = _get_redis()
    return await r.llen(QUEUE_NAME)