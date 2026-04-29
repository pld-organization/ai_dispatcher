from __future__ import annotations

import logging
import os
import time
from typing import Optional

import motor.motor_asyncio

logger = logging.getLogger(__name__)

MONGO_URL     = os.getenv("MONGO_URL",     "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "ai_dispatcher")

_client:     motor.motor_asyncio.AsyncIOMotorClient | None = None
_collection: motor.motor_asyncio.AsyncIOMotorCollection | None = None


async def connect_mongo() -> None:
    global _client, _collection
    if _client is None:
        _client     = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
        _collection = _client[MONGO_DB_NAME]["dispatch_jobs"]
        await _client.admin.command("ping")
        logger.info("MongoDB connected → %s / %s", MONGO_URL, MONGO_DB_NAME)
    else:
        logger.info("[mongo] already connected")


async def close_mongo() -> None:
    global _client, _collection
    if _client:
        _client.close()
        _client     = None
        _collection = None
        logger.info("[mongo] connection closed")


def _get_collection():
    if _collection is None:
        raise RuntimeError("MongoDB not initialised — call connect_mongo() in lifespan")
    return _collection


async def create_job(
    job_id: str,
    label: str,
    filename: str,
    patient_id: str,
    file_type: str,
    content_type: str,
    image_b64: str,
) -> None:
    col = _get_collection()
    await col.insert_one({
        "_id":          job_id,
        "label":        label,
        "filename":     filename,
        "patientId":    patient_id,
        "fileType":     file_type,
        "content_type": content_type,
        "image_b64":    image_b64,
        "status":       "queued",
        "result":       None,
        "error":        None,
        "created_at":   time.time(),
        "updated_at":   time.time(),
    })


async def update_job(job_id: str, **fields) -> None:
    col = _get_collection()
    fields["updated_at"] = time.time()
    await col.update_one({"_id": job_id}, {"$set": fields})


async def get_job(job_id: str) -> Optional[dict]:
    col = _get_collection()
    doc = await col.find_one({"_id": job_id})
    if doc:
        doc["job_id"] = doc.pop("_id")
        doc.pop("image_b64", None)  # ne pas exposer les données binaires dans l'API
    return doc