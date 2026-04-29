from __future__ import annotations

import asyncio
import base64
import logging
import os
import time

import httpx

from db.mongo import update_job
from db.redis_client import dequeue_job

_semaphore = asyncio.Semaphore(int(os.getenv("WORKER_CONCURRENCY", "10")))
logger = logging.getLogger(__name__)

WORKER_POLL_INTERVAL = float(os.getenv("WORKER_POLL_INTERVAL", "0.5"))
WORKER_MAX_RETRIES   = int(os.getenv("WORKER_MAX_RETRIES",    "3"))
JOB_TIMEOUT_SECONDS  = float(os.getenv("JOB_TIMEOUT_SECONDS", "300"))
STORAGE_SERVICE_URL  = os.getenv("STORAGE_URL", "http://127.0.0.1:3001/upload/single")

SERVICE_DISPATCH_MAP: dict[str, str] = {
    "breast":        "http://127.0.0.1:8200/api/v1/breastpredict",
    "lung":          "http://127.0.0.1:8200/api/v1/lungpredict",
    "skin":          "http://127.0.0.1:8400/api/v1/skinclass",
    "blood":         "http://127.0.0.1:8500/api/v1/bloodclass",
    "bone_cancer":   "http://127.0.0.1:8600/api/v1/boneclass",
    "bone_fracture": "http://127.0.0.1:8700/api/v1/boneclass",
    "liver":         "http://127.0.0.1:8800/api/v1/liverpredict",
    "colon_cell":    "http://127.0.0.1:8900/api/v1/colonclass",
    "lung_cell":     "http://127.0.0.1:8200/api/v1/lungclass",
    "brain":         "http://127.0.0.1:9100/api/v1/brainpredict",
}

SEGMENTATION_LABELS   = {"lung", "liver", "brain"}
CLASSIFICATION_LABELS = {"skin", "blood", "bone_cancer", "colon_cell", "lung_cell"}


def _build_prediction_body(label: str, ai_data: dict) -> dict:
    if label in SEGMENTATION_LABELS:
        return {
            "prediction":       ai_data.get("prediction"),
            "confidence":       ai_data.get("confidence"),
            "type":             ai_data.get("type"),
            "all_scores":       ai_data.get("all_scores"),
            "mesh":             ai_data.get("mesh"),
            "patient_analysis": ai_data.get("patient_analysis"),
        }
    elif label in CLASSIFICATION_LABELS:
        return {
            "prediction":        ai_data.get("prediction"),
            "class_index":       ai_data.get("class_index"),
            "confidence":        ai_data.get("confidence"),
            "type":              ai_data.get("type"),
            "diagnostics":       ai_data.get("diagnostics"),
            "all_probabilities": ai_data.get("all_probabilities"),
            "original_image":    ai_data.get("original_image"),
        }
    else:
        return {
            "type":           ai_data.get("type", "2D"),
            "original_image": ai_data.get("original_image"),
            "visual_result":  ai_data.get("visual_result"),
        }


async def _process_job(job: dict) -> None:
    async with _semaphore:                             
        job_id       = job["job_id"]
        label        = job["label"]
        filename     = job["filename"]
        patient_id   = job["patientId"]
        file_type    = job["fileType"]
        content_type = job.get("content_type", "application/octet-stream")

        try:
            img_bytes = base64.b64decode(job["image_b64"])
        except Exception as exc:
            logger.error("job=%s failed to decode image_b64: %s", job_id, exc)
            await update_job(job_id, status="failed", error=f"Image decode error: {exc}")
            return

        logger.info("job=%s start  label=%s file=%s", job_id, label, filename)

        if time.time() - job.get("created_at", time.time()) > JOB_TIMEOUT_SECONDS:
            logger.warning("job=%s timed out in queue", job_id)
            await update_job(job_id, status="failed", error="Job timed out in queue")
            return

        await update_job(job_id, status="processing")

        target_url = SERVICE_DISPATCH_MAP.get(label)
        if not target_url:
            await update_job(job_id, status="failed", error=f"No service mapped for: {label}")
            return

        retries    = 0
        last_error = None

        while retries < WORKER_MAX_RETRIES:
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    ai_resp = await client.post(
                        target_url,
                        files={"file": (filename, img_bytes, content_type)},
                    )
                    ai_resp.raise_for_status()
                    ai_data = ai_resp.json()

                    combined_payload = {
                        "patientId":     patient_id,
                        "fileType":      file_type,
                        "modelName":     f"{label.replace('_', ' ').title()} AI",
                        "filescantype":  label,
                        "modelAccuracy": ai_data.get("confidence", "N/A"),
                        "prediction":    _build_prediction_body(label, ai_data),
                        "originalName":  filename,
                        "mimetype":      content_type,
                        "size":          len(img_bytes),
                    }

                    storage_resp = await client.post(STORAGE_SERVICE_URL, json=combined_payload)
                    storage_resp.raise_for_status()

                    await update_job(job_id, status="done", result=storage_resp.json())
                    logger.info("job=%s done", job_id)
                    return

            except Exception as exc:
                retries   += 1
                last_error = str(exc)
                logger.warning("job=%s retry=%d error=%s", job_id, retries, last_error)
                await asyncio.sleep(0.5 * retries)

        await update_job(job_id, status="failed", error=last_error)
        logger.error("job=%s permanently failed: %s", job_id, last_error)


def _on_task_done(task: asyncio.Task) -> None:
    if not task.cancelled() and task.exception():
        logger.error("Unhandled exception in job task: %s", task.exception())


async def queue_runner() -> None:
    logger.info("worker started — poll_interval=%.1fs", WORKER_POLL_INTERVAL)
    while True:
        try:
            job = await dequeue_job()
            if job is None:
                await asyncio.sleep(WORKER_POLL_INTERVAL)
                continue
            task = asyncio.create_task(_process_job(job))
            task.add_done_callback(_on_task_done)
        except Exception as exc:
            logger.error("queue_runner loop error: %s", exc)
            await asyncio.sleep(2)