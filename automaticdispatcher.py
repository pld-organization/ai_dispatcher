from __future__ import annotations
import traceback
import base64
import os
import time
import uuid
from typing import List

import httpx
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import json
from medical_classifier_api2 import classify_image_bytes
from db.mongo import create_job, get_job, update_job
from db.redis_client import enqueue_job, queue_length

router = APIRouter()

MODEL_PATH          = "./model/medical_router_v1.pth"
STORAGE_SERVICE_URL = os.getenv("STORAGE_URL", "https://storage-service-yxqy.onrender.com/upload/single")
STORAGE_SERVICE_URL_MULTIPLE = os.getenv("STORAGE_URL", "https://storage-service-yxqy.onrender.com/upload/multiple")


SERVICE_DISPATCH_MAP = {
    "breast":        "http://127.0.0.1:8000/api/v1/breastpredict",
    "lung":          "http://127.0.0.1:8000/api/v1/lungpredict",
    "skin":          "https://repoai-0nq6.onrender.com/api/v1/predict",
    "blood":         "http://127.0.0.1:8000/api/v1/bloodclass",
    "bone_cancer":   "https://bone-cancer-api-qg7x.onrender.com/predict",
    "bone_fracture": "http://127.0.0.1:8000/api/v1/boneclass",
    "liver":         "http://127.0.0.1:8000/api/v1/liverpredict",
    "colon_cell":    "http://127.0.0.1:8000/api/v1/colon/predict",
    "lung_cell":     "http://127.0.0.1:8000/api/v1/lungcellpredict",
    "brain":         "http://127.0.0.1:8000/api/v1/brainpredict",
}

SEGMENTATION_LABELS   = {"lung", "liver", "brain"}
CLASSIFICATION_LABELS = {"skin", "blood", "bone_cancer", "colon_cell", "lung_cell"}

LONG_TIMEOUT = httpx.Timeout(3600.0, connect=10.0, read=3600.0)

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


# POST /api/v1/auto-dispatch-store  (synchrone, 1 fichier)

@router.post("/auto-dispatch-store")
async def handle_auto_dispatch(
    file:      UploadFile = File(...),
    patientId: str        = Form(...),
    fileType:  str        = Form(...),
):
    file_bytes = await file.read()

    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # 1. Get prediction from AI model
            router_result = classify_image_bytes(file_bytes=file_bytes, filename=file.filename, model_path=MODEL_PATH)
            label = router_result["label"]
            target_url = SERVICE_DISPATCH_MAP.get(label)

            if not target_url:
                raise HTTPException(422, f"No service mapped for: {label}")

            ai_resp = await client.post(
                target_url,
                files={"file": (file.filename, file_bytes, file.content_type)},
            )
            ai_resp.raise_for_status()
            ai_data = ai_resp.json()

            # 2. Extract Mesh and convert to bytes only if needed
            mesh_data = ai_data.get("mesh")
            
            # 3. Build the CLEAN prediction body (WITHOUT the raw mesh)
            prediction_body = _build_prediction_body(label, ai_data)
            if isinstance(prediction_body, dict) and "mesh" in prediction_body:
                del prediction_body["mesh"]

            combined_payload = {
                "patientId":     patientId,
                "modelName":     f"{label.replace('_', ' ').title()} AI",
                "filescantype":  label,
                "modelAccuracy": ai_data.get("confidence", "98.2%"),
                "prediction":    json.dumps(prediction_body),
                "originalName":  file.filename,
                "mimetype":      file.content_type,
                "size":          str(len(file_bytes)),
            }

            # 4. Handle 3D (Multiple files under 'files' key)
            if ai_data.get("type") == "3D":
                storage_files = [
                    ("files", (file.filename, file_bytes, file.content_type))
                ]
                if mesh_data:
                    mesh_json_bytes = json.dumps(mesh_data).encode('utf-8')
                    storage_files.append(
                        ("files", (f"mesh_{patientId}.json", mesh_json_bytes, "application/json"))
                    )
                
                storage_response = await client.post(
                    STORAGE_SERVICE_URL_MULTIPLE, # This is port 3001/upload/multiple
                    data=combined_payload,
                    files=storage_files
                )
            
            # 5. Handle 2D (Single file under 'file' key)
            else:
                storage_files = [
                    ("file", (file.filename, file_bytes, file.content_type)) # Key changed to 'file'
                ]
                storage_response = await client.post(
                    STORAGE_SERVICE_URL, # This is port 3001/upload/single
                    data=combined_payload,
                    files=storage_files
                )
            
            storage_response.raise_for_status()
            return storage_response.json()

        except httpx.HTTPStatusError as e:
            print("🔥 HTTP ERROR FROM SERVICE")
            print("URL:", e.request.url)
            print("STATUS:", e.response.status_code)
            print("BODY:", e.response.text)

            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "error": "Downstream service error",
                    "status": e.response.status_code,
                    "response": e.response.text,
                }
            )

        except Exception as e:
            print("🔥 INTERNAL ERROR")
            print(traceback.format_exc())

            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Internal pipeline failure",
                    "message": str(e),
                    "trace": traceback.format_exc()
                }
            )


# POST /api/v1/auto-dispatch-queue  (async, N fichiers) 
@router.post("/auto-dispatch-queue")
async def handle_auto_dispatch_queued(
    files:     List[UploadFile] = File(...),
    patientId: str              = Form(...),
    fileType:  str              = Form(...),
):
    submitted = []
    errors    = []

    for file in files:
        try:
            img_bytes = await file.read()
            filename  = file.filename or "unknown"

            router_result = classify_image_bytes(
                file_bytes=img_bytes,
                filename=filename,
                model_path=MODEL_PATH,
            )
            label  = router_result["label"]
            job_id = str(uuid.uuid4())

            await create_job(
                job_id=job_id,
                label=label,
                filename=filename,
                patient_id=patientId,
                file_type=fileType,
                content_type=file.content_type or "application/octet-stream",
                image_b64=base64.b64encode(img_bytes).decode(),
            )

            await enqueue_job({
                "job_id":       job_id,
                "label":        label,
                "filename":     filename,
                "patientId":    patientId,
                "fileType":     fileType,
                "content_type": file.content_type or "application/octet-stream",
                "image_b64":    base64.b64encode(img_bytes).decode(), 
                "created_at":   time.time(),
            })

            submitted.append({
                "job_id":   job_id,
                "filename": filename,
                "label":    label,
                "status":   "queued",
            })

        except Exception as exc:
            errors.append({
                "filename": getattr(file, "filename", "unknown"),
                "error":    str(exc),
            })

    q_size = await queue_length()
    return JSONResponse({
        "submitted":  submitted,
        "errors":     errors,
        "queue_size": q_size,
    })


# GET /api/v1/jobs/{job_id} 
@router.get("/jobs/{job_id}")
async def get_job_result(job_id: str):
    doc = await get_job(job_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return doc


# POST /api/v1/jobs/{job_id}/retry 
@router.post("/jobs/{job_id}/retry")
async def retry_job(job_id: str):
    doc = await get_job(job_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    if doc.get("status") != "failed":
        raise HTTPException(status_code=400, detail=f"Job is not in failed state (status={doc.get('status')})")
    if not doc.get("image_b64"):
        raise HTTPException(status_code=400, detail="Job has no image data stored — cannot retry")

    await enqueue_job({
        "job_id":       job_id,
        "label":        doc["label"],
        "filename":     doc["filename"],
        "patientId":    doc["patientId"],
        "fileType":     doc["fileType"],
        "content_type": doc.get("content_type", "application/octet-stream"),
        "image_b64":    doc["image_b64"],
        "created_at":   time.time(),
    })

    await update_job(job_id, status="queued", error=None)

    return {"job_id": job_id, "status": "requeued"}


# GET /api/v1/queue/status 
@router.get("/queue/status")
async def get_queue_status():
    return {"queue_length": await queue_length()}
