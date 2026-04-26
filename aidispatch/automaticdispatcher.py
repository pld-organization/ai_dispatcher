import httpx
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from medical_classifier_api2 import classify_image_bytes

router = APIRouter()

MODEL_PATH = "./model/medical_router_v1.pth"

STORAGE_SERVICE_URL = os.getenv("STORAGE_URL", "http://127.0.0.1:3001/upload/single")

SERVICE_DISPATCH_MAP = {
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

SEGMENTATION_LABELS = {"lung", "liver", "brain"}
CLASSIFICATION_LABELS = {"skin", "blood", "bone_cancer", "colon_cell", "lung_cell"}


@router.post("/auto-dispatch-store")
async def handle_auto_dispatch(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
):
    file_bytes = await file.read()

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
         
            router_result = classify_image_bytes(
                file_bytes=file_bytes,
                filename=file.filename,
                model_path=MODEL_PATH
            )

            label = router_result["label"]
            target_url = SERVICE_DISPATCH_MAP.get(label)

            if not target_url:
                raise HTTPException(422, f"No service mapped for: {label}")

            # Forward to AI service
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_resp = await client.post(target_url, files=ai_files)
            ai_resp.raise_for_status()

            ai_data = ai_resp.json()

            # Payload shaping
            if label in SEGMENTATION_LABELS:
                prediction_body = {
                    "prediction": ai_data.get("prediction"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "all_scores": ai_data.get("all_scores"),
                    "mesh": ai_data.get("mesh"),
                    "patient_analysis": ai_data.get("patient_analysis"),
                }

            elif label in CLASSIFICATION_LABELS:
                prediction_body = {
                    "prediction": ai_data.get("prediction"),
                    "class_index": ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "all_probabilities": ai_data.get("all_probabilities"),
                    "original_image": ai_data.get("original_image")
                }

            else:
                prediction_body = {
                    "type": ai_data.get("type", "2D"),
                    "original_image": ai_data.get("original_image"),
                    "visual_result": ai_data.get("visual_result")
                }

            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": f"{label.replace('_', ' ').title()} AI",
                "filescantype": label,
                "modelAccuracy": ai_data.get("confidence", "98.2%"),
                "prediction": prediction_body,
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }

            storage_response = await client.post(STORAGE_SERVICE_URL, json=combined_payload)
            storage_response.raise_for_status()

            return storage_response.json()

        except httpx.HTTPStatusError as e:
            raise HTTPException(e.response.status_code, f"AI Service Error: {e.response.text}")

        except Exception as e:
            raise HTTPException(500, f"Internal Router Error: {str(e)}")