import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException , APIRouter
from typing import Optional
import traceback
import logging
import json
router = APIRouter()

LONG_TIMEOUT = httpx.Timeout(3600.0, connect=10.0, read=3600.0)
# URLs for your other services 
# needs to be in an env file i know 
AI_BREAST_SERVICE_URL = "http://127.0.0.1:8000/api/v1/breastpredict"
AI_LUNG_SERVICE_URL = "http://127.0.0.1:8000/api/v1/lungpredict"
AI_SKIN_SERVICE_URL = "http://127.0.0.1:8000/api/v1/skinclass"
AI_BLOOD_SERVICE_URL = "http://127.0.0.1:8000/api/v1/bloodclass"
AI_BONECANCER_SERVICE_URL = "http://127.0.0.1:8000/api/v1/boneclass"
AI_BONEFRACTURE_SERVICE_URL = "http://127.0.0.1:8000/api/v1/boneclass"
AI_LIVER_SERVICE_URL = "http://127.0.0.1:8000/api/v1/liverpredict"
AI_COLONCELL_SERVICE_URL = "http://127.0.0.1:8000/api/v1/colonclass"
AI_LUNGCELL_SERVICE_URL = "http://127.0.0.1:8000/api/v1/lungcellpredict"
AI_BRAIN_SERVICE_URL = "http://127.0.0.1:8000/api/v1/brainpredict"
AI_BLOODANALYSIS_SERVICE_URL = "http://127.0.0.1:8000/api/v1/bloodanalysis"

STORAGE_SERVICE_URL = "http://127.0.0.1:3001/upload/single"



#change
@router.post("/lung-classify-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("lungcell Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            ai_response = await client.post(
                AI_LUNGCELL_SERVICE_URL,
                files=ai_files
            )

            ai_response.raise_for_status()
            ai_data = ai_response.json()
            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            storage_data = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "lungcell",
                "prediction": json.dumps({
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "original_image":ai_data.get("original_image")    
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }
            
            storage_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }
            # --- STEP 3: SEND COMBINED DATA TO STORAGE SERVICE ---
            # Sending the merged JSON to your NestJS storage service
            storage_response = await client.post(
                STORAGE_SERVICE_URL,
                data=storage_data,
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


@router.post("/breast-predict-stores")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Breast Cancer AI")
):

    file_bytes = await file.read()

    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # ─────────────────────────────
            # STEP 1: AI SERVICE CALL
            # ─────────────────────────────
            ai_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            ai_response = await client.post(
                AI_BREAST_SERVICE_URL,
                files=ai_files
            )

            ai_response.raise_for_status()
            ai_data = ai_response.json()

            # ─────────────────────────────
            # STEP 2: PREPARE STORAGE DATA
            # ─────────────────────────────
            storage_data = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "breast",
                "modelAccuracy": "98.2%",
                "prediction": json.dumps({
                    "type": ai_data.get("type", "2D"),
                    "original_image": ai_data.get("original_image"),
                    "visual_result": ai_data.get("visual_result")
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": str(len(file_bytes))
            }

            storage_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            # ─────────────────────────────
            # STEP 3: SEND TO NESTJS STORAGE
            # ─────────────────────────────
            storage_response = await client.post(
                STORAGE_SERVICE_URL,
                data=storage_data,
                files=storage_files
            )

            storage_response.raise_for_status()

            return storage_response.json()

        # ─────────────────────────────
        # ERROR HANDLING (CLEAN)
        # ─────────────────────────────
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

@router.post("/skin-classification-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("skin Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            ai_response = await client.post(
                AI_SKIN_SERVICE_URL ,
                files=ai_files
            )

            ai_response.raise_for_status()
            ai_data = ai_response.json()
            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            storage_data = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "skin cancer",
                "prediction": json.dumps({
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "original_image":ai_data.get("original_image")    
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }
            
            storage_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }
            # --- STEP 3: SEND COMBINED DATA TO STORAGE SERVICE ---
            # Sending the merged JSON to your NestJS storage service
            storage_response = await client.post(
                STORAGE_SERVICE_URL,
                data=storage_data,
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




@router.post("/blood-classify-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Blood Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            ai_response = await client.post(
                AI_BLOOD_SERVICE_URL,
                files=ai_files
            )

            ai_response.raise_for_status()
            ai_data = ai_response.json()
            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            storage_data = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "blood cancer",
                "prediction": json.dumps({
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "original_image":ai_data.get("original_image")    
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }
            
            storage_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }
            # --- STEP 3: SEND COMBINED DATA TO STORAGE SERVICE ---
            # Sending the merged JSON to your NestJS storage service
            storage_response = await client.post(
                STORAGE_SERVICE_URL,
                data=storage_data,
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


#change
@router.post("/liver-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Liver Cancer AI")
):
    file_bytes = await file.read()

    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # STEP 1: AI CALL
            ai_response = await client.post(
                AI_LIVER_SERVICE_URL,
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()

            mesh_data = ai_data.get("mesh")

            # STEP 2: CLEAN PREDICTION
            prediction_body = {
                "prediction": ai_data.get("prediction"),
                "confidence": ai_data.get("confidence"),
                "type": ai_data.get("type"),
                "all_scores": ai_data.get("all_scores"),
                "patient_analysis": ai_data.get("patient_analysis"),
            }

            # STEP 3: PAYLOAD
            combined_payload = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "liver",
                "modelAccuracy": ai_data.get("confidence", "98%"),
                "prediction": json.dumps(prediction_body),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes),
            }

            # STEP 4: FILE HANDLING

            storage_files = [
                ("files", (file.filename, file_bytes, file.content_type))
            ]
            if mesh_data:
                mesh_json_bytes = json.dumps(mesh_data).encode("utf-8")
                storage_files.append(
                    ("files", (f"mesh_{patientId}.json", mesh_json_bytes, "application/json"))
                )
            storage_response = await client.post(
                STORAGE_SERVICE_URL_MULTIPLE,
                data=combined_payload,
                files=storage_files
            )

            

            storage_response.raise_for_status()
            return storage_response.json()

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
@router.post("/brain-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("brain Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # STEP 1: AI CALL
            ai_response = await client.post(
                AI_BRAIN_SERVICE_URL,
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()

            mesh_data = ai_data.get("mesh")

            # STEP 2: CLEAN PREDICTION
            prediction_body = {
                "prediction": ai_data.get("prediction"),
                "confidence": ai_data.get("confidence"),
                "type": ai_data.get("type"),
                "all_scores": ai_data.get("all_scores"),
                "patient_analysis": ai_data.get("patient_analysis"),
            }

            # STEP 3: PAYLOAD
            combined_payload = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "Brain",
                "modelAccuracy": ai_data.get("confidence", "98%"),
                "prediction": json.dumps(prediction_body),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes),
            }

            # STEP 4: FILE HANDLING

            storage_files = [
                ("files", (file.filename, file_bytes, file.content_type))
            ]
            if mesh_data:
                mesh_json_bytes = json.dumps(mesh_data).encode("utf-8")
                storage_files.append(
                    ("files", (f"mesh_{patientId}.json", mesh_json_bytes, "application/json"))
                )
            storage_response = await client.post(
                STORAGE_SERVICE_URL_MULTIPLE,
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



#change
@router.post("/lung-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Lung Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # STEP 1: AI CALL
            ai_response = await client.post(
                AI_LUNG_SERVICE_URL,
                files={"file": (file.filename, file_bytes, file.content_type)}
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()

            mesh_data = ai_data.get("mesh")

            # STEP 2: CLEAN PREDICTION
            prediction_body = {
                "prediction": ai_data.get("prediction"),
                "confidence": ai_data.get("confidence"),
                "type": ai_data.get("type"),
                "all_scores": ai_data.get("all_scores"),
                "patient_analysis": ai_data.get("patient_analysis"),
            }

            # STEP 3: PAYLOAD
            combined_payload = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "Lung",
                "modelAccuracy": ai_data.get("confidence", "98%"),
                "prediction": json.dumps(prediction_body),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes),
            }

            # STEP 4: FILE HANDLING

            storage_files = [
                ("files", (file.filename, file_bytes, file.content_type))
            ]
            if mesh_data:
                mesh_json_bytes = json.dumps(mesh_data).encode("utf-8")
                storage_files.append(
                    ("files", (f"mesh_{patientId}.json", mesh_json_bytes, "application/json"))
                )
            storage_response = await client.post(
                STORAGE_SERVICE_URL_MULTIPLE,
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




#change
@router.post("/bone-classification-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("bone Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            ai_response = await client.post(
                AI_BONECANCER_SERVICE_URL,
                files=ai_files
            )

            ai_response.raise_for_status()
            ai_data = ai_response.json()
            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            storage_data = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "bone cancer",
                "prediction": json.dumps({
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "original_image":ai_data.get("original_image")    
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }
            
            storage_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }
            # --- STEP 3: SEND COMBINED DATA TO STORAGE SERVICE ---
            # Sending the merged JSON to your NestJS storage service
            storage_response = await client.post(
                STORAGE_SERVICE_URL,
                data=storage_data,
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



@router.post("/bone-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("bone fracture AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_BONEFRACTURE_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "modelName": modelName,
                "modelAccuracy": "98.2%", # Usually comes from AI response
                "prediction": json.dumps({
                    "type": ai_data.get("type", "2D"),
                    "original_image": ai_data.get("original_image"),
                    "visual_result": ai_data.get("visual_result")
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }

            # --- STEP 3: SEND COMBINED DATA TO STORAGE SERVICE ---
            # Sending the merged JSON to your NestJS storage service
            storage_response = await client.post(
                STORAGE_SERVICE_URL, 
                json=combined_payload
            )
            storage_response.raise_for_status()

            # Return the final DB-ready object to the frontend
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



@router.post("/colon-classification-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("colon Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient(timeout=LONG_TIMEOUT) as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }

            ai_response = await client.post(
                AI_COLONCELL_SERVICE_URL,
                files=ai_files
            )

            ai_response.raise_for_status()
            ai_data = ai_response.json()
            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            storage_data = {
                "patientId": patientId,
                "modelName": modelName,
                "filescantype": "colon",
                "prediction": json.dumps({
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "original_image":ai_data.get("original_image")    
                }),
                "originalName": file.filename,
                "mimetype": file.content_type,
                "size": len(file_bytes)
            }
            
            storage_files = {
                "file": (file.filename, file_bytes, file.content_type)
            }
            # --- STEP 3: SEND COMBINED DATA TO STORAGE SERVICE ---
            # Sending the merged JSON to your NestJS storage service
            storage_response = await client.post(
                STORAGE_SERVICE_URL,
                data=storage_data,
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