import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException , APIRouter
from typing import Optional

router = APIRouter()


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
AI_LUNGCELL_SERVICE_URL = "http://127.0.0.1:8000/api/v1/lungclass"
AI_BRAIN_SERVICE_URL = "http://127.0.0.1:8000/api/v1/brainpredict"
AI_BLOODANALYSIS_SERVICE_URL = "http://127.0.0.1:8000/api/v1/bloodanalysis"

STORAGE_SERVICE_URL = "http://127.0.0.1:3001/upload/single"


@router.post("breast-predict-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Breast Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_BREAST_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "breast",
                "modelAccuracy": "98.2%", # Usually comes from AI response
                "prediction": {
                    "type": ai_data.get("type", "2D"),
                    "original_image": ai_data.get("original_image"),
                    "visual_result": ai_data.get("visual_result")
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")



@router.post("skin-classification-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("skin Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_SKIN_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "skin",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "all_probabilities": ai_data.get("all_probabilities"),
                    "original_image":ai_data.get("original_image")    
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")



@router.post("blood-classify-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Blood Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_BLOOD_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "blood",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "all_probabilities": ai_data.get("all_probabilities"),
                    "original_image":ai_data.get("original_image")
                    
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")


#change
@router.post("/liver-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Liver Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_LIVER_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "liver",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "all_scores": ai_data.get("all_scores"),
                    "mesh": ai_data.get("mesh"),
                    "patient_analysis": ai_data.get("patient_analysis"),
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

@router.post("/brain-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("brain Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_BRAIN_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "brain",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "all_scores": ai_data.get("all_scores"),
                    "mesh": ai_data.get("mesh"),
                    "patient_analysis": ai_data.get("patient_analysis"),
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")



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
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_LUNG_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "lung",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "all_scores": ai_data.get("all_scores"),
                    "mesh": ai_data.get("mesh"),
                    "patient_analysis": ai_data.get("patient_analysis"),
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")


#change
@router.post("lung-classify-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Breast Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_LUNGCELL_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "skin",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "all_probabilities": ai_data.get("all_probabilities"),
                    "original_image":ai_data.get("original_image")
                    
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")


#change
@router.post("bone-classification-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Breast Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_BONECANCER_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "skin",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "all_probabilities": ai_data.get("all_probabilities"),
                    "original_image":ai_data.get("original_image")
                    
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")



@router.post("bone-segment-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Breast Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
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
                "fileType": fileType,
                "modelName": modelName,
                "modelAccuracy": "98.2%", # Usually comes from AI response
                "prediction": {
                    "type": ai_data.get("type", "2D"),
                    "original_image": ai_data.get("original_image"),
                    "visual_result": ai_data.get("visual_result")
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")



@router.post("colon-classification-store")
async def handle_analysis(
    file: UploadFile = File(...),
    patientId: str = Form(...),
    fileType: str = Form(...),
    modelName: str = Form("Breast Cancer AI")
):
    # 1. Read the file into memory so we can send it to multiple places
    file_bytes = await file.read()
    
    async with httpx.AsyncClient() as client:
        try:
            # --- STEP 1: SEND TO AI SERVICE ---
            # We recreate the form-data for the AI API
            ai_files = {'file': (file.filename, file_bytes, file.content_type)}
            ai_response = await client.post(AI_COLONCELL_SERVICE_URL, files=ai_files)
            ai_response.raise_for_status()
            ai_data = ai_response.json() # This has visual_result and original_image

            # --- STEP 2: COMBINE DATA ---
            # Merging your incoming metadata with the AI's response
            combined_payload = {
                "patientId": patientId,
                "fileType": fileType,
                "modelName": modelName,
                "filescantype": "skin",
                "prediction": {
                    "prediction": ai_data.get("prediction"),
                    "class_index":ai_data.get("class_index"),
                    "confidence": ai_data.get("confidence"),
                    "type": ai_data.get("type"),
                    "diagnostics": ai_data.get("diagnostics"),
                    "all_probabilities": ai_data.get("all_probabilities"),
                    "original_image":ai_data.get("original_image")
                    
                },
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
            raise HTTPException(status_code=e.response.status_code, detail=f"Service Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")