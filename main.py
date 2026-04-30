from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from routers import router as manual
from automaticdispatcher import router as auto
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)  # ← this line was missing
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )

app.include_router(auto, prefix="/api/v1")
app.include_router(manual, prefix="/api/v2")