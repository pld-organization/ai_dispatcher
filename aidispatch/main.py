import uvicorn
from fastapi import FastAPI
from router import router as manual

from automaticdispatcher import router as auto

app = FastAPI(
    title="Medical Image Classification API",
    description="Routes medical images through a classifier then processes them with specialist models.",
    version="1.0.0"
)

app.include_router(auto, prefix="/api/v1")
app.include_router(manual, prefix="/api/v2")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8100, reload=True)
