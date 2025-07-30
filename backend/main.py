from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from api.videos import router as videos_router
from api.clips import router as clips_router
from services.logger_service import get_logger

# Initialize logger
logger = get_logger("main")

app = FastAPI(title="VideoDB Clip Generator API", version="1.0.0")

# CORS middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Include routers
app.include_router(videos_router)
app.include_router(clips_router)

logger.info("FastAPI application initialized with CORS middleware and routers")

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "VideoDB Clip Generator API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VideoDB Clip Generator API server on host=0.0.0.0, port=8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 