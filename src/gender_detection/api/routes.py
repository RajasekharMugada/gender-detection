from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ..services.live_audio_processor import LiveAudioProcessor
import asyncio

router = APIRouter()

@router.get("/detect-gender")
async def detect_gender():
    try:
        processor = LiveAudioProcessor()
        return StreamingResponse(
            processor.stream_results(),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream'
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy"} 