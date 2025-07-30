from fastapi import APIRouter, HTTPException
from typing import List, Optional

from models.schemas import ClipGenerationRequest, ClipInfo
from services.clip_service import ClipService
from services.logger_service import get_logger

# Initialize logger
logger = get_logger("clips_api")

router = APIRouter(prefix="/clips", tags=["clips"])

# Initialize clip service (will be recreated with API key when needed)
clip_service = None
logger.info("ClipService will be initialized with API key when needed")

@router.post("/create", response_model=ClipInfo)
async def create_clip(request: ClipGenerationRequest):
    """Generate a clip based on user query"""
    logger.info(f"Creating clip for video_id={request.video_id}, query='{request.user_query}', index_type={request.index_type}")
    try:
        if not request.api_key:
            logger.warning("Clip creation attempted without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize clip service with API key
        clip_service = ClipService(api_key=request.api_key)
        clip_info = await clip_service.generate_clip(
            video_id=request.video_id,
            user_query=request.user_query,
            index_type=request.index_type,
            include_ranking=request.include_ranking,
            max_duration=request.max_duration,
            top_n=request.top_n
        )
        logger.info(f"Clip created successfully for video_id={request.video_id}")
        return ClipInfo(**clip_info)
    
    except Exception as e:
        logger.error(f"Failed to create clip for video_id={request.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[ClipInfo])
async def get_clips(api_key: str):
    """Get all generated clips"""
    logger.info("Retrieving all clips")
    try:
        if not api_key:
            logger.warning("Clips requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize clip service with API key
        clip_service = ClipService(api_key=api_key)
        clips = await clip_service.get_all_clips()
        logger.info(f"Retrieved {len(clips)} clips successfully")
        return [ClipInfo(**clip) for clip in clips]
    except Exception as e:
        logger.error(f"Failed to retrieve clips: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics")
async def get_clips_analytics(api_key: str):
    """Get detailed analytics for clips"""
    logger.info("Retrieving clips analytics")
    try:
        if not api_key:
            logger.warning("Analytics requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize clip service with API key
        clip_service = ClipService(api_key=api_key)
        clips = await clip_service.get_all_clips()
        
        # Calculate analytics
        total_clips = len(clips)
        total_duration = sum(clip.get('duration', 0) for clip in clips)
        avg_duration = total_duration / total_clips if total_clips > 0 else 0
        
        # Most common queries
        queries = [clip.get('query', '') for clip in clips]
        query_frequency = {}
        for query in queries:
            if query:
                query_frequency[query] = query_frequency.get(query, 0) + 1
        
        # Clips with overlays
        clips_with_overlays = sum(1 for clip in clips if clip.get('has_overlay', False))
        
        # Index type distribution
        index_types = {}
        for clip in clips:
            index_type = clip.get('index_type', 'unknown')
            index_types[index_type] = index_types.get(index_type, 0) + 1
        
        analytics = {
            "total_clips": total_clips,
            "total_duration": total_duration,
            "average_duration": avg_duration,
            "clips_with_overlays": clips_with_overlays,
            "index_type_distribution": index_types,
            "most_common_queries": sorted(query_frequency.items(), key=lambda x: x[1], reverse=True)[:10],
            "recent_clips": sorted(clips, key=lambda x: x.get('created_at', ''), reverse=True)[:5]
        }
        
        logger.info(f"Retrieved analytics for {total_clips} clips")
        return analytics
        
    except Exception as e:
        logger.error(f"Failed to retrieve clips analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{clip_id}", response_model=ClipInfo)
async def get_clip(clip_id: str, api_key: str):
    """Get specific clip details"""
    logger.info(f"Retrieving clip with id={clip_id}")
    try:
        if not api_key:
            logger.warning("Clip requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize clip service with API key
        clip_service = ClipService(api_key=api_key)
        clip = await clip_service.get_clip(clip_id)
        logger.info(f"Clip {clip_id} retrieved successfully")
        return ClipInfo(**clip)
    except Exception as e:
        logger.error(f"Failed to retrieve clip {clip_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/create-with-overlay", response_model=ClipInfo)
async def create_clip_with_overlay(request: ClipGenerationRequest):
    """Generate a clip with image/audio overlay using user-provided metadata"""
    logger.info(f"Creating clip with overlay for video_id={request.video_id}, image_id={request.image_id}, audio_id={request.audio_id}")
    try:
        if not request.api_key:
            logger.warning("Clip with overlay creation attempted without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize clip service with API key
        clip_service = ClipService(api_key=request.api_key)
        clip_info = await clip_service.generate_clip_with_overlay(
            video_id=request.video_id,
            user_query=request.user_query,
            index_type=request.index_type,
            image_id=request.image_id,
            audio_id=request.audio_id,
            include_ranking=request.include_ranking,
            max_duration=request.max_duration,
            top_n=request.top_n,
            # Image overlay metadata
            image_width=request.image_width,
            image_height=request.image_height,
            image_x=request.image_x,
            image_y=request.image_y,
            image_duration=request.image_duration,
            # Audio overlay metadata
            audio_start=request.audio_start,
            audio_end=request.audio_end,
            audio_disable_other_tracks=request.audio_disable_other_tracks
        )
        logger.info(f"Clip with overlay created successfully for video_id={request.video_id}")
        return ClipInfo(**clip_info)
    
    except Exception as e:
        logger.error(f"Failed to create clip with overlay for video_id={request.video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 