from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List
from datetime import datetime

from models.schemas import VideoUploadRequest, VideoInfo, ImageUploadRequest
from services.video_service import VideoService
from services.logger_service import get_logger

# Initialize logger
logger = get_logger("videos_api")

router = APIRouter(prefix="/videos", tags=["videos"])

# Initialize video service (will be recreated with API key when needed)
video_service = None
logger.info("VideoService will be initialized with API key when needed")

@router.post("/upload", response_model=VideoInfo)
async def upload_video(
    request: VideoUploadRequest,
    background_tasks: BackgroundTasks
):
    """Upload a video and optionally index it"""
    logger.info(f"Uploading video with URL={request.video_url}, index_type={request.index_type}")
    try:
        if not request.video_url:
            logger.warning("Video upload attempted without URL")
            raise HTTPException(status_code=400, detail="Video URL is required")
        
        if not request.api_key:
            logger.warning("Video upload attempted without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize video service with API key
        video_service = VideoService(api_key=request.api_key)
        
        # Upload video
        video_info = await video_service.upload_video(
            url=request.video_url,
            index_type=request.index_type,
            custom_prompt=request.custom_prompt
        )
        
        logger.info(f"Video uploaded successfully with id={video_info['id']}")
        
        # Index video in background if needed
        if request.index_type == "multimodal":
            background_tasks.add_task(
                video_service.index_video_multimodal,
                video_info["id"],
                request.custom_prompt,
                request.scene_extract_time
            )
            logger.info(f"Multimodal indexing scheduled for video_id={video_info['id']} with scene_extract_time={request.scene_extract_time}")
        elif request.index_type == "scene":
            background_tasks.add_task(
                video_service.index_video_scene_only,
                video_info["id"],
                request.custom_prompt,
                request.scene_extract_time
            )
            logger.info(f"Scene indexing scheduled for video_id={video_info['id']} with scene_extract_time={request.scene_extract_time}")
        else:
            background_tasks.add_task(
                video_service.index_video_spoken,
                video_info["id"]
            )
            logger.info(f"Spoken word indexing scheduled for video_id={video_info['id']}")
        
        return VideoInfo(**video_info)
    
    except Exception as e:
        logger.error(f"Failed to upload video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collections/status")
async def get_collection_status(api_key: str):
    """Get status of all collections"""
    logger.info("Retrieving collection status")
    try:
        if not api_key:
            logger.warning("Collection status requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize video service with API key
        video_service = VideoService(api_key=api_key)
        status = video_service.get_collection_status()
        logger.info("Collection status retrieved successfully")
        return status
    except Exception as e:
        logger.error(f"Failed to retrieve collection status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[VideoInfo])
async def get_videos(api_key: str):
    """Get all uploaded videos with proper index type labeling"""
    logger.info("Retrieving all videos")
    try:
        if not api_key:
            logger.warning("Videos requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize video service with API key
        video_service = VideoService(api_key=api_key)
        videos = await video_service.get_all_videos()
        logger.info(f"Retrieved {len(videos)} videos successfully")
        return [VideoInfo(**video) for video in videos]
    except Exception as e:
        logger.error(f"Failed to retrieve videos: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{video_id}", response_model=VideoInfo)
async def get_video(video_id: str, api_key: str):
    """Get specific video details"""
    logger.info(f"Retrieving video with id={video_id}")
    try:
        if not api_key:
            logger.warning("Video requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize video service with API key
        video_service = VideoService(api_key=api_key)
        video = await video_service.get_video(video_id)
        logger.info(f"Video {video_id} retrieved successfully")
        return VideoInfo(**video)
    except Exception as e:
        logger.error(f"Failed to retrieve video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{video_id}")
async def delete_video(video_id: str, api_key: str):
    """Delete a video"""
    logger.info(f"Deleting video with id={video_id}")
    try:
        if not api_key:
            logger.warning("Video deletion requested without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize video service with API key
        video_service = VideoService(api_key=api_key)
        success = await video_service.delete_video(video_id)
        if success:
            logger.info(f"Video {video_id} deleted successfully")
            return {"message": "Video deleted successfully"}
        else:
            logger.warning(f"Video {video_id} not found for deletion")
            raise HTTPException(status_code=404, detail="Video not found")
    except Exception as e:
        logger.error(f"Failed to delete video {video_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-image")
async def upload_image(request: ImageUploadRequest):
    """Upload an image for overlay - simplified version"""
    logger.info(f"Uploading image with URL={request.image_url}")
    try:
        if not request.image_url:
            logger.warning("Image upload attempted without URL")
            raise HTTPException(status_code=400, detail="Image URL is required")
        
        if not request.api_key:
            logger.warning("Image upload attempted without API key")
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Initialize video service with API key
        video_service = VideoService(api_key=request.api_key)
        
        # Upload image with simplified approach
        image_info = await video_service.upload_image(url=request.image_url)
        logger.info(f"Image uploaded successfully with id={image_info['id']}")
        return image_info
    
    except Exception as e:
        logger.error(f"Failed to upload image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 