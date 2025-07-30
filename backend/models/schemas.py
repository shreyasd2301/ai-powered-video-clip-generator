from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class VideoUploadRequest(BaseModel):
    video_url: Optional[str] = None
    index_type: str = "multimodal"  # "spoken_word" or "multimodal"
    custom_prompt: Optional[str] = None
    scene_extract_time: Optional[int] = 5  # Scene extraction time in seconds
    api_key: Optional[str] = None

class ClipGenerationRequest(BaseModel):
    video_id: str
    user_query: str
    index_type: str = "multimodal"
    include_ranking: bool = True
    max_duration: Optional[int] = 180
    top_n: Optional[int] = 10
    api_key: Optional[str] = None
    # Image overlay metadata
    image_id: Optional[str] = None
    image_width: Optional[int] = 40
    image_height: Optional[int] = 40
    image_x: Optional[int] = 20
    image_y: Optional[int] = 10
    image_duration: Optional[int] = 7
    # Audio overlay metadata
    audio_id: Optional[str] = None
    audio_start: Optional[int] = 3
    audio_end: Optional[int] = 4
    audio_disable_other_tracks: Optional[bool] = True

class ImageUploadRequest(BaseModel):
    image_url: Optional[str] = None
    api_key: Optional[str] = None

class VideoInfo(BaseModel):
    id: str
    name: str
    duration: float
    indexed: bool
    index_type: str  # "spoken_word" or "multimodal"
    created_at: datetime
    custom_prompt: Optional[str] = None
    collection: Optional[str] = None  # "spoken_word" or "multimodal"
    scene_index_id: Optional[str] = None  # Store the scene index ID

class ClipInfo(BaseModel):
    id: str
    video_id: str
    query: str
    stream_url: str
    duration: float
    created_at: datetime

class SceneIndexInfo(BaseModel):
    """Schema for storing scene index information"""
    video_id: str
    scene_index_id: str
    index_type: str
    created_at: datetime
    custom_prompt: Optional[str] = None 