import os
import videodb
from videodb import MediaType, SceneExtractionType, SearchType, IndexType
from datetime import datetime
from typing import List, Dict, Optional
from videodb import SceneExtractionType

import asyncio
from dotenv import load_dotenv
from services.logger_service import get_logger
from services.scene_index_service import SceneIndexService

load_dotenv()
# Default scene extract time - can be overridden by user input
DEFAULT_SCENE_EXTRACT_TIME = int(os.getenv("SCENE_EXTRACT_TIME", "5"))

# Initialize logger
logger = get_logger("video_service")

class VideoService:
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            self.conn = videodb.connect(api_key=api_key)
        else:
            self.conn = videodb.connect()
        self.scene_index_service = SceneIndexService()
        
        # Create or get collections
        try:
            # Get or create spoken word collection (default)
            self.spoken_collection = self.conn.get_collection()
            logger.info("Spoken word collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize spoken word collection: {str(e)}")
            raise Exception(f"Failed to initialize spoken word collection: {str(e)}")
        
        try:
            # Get or create multimodal collection
            self.multimodal_collection = self._create_collection_if_not_exists("c-f5dcd278-5f63-4072-9818-eb820e1b1765")
            logger.info("Multimodal collection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize multimodal collection: {str(e)}")
            raise Exception(f"Failed to initialize multimodal collection: {str(e)}")
        
        self.videos_cache = {}
        logger.info("VideoService initialized with separate collections")
    
    def _get_collection(self, index_type: str):
        """Get the appropriate collection based on index type"""
        if index_type == "multimodal":
            return self.multimodal_collection
        else:
            return self.spoken_collection
    
    def _create_collection_if_not_exists(self, collection_id: str):
        """Create a collection if it doesn't exist"""
        try:
            # Try to get the collection first
            collection = self.conn.get_collection(collection_id=collection_id)
            logger.info(f"Collection '{collection_id}' already exists")
            return collection
        except Exception as e:
            logger.info(f"Collection '{collection_id}' not found, creating...")
            try:
                # Create the collection
                collection = self.conn.create_collection(collection_id, description="Multimodal videos collection")
                logger.info(f"Collection '{collection_id}' created successfully")
                return collection
            except Exception as create_error:
                logger.error(f"Failed to create collection '{collection_id}': {str(create_error)}")
                raise Exception(f"Failed to create collection '{collection_id}': {str(create_error)}")
    
    def _ensure_collection_exists(self, index_type: str):
        """Ensure the collection exists for the given index type"""
        try:
            collection = self._get_collection(index_type)
            # Try to access the collection to verify it exists
            if hasattr(collection, 'name'):
                logger.debug(f"Collection {collection.name} exists for {index_type}")
            return True
        except Exception as e:
            logger.error(f"Collection for {index_type} not available: {str(e)}")
            return False
    
    async def upload_video(self, url: str, index_type: str = "multimodal", custom_prompt: Optional[str] = None) -> Dict:
        """Upload a video to VideoDB using appropriate collection"""
        logger.info(f"Uploading video with URL={url}, index_type={index_type}")
        try:
            # Ensure collection exists
            if not self._ensure_collection_exists(index_type):
                raise Exception(f"Collection for {index_type} is not available")
            
            # Get the appropriate collection
            collection = self._get_collection(index_type)
            logger.debug(f"Using collection: {collection.name if hasattr(collection, 'name') else 'default'}")
            
            # Upload video to the appropriate collection
            video = collection.upload(url=url)
            logger.debug(f"Video uploaded with id={video.id}, name={video.name}")
            
            # Basic video info
            video_info = {
                "id": video.id,
                "name": video.name,
                "duration": getattr(video, 'duration', 0),
                "indexed": False,
                "index_type": index_type,
                "created_at": datetime.now(),
                "custom_prompt": custom_prompt,
                "collection": "multimodal" if index_type == "multimodal" else "spoken_word"
            }
            
            # Cache video info
            self.videos_cache[video.id] = video_info
            logger.info(f"Video {video.id} cached successfully in {index_type} collection")
            
            return video_info
        
        except Exception as e:
            logger.error(f"Failed to upload video: {str(e)}")
            raise Exception(f"Failed to upload video: {str(e)}")
    
    async def index_video_spoken(self, video_id: str) -> bool:
        """Index video for spoken words only (default index)"""
        logger.info(f"Indexing video {video_id} for spoken words")
        try:
            video = self.spoken_collection.get_video(video_id)
            video.index_spoken_words()
            logger.debug(f"Spoken words indexed for video {video_id}")
            
            # Update cache
            if video_id in self.videos_cache:
                self.videos_cache[video_id]["indexed"] = True
                self.videos_cache[video_id]["index_type"] = "spoken_word"
                logger.debug(f"Cache updated for video {video_id}")
            
            return True
        
        except Exception as e:
            logger.error(f"Failed to index video {video_id} for spoken words: {str(e)}")
            raise Exception(f"Failed to index video for spoken words: {str(e)}")
    
    async def index_video_multimodal(self, video_id: str, custom_prompt: Optional[str] = None, scene_extract_time: Optional[int] = None) -> bool:
        """Index video for both spoken words and scenes"""
        logger.info(f"Indexing video {video_id} for multimodal content")
        try:
            video = self.multimodal_collection.get_video(video_id)
            
            # Index spoken words
            video.index_spoken_words()
            
            # Use provided scene_extract_time or default
            extract_time = scene_extract_time if scene_extract_time is not None else DEFAULT_SCENE_EXTRACT_TIME
            logger.info(f"Using scene extract time: {extract_time} seconds")
            
            # Index scenes
            scene_prompt = custom_prompt or """
                1. **Character**: (e.g., "Speaker A", "Interviewee", "Host")
                2. **Emotion**: Identify the dominant emotion depicted (e.g., happiness, nervousness, surprise).
                3. **Facial Expression**: Describe face details (e.g., eyebrows raised, mouth slightly open, eyes narrowed).
                4. **Action/Posture**: Describe what the character is doing (e.g., leaning forward, gesturing with hands, turning head).
                5. **Background/Context**: Note surrounding elements (e.g., "plain white wall," "studio lights," "audience in blur," "office setting").
                6. **Overall Mood/Ambiance**: Summarize tone and atmosphere (e.g., "intense and focused," "warmly conversational," "tension between speakers").
                """
            scene_index_id = video.index_scenes(
                extraction_type=SceneExtractionType.time_based,
                extraction_config={
                    "time": extract_time,
                    "select_frames": ['first', 'last', 'middle']
                },
                prompt=scene_prompt
            )
            
            # Save scene index ID to JSON storage
            self.scene_index_service.save_scene_index(
                video_id=video_id,
                scene_index_id=scene_index_id,
                index_type="multimodal",
                custom_prompt=custom_prompt
            )
            
            # Update cache
            if video_id in self.videos_cache:
                self.videos_cache[video_id]["indexed"] = True
                self.videos_cache[video_id]["scene_index_id"] = scene_index_id
                self.videos_cache[video_id]["index_type"] = "multimodal"
                self.videos_cache[video_id]["custom_prompt"] = custom_prompt
                logger.debug(f"Cache updated for video {video_id}")
            
            logger.info(f"Multimodal indexing completed for video {video_id} with scene index {scene_index_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to index video {video_id} for multimodal: {str(e)}")
            raise Exception(f"Failed to index video for multimodal: {str(e)}")
    
    async def get_video(self, video_id: str) -> Dict:
        """Get video details with proper index type detection"""
        logger.debug(f"Getting video with id={video_id}")
        try:
            if video_id in self.videos_cache:
                logger.debug(f"Video {video_id} found in cache")
                return self.videos_cache[video_id]
            
            # Try to find video in both collections
            video_info = None
            
            # First try multimodal collection
            try:
                video = self.multimodal_collection.get_video(video_id)
                video_info = {
                    "id": video.id,
                    "name": video.name,
                    "duration": getattr(video, 'duration', 0),
                    "indexed": False,
                    "index_type": "multimodal",
                    "created_at": datetime.now(),
                    "custom_prompt": None,
                    "collection": "multimodal",
                    "scene_index_id": None
                }
                
                # Check if indexed
                try:
                    transcript = video.get_transcript()
                    video_info["indexed"] = True
                    logger.debug(f"Video {video_id} found in multimodal collection and is indexed")
                except:
                    video_info["indexed"] = False
                    logger.debug(f"Video {video_id} found in multimodal collection but not indexed")
                
                # Get scene index information
                scene_index_info = self.scene_index_service.get_scene_index_info(video_id)
                if scene_index_info:
                    video_info["scene_index_id"] = scene_index_info.scene_index_id
                    video_info["custom_prompt"] = scene_index_info.custom_prompt
                    logger.debug(f"Video {video_id} has scene index {scene_index_info.scene_index_id}")
                    
            except:
                # Try spoken word collection
                try:
                    video = self.spoken_collection.get_video(video_id)
                    video_info = {
                        "id": video.id,
                        "name": video.name,
                        "duration": getattr(video, 'duration', 0),
                        "indexed": False,
                        "index_type": "spoken_word",
                        "created_at": datetime.now(),
                        "custom_prompt": None,
                        "collection": "spoken_word",
                        "scene_index_id": None
                    }
                    
                    # Check if indexed
                    try:
                        transcript = video.get_transcript()
                        video_info["indexed"] = True
                        logger.debug(f"Video {video_id} found in spoken word collection and is indexed")
                    except:
                        video_info["indexed"] = False
                        logger.debug(f"Video {video_id} found in spoken word collection but not indexed")
                    
                    # Get scene index information (even for spoken word videos, in case they were indexed for scenes)
                    scene_index_info = self.scene_index_service.get_scene_index_info(video_id)
                    if scene_index_info:
                        video_info["scene_index_id"] = scene_index_info.scene_index_id
                        video_info["custom_prompt"] = scene_index_info.custom_prompt
                        logger.debug(f"Video {video_id} has scene index {scene_index_info.scene_index_id}")
                        
                except Exception as e:
                    logger.error(f"Video {video_id} not found in any collection: {str(e)}")
                    raise Exception(f"Video not found: {str(e)}")
            
            if video_info:
                self.videos_cache[video_id] = video_info
                logger.debug(f"Video {video_id} cached")
                return video_info
            else:
                raise Exception("Failed to retrieve video information")
        
        except Exception as e:
            logger.error(f"Failed to get video {video_id}: {str(e)}")
            raise Exception(f"Failed to get video: {str(e)}")
    
    def get_collection_status(self) -> Dict:
        """Get status of all collections"""
        status = {
            "spoken_word": {
                "name": "default",
                "available": False,
                "video_count": 0,
                "error": None
            },
            "multimodal": {
                "name": "multimodal-videos",
                "available": False,
                "video_count": 0,
                "error": None
            }
        }
        
        # Check spoken word collection
        try:
            spoken_videos = self.spoken_collection.get_videos()
            status["spoken_word"]["available"] = True
            status["spoken_word"]["video_count"] = len(spoken_videos)
            logger.debug(f"Spoken word collection: {len(spoken_videos)} videos")
        except Exception as e:
            status["spoken_word"]["error"] = str(e)
            logger.warning(f"Spoken word collection error: {str(e)}")
        
        # Check multimodal collection
        try:
            multimodal_videos = self.multimodal_collection.get_videos()
            status["multimodal"]["available"] = True
            status["multimodal"]["video_count"] = len(multimodal_videos)
            logger.debug(f"Multimodal collection: {len(multimodal_videos)} videos")
        except Exception as e:
            status["multimodal"]["error"] = str(e)
            logger.warning(f"Multimodal collection error: {str(e)}")
        
        return status
    
    async def get_all_videos(self) -> List[Dict]:
        """Get all videos from both collections with proper index type labeling"""
        logger.info("Getting all videos from both collections")
        try:
            video_list = []
            
            # Get videos from spoken word collection
            try:
                spoken_videos = self.spoken_collection.get_videos()
                for video in spoken_videos:
                    video_info = await self.get_video(video.id)
                    video_list.append(video_info)
                logger.debug(f"Retrieved {len(spoken_videos)} videos from spoken word collection")
            except Exception as e:
                logger.warning(f"Failed to get videos from spoken word collection: {str(e)}")
            
            # Get videos from multimodal collection
            try:
                multimodal_videos = self.multimodal_collection.get_videos()
                for video in multimodal_videos:
                    video_info = await self.get_video(video.id)
                    video_list.append(video_info)
                logger.debug(f"Retrieved {len(multimodal_videos)} videos from multimodal collection")
            except Exception as e:
                logger.warning(f"Failed to get videos from multimodal collection: {str(e)}")
            
            logger.info(f"Retrieved {len(video_list)} total videos from both collections")
            return video_list
        
        except Exception as e:
            logger.error(f"Failed to get all videos: {str(e)}")
            raise Exception(f"Failed to get all videos: {str(e)}")
    
    async def upload_image(self, url: str, overlay_settings: Optional[Dict] = None) -> Dict:
        """Upload an image for overlay - simplified version following notebook pattern"""
        try:
            # Simple image upload without complex metadata
            image = self.conn.upload(url=url, media_type=MediaType.image)
            
            image_info = {
                "id": image.id,
                "url": url,
                "uploaded_at": datetime.now()
            }
            
            logger.info(f"Image uploaded successfully with id={image.id}")
            return image_info
        
        except Exception as e:
            logger.error(f"Failed to upload image: {str(e)}")
            raise Exception(f"Failed to upload image: {str(e)}")
    
    async def delete_video(self, video_id: str) -> bool:
        """Delete a video from appropriate collection"""
        try:
            # Remove from cache
            if video_id in self.videos_cache:
                del self.videos_cache[video_id]
            
            # Delete scene index data
            self.scene_index_service.delete_scene_index(video_id)
            
            # Try to delete from both collections (video will only exist in one)
            try:
                self.multimodal_collection.delete_video(video_id)
                logger.info(f"Video {video_id} deleted from multimodal collection")
            except:
                try:
                    self.spoken_collection.delete_video(video_id)
                    logger.info(f"Video {video_id} deleted from spoken word collection")
                except:
                    logger.warning(f"Video {video_id} not found for deletion")
            
            return True
        
        except Exception as e:
            raise Exception(f"Failed to delete video: {str(e)}")
    
    def get_video_object(self, video_id: str):
        """Get the actual VideoDB video object from appropriate collection"""
        try:
            # Try multimodal collection first
            try:
                return self.multimodal_collection.get_video(video_id)
            except:
                return self.spoken_collection.get_video(video_id)
        except Exception as e:
            raise Exception(f"Failed to get video object: {str(e)}")
    
    def get_scenes(self, video_id: str, scene_index_id: Optional[str] = None):
        """Get video scenes from appropriate collection"""
        try:
            video = self.get_video_object(video_id)
            
            # If no scene_index_id provided, try to get it from storage
            if not scene_index_id:
                scene_index_id = self.scene_index_service.get_scene_index_id(video_id)
                if scene_index_id:
                    logger.debug(f"Retrieved scene index {scene_index_id} for video {video_id} from storage")
                else:
                    logger.warning(f"No scene index found for video {video_id} in storage")
                    return []
            
            # Get scenes using the scene index ID
            scenes = video.get_scene_index(scene_index_id)
            logger.debug(f"Retrieved {len(scenes) if scenes else 0} scenes for video {video_id}")
            return scenes, scene_index_id
            
        except Exception as e:
            logger.error(f"Failed to get scenes for video {video_id}: {str(e)}")
            raise Exception(f"Failed to get scenes: {str(e)}") 

    def search_spoken_content(self, video_id: str, query: str, search_type: str = "semantic"):
        """Search spoken content in video."""
        try:
            logger.info( f"query: {query.encode('ascii', 'ignore').decode()}")
            video = self.get_video_object(video_id)
            results = video.search(
                query=query,
                # index_type=IndexType.spoken_word,
                # search_type=SearchType.semantic,
                limit=1
            )
            # logger.info(f"Spoken results: {results}")
            for shot in results.get_shots():
                safe_text = shot.text.encode('ascii', 'ignore').decode()
                logger.info(f"Result: {shot.start} - {shot.end} - {safe_text}")
            return results
        except Exception as e:
            logger.error(f"Error searching spoken content: {e}")
            return []
    
    def search_visual_content(self, video_id: str, query: str, search_type: str = "semantic"):
        """Search visual content in video."""
        try:
            video = self.get_video_object(video_id)
            results = video.search(
                query=query,    
                # index_type=IndexType.scene,
                # search_type=SearchType.semantic
                limit=1
            )
            # logger.info(f"Visual results: {results}")
            return results
        except Exception as e:
            logger.error(f"Error searching visual content: {e}")
            return []

    def get_transcript(self, video_id: str):
        """Get transcript from video."""
        try:
            video = self.get_video_object(video_id)
            transcript = video.get_transcript_text(force=True)
            # logger.info(f"Transcript: {transcript}")
            return transcript
        except Exception as e:
            logger.error(f"Error getting transcript: {e}")
            raise