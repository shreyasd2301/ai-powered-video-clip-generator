import json
import os
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path
from services.logger_service import get_logger
from models.schemas import SceneIndexInfo

logger = get_logger("scene_index_service")

class SceneIndexService:
    def __init__(self):
        # Create data directory if it doesn't exist
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Scene index storage file
        self.scene_index_file = self.data_dir / "scene_indexes.json"
        self._load_scene_indexes()
    
    def _load_scene_indexes(self):
        """Load scene indexes from JSON file"""
        try:
            if self.scene_index_file.exists():
                with open(self.scene_index_file, 'r') as f:
                    data = json.load(f)
                    self.scene_indexes = {
                        video_id: SceneIndexInfo(**index_data) 
                        for video_id, index_data in data.items()
                    }
                logger.info(f"Loaded {len(self.scene_indexes)} scene indexes from {self.scene_index_file}")
            else:
                self.scene_indexes = {}
                logger.info("No existing scene index file found, starting with empty storage")
        except Exception as e:
            logger.error(f"Failed to load scene indexes: {str(e)}")
            self.scene_indexes = {}
    
    def _save_scene_indexes(self):
        """Save scene indexes to JSON file"""
        try:
            # Convert SceneIndexInfo objects to dictionaries
            data = {
                video_id: {
                    "video_id": info.video_id,
                    "scene_index_id": info.scene_index_id,
                    "index_type": info.index_type,
                    "created_at": info.created_at.isoformat(),
                    "custom_prompt": info.custom_prompt
                }
                for video_id, info in self.scene_indexes.items()
            }
            
            with open(self.scene_index_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved {len(self.scene_indexes)} scene indexes to {self.scene_index_file}")
        except Exception as e:
            logger.error(f"Failed to save scene indexes: {str(e)}")
            raise Exception(f"Failed to save scene indexes: {str(e)}")
    
    def save_scene_index(self, video_id: str, scene_index_id: str, index_type: str, custom_prompt: Optional[str] = None):
        """Save a scene index ID for a video"""
        try:
            scene_index_info = SceneIndexInfo(
                video_id=video_id,
                scene_index_id=scene_index_id,
                index_type=index_type,
                created_at=datetime.now(),
                custom_prompt=custom_prompt
            )
            
            self.scene_indexes[video_id] = scene_index_info
            self._save_scene_indexes()
            
            logger.info(f"Saved scene index {scene_index_id} for video {video_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save scene index for video {video_id}: {str(e)}")
            raise Exception(f"Failed to save scene index: {str(e)}")
    
    def get_scene_index_id(self, video_id: str) -> Optional[str]:
        """Get the scene index ID for a video"""
        try:
            if video_id in self.scene_indexes:
                scene_index_info = self.scene_indexes[video_id]
                logger.debug(f"Found scene index {scene_index_info.scene_index_id} for video {video_id}")
                return scene_index_info.scene_index_id
            else:
                logger.warning(f"No scene index found for video {video_id}")
                return None
        except Exception as e:
            logger.error(f"Failed to get scene index for video {video_id}: {str(e)}")
            return None
    
    def get_scene_index_info(self, video_id: str) -> Optional[SceneIndexInfo]:
        """Get the complete scene index info for a video"""
        try:
            if video_id in self.scene_indexes:
                return self.scene_indexes[video_id]
            else:
                return None
        except Exception as e:
            logger.error(f"Failed to get scene index info for video {video_id}: {str(e)}")
            return None
    
    def delete_scene_index(self, video_id: str) -> bool:
        """Delete a scene index for a video"""
        try:
            if video_id in self.scene_indexes:
                del self.scene_indexes[video_id]
                self._save_scene_indexes()
                logger.info(f"Deleted scene index for video {video_id}")
                return True
            else:
                logger.warning(f"No scene index found for video {video_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete scene index for video {video_id}: {str(e)}")
            return False
    
    def get_all_scene_indexes(self) -> Dict[str, SceneIndexInfo]:
        """Get all scene indexes"""
        return self.scene_indexes.copy()
    
    def video_has_scene_index(self, video_id: str) -> bool:
        """Check if a video has a scene index"""
        return video_id in self.scene_indexes