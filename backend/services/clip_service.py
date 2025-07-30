import keyword
import os
import uuid
import json
from datetime import datetime
from typing import Dict, Optional, List
import asyncio
from winreg import REG_RESOURCE_REQUIREMENTS_LIST
from dotenv import load_dotenv

# Import existing video_prompter functionality
import sys
sys.path.append('../..')

from llm_agent import LLM
from videodb import play_stream
from videodb.timeline import Timeline, VideoAsset, ImageAsset, AudioAsset
from services.logger_service import get_logger
from services.video_service import VideoService

load_dotenv()

# Initialize logger
logger = get_logger("clip_service")

# Storage file path
CLIPS_STORAGE_FILE = "data/clips_analytics.json"

def ensure_data_directory():
    """Ensure the data directory exists"""
    os.makedirs("data", exist_ok=True)

def load_clips_data() -> List[Dict]:
    """Load clips data from JSON file"""
    ensure_data_directory()
    try:
        if os.path.exists(CLIPS_STORAGE_FILE):
            with open(CLIPS_STORAGE_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Failed to load clips data: {str(e)}")
        return []

def save_clips_data(clips_data: List[Dict]):
    """Save clips data to JSON file"""
    ensure_data_directory()
    try:
        with open(CLIPS_STORAGE_FILE, 'w') as f:
            json.dump(clips_data, f, indent=2, default=str)
        logger.info(f"Saved {len(clips_data)} clips to storage")
    except Exception as e:
        logger.error(f"Failed to save clips data: {str(e)}")


def generate_teaser_script(user_query, transcript):
    """Generate a teaser script for a video query"""
    logger.info(f"Generating teaser script for query: {user_query}")
    try:
        llm = LLM()
        prompt = f"""
        You are a creative scriptwriter and AI assistant. Your task is to extract the most emotionally intense or thought-provoking moment from a transcript and write a short teaser script. 
        This script will be used to locate the corresponding section in the video and generate short clips for social media or promotional use.
        
        Requirements:

        100 words max
        Highlights the most emotionally intense, surprising, or thought-provoking part of the video
        Uses direct or slightly paraphrased quotes that can help locate the original timestamps
        Avoids generic summaries. Focus on captivating lines that would make someone want to click to watch more.

        Transcript:
        ```{transcript}```
        User Query: {user_query}
        """
        prompt += """
        JSON Format the response strictly as:
        <script>
        {{
            "script": ["line1", ...]
            "keywords": ["keyword1", ...]
        }}
        """

        response = llm.chat(message=prompt)
        output = response["choices"][0]["message"]["content"]
        import json
        res = json.loads(output)
        script = res.get('script', [])
        keywords = res.get('keywords', [])
        logger.info(f"Generated teaser script: {script}, length: {len(script)}")
        logger.info(f"Generated keywords: {keywords}, length: {len(keywords)}")
        return script, keywords
    except Exception as e:
        logger.info(f"Response: {response}")
        logger.error(f"Failed to generate teaser script: {str(e)}")
        return ""

class ClipService:
    def __init__(self, api_key: Optional[str] = None):
        self.clips_cache = {}
        self.llm = LLM()
        self.api_key = api_key
        # Load existing clips from storage
        self.clips_cache = {clip['id']: clip for clip in load_clips_data()}
        logger.info(f"ClipService initialized with {len(self.clips_cache)} existing clips")
    
    async def _generate_timeline_segments(
        self,
        video_id: str,
        user_query: str,
        index_type: str = "multimodal",
        include_ranking: bool = True,
        max_duration: int = 180,
        top_n: Optional[int] = 10
    ) -> tuple:
        """Generate timeline segments for a video query"""
        video_service = VideoService(api_key=self.api_key)
        video = video_service.get_video_object(video_id)
        # Get transcript from video

        transcript = video_service.get_transcript(video_id)
        # logger.info(f"Transcript: {transcript}")

        # Generate a script for a teaser clip
        spoken_query, keywords = generate_teaser_script(user_query, transcript)

        spoken_timestamps = []
        for i, query in enumerate(spoken_query):
            spoken_result = video_service.search_spoken_content(video_id, query)
            spoken_timestamps.extend([(shot.start, shot.end, shot.text) for shot in spoken_result.get_shots()])
        logger.info(f"Spoken timestamps: {spoken_timestamps}, length: {len(spoken_timestamps)}")

        visual_timestamps = []
        if index_type == "multimodal":  
            _, visual_query = self._divide_query(user_query)
            logger.info(f"Visual query: {visual_query}")
            for query in visual_query:
                visual_result = video_service.search_visual_content(video_id, query)
                visual_timestamps.extend([(shot.start, shot.end) for shot in visual_result.get_shots()])
        
        logger.info(f"Visual timestamps: {visual_timestamps}, length: {len(visual_timestamps)}")

        # Merge results
        result = self._merge_transcript_intervals(spoken_timestamps)
        logger.info(f"Combined results: {result}, count: {len(result)}")
        
        # Apply ranking if requested
        if include_ranking and len(result) > 1:
            logger.info("Applying ranking to results")
            result = await self._rank_results(result, spoken_query, duration=max_duration)
        
        # Limit results if specified
        if top_n and len(result) > top_n:
            logger.info(f"Limiting results to top {top_n}")
            result = result[:top_n]
        
        # Check if any segments were found
        if len(result) == 0:
            logger.warning(f"No video segments found for query: '{user_query}' with index_type: {index_type}")
            raise Exception(f"No video segments found matching the query: '{user_query}'. Try a different query or check if the video contains the requested content.")
        
        return result, video_service

    async def generate_clip(
        self, 
        video_id: str, 
        user_query: str, 
        index_type: str = "multimodal",
        include_ranking: bool = True,
        max_duration: int = 240,
        top_n: Optional[int] = 10
    ) -> Dict:
        """Generate a clip based on user query"""
        logger.info(f"Generating clip for video_id={video_id}, query='{user_query}', index_type={index_type}, include_ranking={include_ranking}, max_duration={max_duration}, top_n={top_n}")
        try:
            
            # Generate timeline segments
            result, video_service = await self._generate_timeline_segments(
                video_id, user_query, index_type, include_ranking, max_duration, top_n
            )
            
            timeline, duration = self._build_video_timeline(
                video_service, result, video_id,
                max_duration=max_duration
            )
            logger.info(f"Timeline built with duration: {duration}")
            
            # Generate stream
            stream = timeline.generate_stream()
            stream_url = play_stream(stream)
            logger.info(f"Stream URL generated: {stream_url}")
            
            # Create clip info
            clip_id = str(uuid.uuid4())
            clip_info = {
                "id": clip_id,
                "video_id": video_id,
                "query": user_query,
                "stream_url": stream_url,
                "duration": duration,
                "created_at": datetime.now(),
                "index_type": index_type,
                "segments_count": len(result)
            }
            
            # Cache clip info
            self.clips_cache[clip_id] = clip_info
            
            # Save to persistent storage
            clips_data = list(self.clips_cache.values())
            save_clips_data(clips_data)
            
            logger.info(f"Clip generated successfully with id={clip_id}, duration={duration}")
            
            return clip_info
        
        except Exception as e:
            logger.error(f"Failed to generate clip for video_id={video_id}: {str(e)}")
            raise Exception(f"Failed to generate clip: {str(e)}")
    
    async def generate_clip_with_overlay(
        self, 
        video_id: str, 
        user_query: str, 
        index_type: str = "multimodal",
        image_id: Optional[str] = None,
        audio_id: Optional[str] = None,
        include_ranking: bool = True,
        max_duration: int = 240,
        top_n: Optional[int] = 10,
        # Image overlay metadata
        image_width: Optional[int] = 40,
        image_height: Optional[int] = 40,
        image_x: Optional[int] = 20,
        image_y: Optional[int] = 10,
        image_duration: Optional[int] = 7,
        # Audio overlay metadata
        audio_start: Optional[int] = 3,
        audio_end: Optional[int] = 4,
        audio_disable_other_tracks: Optional[bool] = True
    ) -> Dict:
        """Generate a clip with image/audio overlay - simplified version"""
        logger.info(f"Generating clip with overlay for video_id={video_id}, image_id={image_id}, audio_id={audio_id}")
        try:
            # Generate timeline segments once
            result, video_service = await self._generate_timeline_segments(
                video_id, user_query, index_type, include_ranking, max_duration, top_n
            )
            
            # Build timeline with video segments
            timeline, duration = self._build_video_timeline(
                video_service, result, video_id, max_duration=max_duration
            )
            logger.info(f"Timeline built with duration: {duration}")
            
            # Add image overlay if provided - using user-provided metadata
            if image_id:
                logger.info(f"Adding image overlay with id={image_id}, width={image_width}, height={image_height}, x={image_x}, y={image_y}, duration={image_duration}")
                from videodb.asset import ImageAsset
                
                # Create image asset with user-provided settings
                image_asset = ImageAsset(
                    asset_id=image_id,
                    width=image_width, height=image_height,
                    x=image_x, y=image_y,
                    duration=image_duration
                )
                timeline.add_overlay(0, image_asset)
                logger.info(f"Image overlay added successfully with user-provided settings")
            
            # Add audio overlay if provided - using user-provided metadata
            if audio_id:
                logger.info(f"Adding audio overlay with id={audio_id}, start={audio_start}, end={audio_end}, disable_other_tracks={audio_disable_other_tracks}")
                from videodb.asset import AudioAsset
                
                audio_asset = AudioAsset(
                    asset_id=audio_id,
                    start=audio_start,
                    end=audio_end,
                    disable_other_tracks=audio_disable_other_tracks
                )
                timeline.add_overlay(0, audio_asset)
                logger.info(f"Audio overlay added successfully with user-provided settings")
            
            # Generate stream with overlays
            stream = timeline.generate_stream()
            stream_url = play_stream(stream)
            logger.info(f"Stream URL generated with overlays: {stream_url}")
            
            # Create clip info
            clip_id = str(uuid.uuid4())
            clip_info = {
                "id": clip_id,
                "video_id": video_id,
                "query": user_query,
                "stream_url": stream_url,
                "duration": duration,
                "created_at": datetime.now(),
                "index_type": index_type,
                "segments_count": len(result),
                "has_overlay": bool(image_id or audio_id),
                "image_overlay": image_id,
                "audio_overlay": audio_id
            }
            
            # Cache clip info
            self.clips_cache[clip_id] = clip_info
            
            # Save to persistent storage
            clips_data = list(self.clips_cache.values())
            save_clips_data(clips_data)
            
            logger.info(f"Clip with overlay generated successfully with id={clip_id}, duration={duration}")
            
            return clip_info
        
        except Exception as e:
            logger.error(f"Failed to generate clip with overlay for video_id={video_id}: {str(e)}")
            raise Exception(f"Failed to generate clip with overlay: {str(e)}")

    def _divide_query(self, query):
        """Helper method to divide query into spoken and visual parts"""
        import openai
        from openai import OpenAI
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai.api_key)
        
        prompt = """
        Divide the following query into two distinct parts: one for spoken content and one for visual content.

        The spoken content should refer to any narration, dialogue, or verbal explanations.
        The visual content should refer to any images, videos, or graphical representations.

        Format the response strictly as:
        Spoken: <spoken_query>
        Visual: <visual_query>

        Query: {query}
    """
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt.format(query=query)}
            ],
            temperature=0.1
        )
        message = response.choices[0].message.content
        divided_query = message.strip().split("\n")
        spoken_query = ""
        visual_query = ""
        for i in divided_query:
            if "Spoken:" in i:
                spoken_query = i.replace("Spoken:", "").strip()
            elif "Visual:" in i:
                visual_query = i.replace("Visual:", "").strip()
        
        return spoken_query, visual_query

    def _merge_transcript_intervals(self, intervals):
        """Helper method to merge overlapping intervals"""
        if not intervals:
            return []

        # Sort intervals by start time
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]

        for current in intervals[1:]:
            last = merged[-1]
            # If intervals overlap or are the same
            if current[0] <= last[1]:
                # Merge the intervals
                new_start = last[0]
                new_end = max(last[1], current[1])
                
                # Combine texts if they differ
                if len(current) > 2 and len(last) > 2 and current[2] != last[2]:
                    new_text = last[2].strip() + " " + current[2].strip()
                else:
                    new_text = last[2] if len(last) > 2 else ""
                merged[-1] = (new_start, new_end, new_text)
            else:
                merged.append(current)

        return merged

    def _build_video_timeline(self, video_service, result_timestamps, video_id, top_n=None, max_duration=None):
        """Helper method to build video timeline"""
        try:

            # Generate timeline and stream
            timeline = Timeline(video_service.conn)
            logger.info(f"Generated {len(result_timestamps)} timestamps")

            duration = 0
            if top_n:
                existing_count = len(result_timestamps)
                result_timestamps = result_timestamps[:top_n]
                logger.info(f"Picked top {top_n} from {existing_count}")

            for result_timestamp in result_timestamps:
                start = float(result_timestamp[0])
                end = float(result_timestamp[1])
                description = result_timestamp[2] if len(result_timestamp) > 2 else ""
                logger.info(f"Adding {start} - {end} - {description}")
                duration += end - start
                if max_duration and duration > max_duration:
                    logger.info(f"Duration exceeded max_duration")
                    duration -= end - start
                    break
                timeline.add_inline(VideoAsset(asset_id=video_id, start=start, end=end))
            return timeline, duration
        except Exception as e:
            logger.error(f"Failed to build video timeline: {str(e)}")
            raise Exception(f"Failed to build video timeline: {str(e)}")
    
    async def _rank_results(self, results: List[tuple], user_query: str, duration: int) -> List[tuple]:
        """Rank results using LLM"""
        logger.info(f"Ranking {len(results)} results")
        try:
            def ranking_prompt_llm(result_tuple, prompt, duration):
                """Rank results using LLM - simplified version"""
                # Extract text from tuple (start, end, text)
                text = result_tuple[2] if len(result_tuple) > 2 else ""
                logger.debug(f"Ranking text with prompt: {prompt[:50]}...")
                try:
                    llm = LLM()
                    ranking_prompt = f"""Given the text provided below and a specific User Prompt, evaluate the relevance of the text
                    in relation to the user's prompt. Please assign a relevance score ranging from 0 to 10, where 0 indicates no relevance 
                    and 10 signifies perfect alignment with the user's request.
                    The score quality also increases when the text is a complete sentence, making it perfect for a video clip result

                    text: {text}
                    User Prompt: {prompt}

                    Ensure the final output strictly adheres to the JSON format specified, without including additional text or explanations. 
                    Use the following structure for your response:
                    {{
                    "score": <relevance score>
                    }}
                    """
                    # ranking_prompt = f"""Rate this video segment for creating clips:

                    # Text: {text}
                    # Query: {prompt}
                    # Max Duration: {duration}

                    # Score 0-10 based on:
                    # - Relevance to query 
                    # - Complete sentences/thoughts (0-2 points) 
                    # - Clip suitability (0-2 points)

                    # Return JSON: {{"score": <0-10>}}
                    # """

                    response = llm.chat(message=ranking_prompt)
                    output = response["choices"][0]["message"]["content"]
                    import json
                    res = json.loads(output)
                    score = res.get('score', 0)
                    logger.debug(f"Ranking completed with score: {score}")
                    return score
                except Exception as e:
                    logger.error(f"Ranking failed: {str(e)}")
                    return 0

            ranked_results = []
            for result_tuple in results:
                score = ranking_prompt_llm(result_tuple, user_query, duration)
                ranked_results.append((result_tuple, score))
            
            # Sort by score (descending)
            ranked_results.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Ranked results: {ranked_results}, length: {len(ranked_results)}")
            # Return only the tuples (without scores)
            logger.info(f"Ranking completed, top score: {ranked_results[0][1] if ranked_results else 0}")
            return [result_tuple for result_tuple, score in ranked_results]
        
        except Exception as e:
            # If ranking fails, return original results
            logger.warning(f"Ranking failed: {str(e)}, returning original results")
            return results
    
    async def get_clip(self, clip_id: str) -> Dict:
        """Get clip details"""
        logger.debug(f"Getting clip with id={clip_id}")
        try:
            if clip_id in self.clips_cache:
                logger.debug(f"Clip {clip_id} found in cache")
                return self.clips_cache[clip_id]
            else:
                logger.warning(f"Clip {clip_id} not found in cache")
                raise Exception("Clip not found")
        except Exception as e:
            logger.error(f"Failed to get clip {clip_id}: {str(e)}")
            raise Exception(f"Failed to get clip: {str(e)}")
    
    async def get_all_clips(self) -> List[Dict]:
        """Get all generated clips"""
        logger.debug("Getting all clips from cache")
        try:
            clips = list(self.clips_cache.values())
            logger.debug(f"Retrieved {len(clips)} clips from cache")
            return clips
        except Exception as e:
            logger.error(f"Failed to get all clips: {str(e)}")
            raise Exception(f"Failed to get all clips: {str(e)}")
    
    async def delete_clip(self, clip_id: str) -> bool:
        """Delete a clip"""
        logger.debug(f"Deleting clip with id={clip_id}")
        try:
            if clip_id in self.clips_cache:
                del self.clips_cache[clip_id]
                
                # Update persistent storage
                clips_data = list(self.clips_cache.values())
                save_clips_data(clips_data)
                
                logger.debug(f"Clip {clip_id} deleted from cache and storage")
            else:
                logger.warning(f"Clip {clip_id} not found in cache for deletion")
            return True
        except Exception as e:
            logger.error(f"Failed to delete clip {clip_id}: {str(e)}")
            raise Exception(f"Failed to delete clip: {str(e)}") 

