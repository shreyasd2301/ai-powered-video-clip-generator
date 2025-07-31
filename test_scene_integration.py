#!/usr/bin/env python3
"""
Test script for scene integration in ClipService
"""

import asyncio
import os
from dotenv import load_dotenv
from services.clip_service import ClipService

load_dotenv()

async def test_scene_integration():
    """Test scene search functionality"""
    print("Testing scene integration...")
    
    # Initialize clip service
    api_key = os.getenv("VIDEODB_API_KEY")
    if not api_key:
        print("❌ VIDEODB_API_KEY not found in environment")
        return
    
    clip_service = ClipService(api_key=api_key)
    
    # Test video ID (you'll need to replace this with an actual video that has scene indexing)
    test_video_id = "your_test_video_id_here"
    test_query = "find exciting moments"
    
    print(f"Testing scene search for video: {test_video_id}")
    print(f"Query: {test_query}")
    
    try:
        # Test scene search
        scene_timestamps = clip_service.search_scene_content(test_video_id, test_query)
        print(f"✅ Scene search completed. Found {len(scene_timestamps)} timestamps")
        
        for i, (start, end, description) in enumerate(scene_timestamps):
            print(f"  {i+1}. {start}s - {end}s: {description[:100]}...")
        
        # Test clip generation with scene index type
        print("\nTesting clip generation with scene index type...")
        clip_info = await clip_service.generate_clip(
            video_id=test_video_id,
            user_query=test_query,
            index_type="scene",
            max_duration=60
        )
        
        print(f"✅ Clip generated successfully!")
        print(f"  Clip ID: {clip_info['id']}")
        print(f"  Duration: {clip_info['duration']}s")
        print(f"  Stream URL: {clip_info['stream_url']}")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_scene_integration()) 