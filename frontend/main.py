import streamlit as st
import requests
import json
from datetime import datetime
import time
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="VideoDB Clip Generator",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🎬 VideoDB Clip Generator</h1>', unsafe_allow_html=True)
    
    # API Key Configuration
    st.sidebar.markdown("### 🔑 API Configuration")
    api_key = st.sidebar.text_input(
        "VideoDB API Key",
        type="password",
        placeholder="Enter your VideoDB API key",
        help="Get your API key from https://console.videodb.io"
    )
    
    # Scene Extract Time Configuration
    st.sidebar.markdown("### ⏱️ Scene Extraction Settings")
    scene_extract_time = st.sidebar.number_input(
        "Scene Extract Time (seconds)",
        min_value=1,
        max_value=30,
        value=5,
        step=1,
        help="Time interval for scene extraction during multimodal indexing (1-30 seconds)"
    )
    
    if not api_key:
        st.sidebar.error("⚠️ Please enter your VideoDB API key to continue")
        st.warning("🔑 **API Key Required**")
        st.markdown("""
        To use this application, you need to provide your VideoDB API key.
        
        **How to get your API key:**
        1. Go to [VideoDB Console](https://console.videodb.io)
        2. Sign up or log in to your account
        3. Navigate to API Keys section
        4. Copy your API key and paste it in the sidebar
        
        **Security Note:** Your API key is stored only in your browser session and is not saved permanently.
        """)
        return
    
    # Store API key and scene extract time in session state
    st.session_state['videodb_api_key'] = api_key
    st.session_state['scene_extract_time'] = scene_extract_time
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["📹 Video Upload", "🎬 Clip Generation", "🎨 Customization", "📊 Analytics"]
    )
    
    if page == "📹 Video Upload":
        show_video_upload_page()
    elif page == "🎬 Clip Generation":
        show_clip_generation_page()
    elif page == "🎨 Customization":
        show_customization_page()
    elif page == "📊 Analytics":
        show_analytics_page()

def show_video_upload_page():
    st.markdown('<h2 class="sub-header">📹 Upload Video</h2>', unsafe_allow_html=True)
    
    # Show collection status
    st.markdown("### Collection Status")
    collection_status = get_collection_status()
    if collection_status:
        col1, col2 = st.columns(2)
        
        with col1:
            spoken_status = collection_status.get("spoken_word", {})
            if spoken_status.get("available"):
                st.success("✅ Spoken Word Collection")
                st.caption(f"Videos: {spoken_status.get('video_count', 0)}")
            else:
                st.error("❌ Spoken Word Collection")
                if spoken_status.get("error"):
                    st.caption(f"Error: {spoken_status['error']}")
        
        with col2:
            multimodal_status = collection_status.get("multimodal", {})
            if multimodal_status.get("available"):
                st.success("✅ Multimodal Collection")
                st.caption(f"Videos: {multimodal_status.get('video_count', 0)}")
            else:
                st.error("❌ Multimodal Collection")
                if multimodal_status.get("error"):
                    st.caption(f"Error: {multimodal_status['error']}")
    
    # Video upload form
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            video_url = st.text_input(
                "Video URL",
                placeholder="https://example.com/video.mp4",
                help="Enter the URL of the video you want to upload"
            )
            
            index_type = st.selectbox(
                "Index Type",
                ["spoken_word", "multimodal", "scene"],
                help="Choose how the video should be indexed"
            )
            
            custom_prompt = st.text_area(
                "Custom Prompt (Optional)",
                placeholder="For multimodal: Describe how scenes should be analyzed...",
                help="Custom prompt for scene analysis (multimodal only)",
                height=100
            )
        
        with col2:
            st.markdown("### Upload Options")
            st.markdown("""
            - **Spoken Words Only**: Index only the spoken content
            - **Multimodal**: Index both spoken content and visual scenes
            - **Scene**: Index visual scenes for scene-based search
            - **Custom Prompt**: Define how scenes should be described
            """)
            
            if st.button("🚀 Upload Video", type="primary"):
                if video_url:
                    # Show background indexing message immediately
                    with st.spinner("📤 Uploading video and preparing for indexing..."):
                        # Show specific indexing type message
                        current_time = datetime.now().strftime("%H:%M:%S")
                        if index_type == "multimodal":
                            st.info(f"🎬 Multimodal indexing will run in the background (spoken words + visual scenes). This may take a few minutes. Started at {current_time}")
                        elif index_type == "scene":
                            st.info(f"🎭 Scene indexing will run in the background (visual scenes only). This may take a few minutes. Started at {current_time}")
                        else:
                            st.info(f"💬 Spoken word indexing will run in the background. This may take a few minutes. Started at {current_time}")
                        upload_video(video_url, index_type, custom_prompt)
                else:
                    st.error("Please enter a video URL")
    
    # Video management section
    with st.expander("Manage Videos", expanded=True):
        videos = get_videos()
        if videos:
            st.markdown("### Your Videos")
            for video in videos:
                with st.container():
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.markdown(f"**{video['name']}**")
                        st.markdown(f"ID: `{video['id']}`")
                        st.markdown(f"Duration: {video.get('duration', 0):.1f}s")
                    
                    with col2:
                        if video.get('indexed'):
                            if video.get('index_type') == 'multimodal':
                                st.success("✅ Multimodal Indexed")
                                st.caption("Spoken words + Visual scenes")
                                st.caption(f"Collection: {video.get('collection', 'multimodal')}")
                            elif video.get('index_type') == 'scene':
                                st.success("✅ Scene Indexed")
                                st.caption("Visual scenes for scene search")
                                st.caption(f"Collection: {video.get('collection', 'multimodal')}")
                            elif video.get('index_type') == 'spoken_word':
                                st.info("✅ Spoken Words Indexed")
                                st.caption("Text queries only")
                                st.caption(f"Collection: {video.get('collection', 'spoken_word')}")
                            else:
                                st.warning("✅ Indexed (Unknown type)")
                        else:
                            st.error("❌ Not Indexed")
                            st.caption(f"Collection: {video.get('collection', 'unknown')}")
                        
                        # Show additional info for multimodal and scene videos
                        if video.get('index_type') in ['multimodal', 'scene'] and video.get('custom_prompt'):
                            st.caption("📝 Custom prompt applied")
                        
                        # Show progress for videos being processed
                        if not video.get('indexed'):
                            st.progress(0.5)  # Show progress bar
                            st.caption("⏳ Processing in background")
                    
                    with col3:
                        if st.button(f"🗑️", key=f"delete_{video['id']}"):
                            delete_video(video['id'])
                            st.rerun()
            

        else:
            st.info("No videos uploaded yet. Upload your first video above!")

def show_clip_generation_page():
    st.markdown('<h2 class="sub-header">🎬 Generate Clips</h2>', unsafe_allow_html=True)
    
    # Get available videos
    videos = get_videos()
    if not videos:
        st.error("No videos available. Please upload a video first.")
        return
    
    # Video selection
    video_options = {f"{v['name']} ({v['id']})": v['id'] for v in videos if v.get('indexed')}
    
    if not video_options:
        st.error("No indexed videos available. Please wait for indexing to complete.")
        return
    
    selected_video_name = st.selectbox("Select Video", list(video_options.keys()))
    selected_video_id = video_options[selected_video_name]
    
    # Query input
    st.markdown("### Query Input")
    query = st.text_area(
        "Describe what you want to find in the video",
        placeholder="e.g., find all funny jokes done by kapil sharma",
        height=100
    )
    
    # Advanced options
    with st.expander("Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Make index type optional with auto-detect option
            index_type_option = st.selectbox(
                "Index Type",
                ["Auto-detect", "multimodal", "spoken_word", "scene"],
                help="Auto-detect uses the video's original index type, or manually select"
            )
            
            # Convert selection to None for auto-detect or the selected value
            index_type = None if index_type_option == "Auto-detect" else index_type_option
            
            include_ranking = st.checkbox(
                "Include LLM Ranking",
                value=True,
                help="Use AI to rank and filter results"
            )
        
        with col2:
            enable_max_duration = st.checkbox(
                "Enable Max Duration",
                value=False,
                help="Limit clip duration to specified time"
            )
            
            if enable_max_duration:
                max_duration = st.number_input(
                    "Max Duration (seconds)",
                    min_value=10,
                    max_value=320,
                    value=240,
                    help="Maximum clip duration (max 320 seconds)"
                )
            else:
                max_duration = 240  # Send 240 to backend when disabled
            
            top_n = st.number_input(
                "Top N Results",
                min_value=1,
                max_value=20,
                value=10,
                help="Number of top results to include"
            )
    
    # Generate clip
    if st.button("🎬 Generate Clip", type="primary"):
        if query:
            generate_clip(selected_video_id, query, index_type, include_ranking, max_duration, top_n)
        else:
            st.error("Please enter a query")

def show_customization_page():
    st.markdown('<h2 class="sub-header">🎨 Customize Clips</h2>', unsafe_allow_html=True)
    
    # Image upload section - simplified version
    with st.expander("Upload Overlay Image", expanded=True):
        # Default image URL from notebook
        default_image_url = "https://www.freepnglogos.com/uploads/logo-ig-png/logo-ig-instagram-new-logo-vector-download-13.png"
        
        image_url = st.text_input(
            "Image URL",
            value=default_image_url,
            placeholder="https://example.com/logo.png"
        )
        
        if st.button("📤 Upload Image"):
            if image_url:
                image_id = upload_image(image_url)
                if image_id:
                    st.session_state['uploaded_image_id'] = image_id
                    st.success(f"✅ Image uploaded! Use ID: {image_id}")
            else:
                st.error("Please enter an image URL")
    
    # Clip with overlay generation
    with st.expander("Generate Clip with Overlay", expanded=True):
        videos = get_videos()
        if videos:
            video_options = {f"{v['name']} ({v['id']})": v['id'] for v in videos if v.get('indexed')}
            
            if video_options:
                selected_video_name = st.selectbox("Select Video", list(video_options.keys()), key="overlay_video")
                selected_video_id = video_options[selected_video_name]
                
                query = st.text_area(
                    "Query",
                    placeholder="e.g., find all funny jokes done by kapil sharma",
                    key="overlay_query"
                )
                
                # Index type selection for overlay
                index_type_option = st.selectbox(
                    "Index Type",
                    ["Auto-detect", "multimodal", "spoken_word", "scene"],
                    help="Auto-detect uses the video's original index type, or manually select",
                    key="overlay_index_type"
                )
                
                # Convert selection to None for auto-detect or the selected value
                index_type = None if index_type_option == "Auto-detect" else index_type_option
                
                # Max duration control for overlay
                st.markdown("### ⏱️ Duration Settings")
                enable_max_duration = st.checkbox(
                    "Enable Max Duration",
                    value=False,
                    help="Limit clip duration to specified time",
                    key="overlay_enable_max_duration"
                )
                
                if enable_max_duration:
                    max_duration = st.number_input(
                        "Max Duration (seconds)",
                        min_value=10,
                        max_value=320,
                        value=240,
                        help="Maximum clip duration (max 320 seconds)",
                        key="overlay_max_duration"
                    )
                else:
                    max_duration = 240  # Default when disabled
                
                # Overlay assets section
                st.markdown("### 🎨 Overlay Assets")
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**Image Overlay**")
                    uploaded_image_id = st.session_state.get('uploaded_image_id', '')
                    image_id = st.text_input(
                        "Image ID", 
                        value=uploaded_image_id,
                        placeholder="Enter image ID",
                        key="overlay_image_id"
                    )
                    
                    if image_id:
                        st.markdown("**Image Settings:**")
                        col_w, col_h = st.columns(2)
                        with col_w:
                            image_width = st.slider("Width", 10, 200, 40, key="image_width")
                        with col_h:
                            image_height = st.slider("Height", 10, 200, 40, key="image_height")
                        
                        col_x, col_y = st.columns(2)
                        with col_x:
                            image_x = st.slider("X Position", 0, 100, 20, key="image_x")
                        with col_y:
                            image_y = st.slider("Y Position", 0, 100, 10, key="image_y")
                        
                        image_duration = st.slider("Duration (s)", 1, 30, 7, key="image_duration")
                
                with col2:
                    st.markdown("**Audio Overlay**")
                    audio_id = st.text_input(
                        "Audio ID",
                        placeholder="Enter audio ID",
                        key="overlay_audio_id"
                    )
                    
                    if audio_id:
                        st.markdown("**Audio Settings:**")
                        col_start, col_end = st.columns(2)
                        with col_start:
                            audio_start = st.slider("Start (s)", 0, 60, 3, key="audio_start")
                        with col_end:
                            audio_end = st.slider("End (s)", 0, 60, 4, key="audio_end")
                        
                        audio_disable_other_tracks = st.checkbox("Disable Other Tracks", value=True, key="audio_disable_tracks")
                
                if st.button("🎬 Generate with Overlay", type="primary"):
                    if query:
                        # Get overlay settings
                        img_id = image_id if image_id else None
                        aud_id = audio_id if audio_id else None
                        
                        # Get image settings (use defaults if not provided)
                        img_width = st.session_state.get('image_width', 40)
                        img_height = st.session_state.get('image_height', 40)
                        img_x = st.session_state.get('image_x', 20)
                        img_y = st.session_state.get('image_y', 10)
                        img_duration = st.session_state.get('image_duration', 7)
                        
                        # Get audio settings (use defaults if not provided)
                        aud_start = st.session_state.get('audio_start', 3)
                        aud_end = st.session_state.get('audio_end', 4)
                        aud_disable = st.session_state.get('audio_disable_tracks', True)
                        
                        # Get selected index type
                        selected_index_type = st.session_state.get('overlay_index_type', 'multimodal')
                        
                        # Get max duration setting
                        overlay_max_duration = st.session_state.get('overlay_max_duration', 240)
                        overlay_enable_max_duration = st.session_state.get('overlay_enable_max_duration', False)
                        
                        # Use user-specified max duration if enabled, otherwise use 240
                        final_max_duration = overlay_max_duration if overlay_enable_max_duration else 240
                        
                        generate_clip_with_overlay(
                            selected_video_id, query, selected_index_type,
                            img_id, aud_id,
                            img_width, img_height, img_x, img_y, img_duration,
                            aud_start, aud_end, aud_disable,
                            final_max_duration
                        )
                    else:
                        st.error("Please enter a query")
            else:
                st.error("No indexed videos available")
        else:
            st.error("No videos available")

def show_analytics_page():
    st.markdown('<h2 class="sub-header">📊 Analytics</h2>', unsafe_allow_html=True)
    
    # Get videos and clips
    videos = get_videos()
    clips = get_clips()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Video Statistics")
        if videos:
            st.metric("Total Videos", len(videos))
            indexed_count = sum(1 for v in videos if v.get('indexed'))
            st.metric("Indexed Videos", indexed_count)
            
            total_duration = sum(v.get('duration', 0) for v in videos)
            st.metric("Total Duration", f"{total_duration:.1f}s")
        else:
            st.info("No videos uploaded yet")
    
    with col2:
        st.markdown("### Clip Statistics")
        if clips:
            st.metric("Total Clips", len(clips))
            
            avg_duration = sum(c.get('duration', 0) for c in clips) / len(clips)
            st.metric("Average Clip Duration", f"{avg_duration:.1f}s")
            
            # Most common queries
            queries = [c.get('query', '') for c in clips]
            if queries:
                st.markdown("**Most Common Queries:**")
                for query in queries[:5]:
                    st.markdown(f"- {query}")
        else:
            st.info("No clips generated yet")
    
    # Enhanced analytics section
    analytics = get_clips_analytics()
    if analytics:
        st.markdown("### 📊 Detailed Analytics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Duration", f"{analytics.get('total_duration', 0):.1f}s")
        
        with col2:
            st.metric("Average Duration", f"{analytics.get('average_duration', 0):.1f}s")
        
        with col3:
            st.metric("Clips with Overlays", analytics.get('clips_with_overlays', 0))
        
        with col4:
            total_clips = analytics.get('total_clips', 0)
            overlay_percentage = (analytics.get('clips_with_overlays', 0) / total_clips * 100) if total_clips > 0 else 0
            st.metric("Overlay %", f"{overlay_percentage:.1f}%")
        
        # Index type distribution
        if analytics.get('index_type_distribution'):
            st.markdown("### Index Type Distribution")
            index_data = analytics['index_type_distribution']
            for index_type, count in index_data.items():
                st.markdown(f"- **{index_type}**: {count} clips")
        
        # Most common queries
        if analytics.get('most_common_queries'):
            st.markdown("### Most Common Queries")
            for query, count in analytics['most_common_queries'][:5]:
                st.markdown(f"- **{query}** (used {count} times)")
    
    # Enhanced clip analytics
    if clips:
        st.markdown("### 📈 Detailed Clip Analytics")
        
        # Filter and sort options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sort_by = st.selectbox(
                "Sort by",
                ["Created Date", "Duration", "Query Length"],
                help="Sort clips by different criteria"
            )
        
        with col2:
            filter_overlay = st.checkbox(
                "Show only clips with overlays",
                help="Filter to show only clips with image/audio overlays"
            )
        
        with col3:
            search_query = st.text_input(
                "Search queries",
                placeholder="Filter by query text...",
                help="Search through clip queries"
            )
        
        # Apply filters
        filtered_clips = clips
        
        if filter_overlay:
            filtered_clips = [c for c in filtered_clips if c.get('has_overlay', False)]
        
        if search_query:
            filtered_clips = [c for c in filtered_clips if search_query.lower() in c.get('query', '').lower()]
        
        # Sort clips
        if sort_by == "Created Date":
            filtered_clips = sorted(filtered_clips, key=lambda x: x.get('created_at', ''), reverse=True)
        elif sort_by == "Duration":
            filtered_clips = sorted(filtered_clips, key=lambda x: x.get('duration', 0), reverse=True)
        elif sort_by == "Query Length":
            filtered_clips = sorted(filtered_clips, key=lambda x: len(x.get('query', '')), reverse=True)
        
        # Display clips with enhanced information
        st.markdown(f"**Showing {len(filtered_clips)} clips**")
        
        for i, clip in enumerate(filtered_clips):
            with st.expander(f"🎬 Clip {i+1}: {clip.get('query', 'N/A')[:50]}...", expanded=False):
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"**Query:** {clip.get('query', 'N/A')}")
                    st.markdown(f"**Video ID:** {clip.get('video_id', 'N/A')}")
                    st.markdown(f"**Index Type:** {clip.get('index_type', 'N/A')}")
                    st.markdown(f"**Segments:** {clip.get('segments_count', 0)}")
                
                with col2:
                    st.markdown(f"**Duration:** {clip.get('duration', 0):.1f}s")
                    st.markdown(f"**Created:** {clip.get('created_at', 'N/A')}")
                    
                    # Show overlay information
                    if clip.get('has_overlay'):
                        st.markdown("🎨 **Has Overlays:**")
                        if clip.get('image_overlay'):
                            st.markdown(f"  - Image: {clip['image_overlay']}")
                        if clip.get('audio_overlay'):
                            st.markdown(f"  - Audio: {clip['audio_overlay']}")
                
                with col3:
                    if clip.get('stream_url'):
                        st.markdown("**Stream URL:**")
                        st.code(clip['stream_url'], language=None)
                        st.markdown(f"[🎬 Watch Clip]({clip['stream_url']})")
                    else:
                        st.warning("No stream URL available")
    
    # Recent activity (simplified version)
    st.markdown("### Recent Activity")
    if clips:
        recent_clips = sorted(clips, key=lambda x: x.get('created_at', ''), reverse=True)[:3]
        
        for clip in recent_clips:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.markdown(f"**Query:** {clip.get('query', 'N/A')}")
                    st.markdown(f"**Video ID:** {clip.get('video_id', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Duration:** {clip.get('duration', 0):.1f}s")
                    st.markdown(f"**Created:** {clip.get('created_at', 'N/A')}")
                
                with col3:
                    if clip.get('stream_url'):
                        st.markdown(f"[Watch Clip]({clip['stream_url']})")

# API functions
def upload_video(url: str, index_type: str, custom_prompt: str = None):
    try:
        api_key = st.session_state.get('videodb_api_key')
        scene_extract_time = st.session_state.get('scene_extract_time', 5)
        response = requests.post(f"{API_BASE_URL}/videos/upload", json={
            "video_url": url,
            "index_type": index_type,
            "custom_prompt": custom_prompt,
            "scene_extract_time": scene_extract_time,
            "api_key": api_key
        })
        
        if response.status_code == 200:
            video_info = response.json()
            st.success(f"✅ Video uploaded successfully! ID: {video_info['id']}")
        else:
            st.error(f"❌ Failed to upload video: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error uploading video: {str(e)}")

def get_videos() -> List[Dict]:
    try:
        api_key = st.session_state.get('videodb_api_key')
        response = requests.get(f"{API_BASE_URL}/videos", params={"api_key": api_key})
        if response.status_code == 200:
            videos = response.json()
            return videos
        return []
    except Exception as e:
        st.error(f"❌ Error getting videos: {str(e)}")
        return []

def delete_video(video_id: str):
    """Delete a video"""
    try:
        api_key = st.session_state.get('videodb_api_key')
        response = requests.delete(f"{API_BASE_URL}/videos/{video_id}", params={"api_key": api_key})
        if response.status_code == 200:
            st.success("✅ Video deleted successfully!")
            st.rerun()
        else:
            st.error(f"❌ Failed to delete video: {response.text}")
    except Exception as e:
        st.error(f"❌ Error deleting video: {str(e)}")

def generate_clip(video_id: str, query: str, index_type: Optional[str], include_ranking: bool, max_duration: int, top_n: int):
    try:
        with st.spinner("🎬 Generating clip..."):
            api_key = st.session_state.get('videodb_api_key')
            
            # Prepare request payload
            request_payload = {
                "video_id": video_id,
                "user_query": query,
                "include_ranking": include_ranking,
                "max_duration": max_duration,
                "top_n": top_n,
                "api_key": api_key
            }
            
            # Only include index_type if it's not None
            if index_type is not None:
                request_payload["index_type"] = index_type
            
            response = requests.post(f"{API_BASE_URL}/clips/create", json=request_payload)
        
        if response.status_code == 200:
            clip_info = response.json()
            st.success("✅ Clip generated successfully!")
            
            # Display clip info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Clip ID:** {clip_info['id']}")
                st.markdown(f"**Duration:** {clip_info['duration']:.1f}s")
                st.markdown(f"**Segments:** {clip_info.get('segments_count', 0)}")
                st.markdown(f"**Index Type:** {clip_info.get('index_type', 'auto-detected')}")
            
            with col2:
                if clip_info.get('stream_url'):
                    st.markdown(f"**Stream URL:** {clip_info['stream_url']}")
                    st.markdown(f"[🎬 Watch Clip]({clip_info['stream_url']})")
        else:
            st.error(f"❌ Failed to generate clip: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error generating clip: {str(e)}")

def generate_clip_with_overlay(video_id: str, query: str, index_type: Optional[str], image_id: str = None, audio_id: str = None,
                              image_width: int = 40, image_height: int = 40, image_x: int = 20, image_y: int = 10, image_duration: int = 7,
                              audio_start: int = 3, audio_end: int = 4, audio_disable_other_tracks: bool = True,
                              max_duration: int = 240):
    try:
        with st.spinner("🎬 Generating clip with overlay..."):
            api_key = st.session_state.get('videodb_api_key')
            
            # Prepare request payload
            request_payload = {
                "video_id": video_id,
                "user_query": query,
                "include_ranking": True,
                "max_duration": max_duration,
                "top_n": 10,
                "api_key": api_key,
                # Image overlay metadata
                "image_id": image_id,
                "image_width": image_width,
                "image_height": image_height,
                "image_x": image_x,
                "image_y": image_y,
                "image_duration": image_duration,
                # Audio overlay metadata
                "audio_id": audio_id,
                "audio_start": audio_start,
                "audio_end": audio_end,
                "audio_disable_other_tracks": audio_disable_other_tracks
            }
            
            # Only include index_type if it's not None
            if index_type is not None:
                request_payload["index_type"] = index_type
            
            response = requests.post(f"{API_BASE_URL}/clips/create-with-overlay", json=request_payload)
        
        if response.status_code == 200:
            clip_info = response.json()
            st.success("✅ Clip with overlay generated successfully!")
            
            # Display clip info
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Clip ID:** {clip_info['id']}")
                st.markdown(f"**Duration:** {clip_info['duration']:.1f}s")
                st.markdown(f"**Index Type:** {clip_info.get('index_type', 'auto-detected')}")
            
            with col2:
                if clip_info.get('stream_url'):
                    st.markdown(f"**Stream URL:** {clip_info['stream_url']}")
                    st.markdown(f"[🎬 Watch Clip]({clip_info['stream_url']})")
        else:
            st.error(f"❌ Failed to generate clip: {response.text}")
    
    except Exception as e:
        st.error(f"❌ Error generating clip: {str(e)}")

def upload_image(url: str):
    """Upload an image for overlay - simplified version following notebook pattern"""
    try:
        api_key = st.session_state.get('videodb_api_key')
        response = requests.post(f"{API_BASE_URL}/videos/upload-image", json={
            "image_url": url,
            "api_key": api_key
        })
        
        if response.status_code == 200:
            image_info = response.json()
            st.success(f"✅ Image uploaded successfully! ID: {image_info['id']}")
            return image_info['id']
        else:
            st.error(f"❌ Failed to upload image: {response.text}")
            return None
    
    except Exception as e:
        st.error(f"❌ Error uploading image: {str(e)}")
        return None

def get_clips() -> List[Dict]:
    try:
        api_key = st.session_state.get('videodb_api_key')
        response = requests.get(f"{API_BASE_URL}/clips", params={"api_key": api_key})
        if response.status_code == 200:
            return response.json()
        return []
    except Exception as e:
        st.error(f"❌ Error getting clips: {str(e)}")
        return []

def get_clips_analytics() -> Dict:
    """Get detailed analytics for clips"""
    try:
        api_key = st.session_state.get('videodb_api_key')
        response = requests.get(f"{API_BASE_URL}/clips/analytics", params={"api_key": api_key})
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"❌ Error getting clips analytics: {str(e)}")
        return {}

def get_collection_status():
    """Get collection status"""
    try:
        api_key = st.session_state.get('videodb_api_key')
        response = requests.get(f"{API_BASE_URL}/videos/collections/status", params={"api_key": api_key})
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get collection status: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error getting collection status: {str(e)}")
        return None

if __name__ == "__main__":
    main() 