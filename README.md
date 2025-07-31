# VideoDB Clip Generator

A powerful video clip generation application built with FastAPI backend and Streamlit frontend, leveraging VideoDB and OpenAI technology to create intelligent video clips from user queries.

## 🚀 Features

- **Video Upload**: Upload videos from YouTube
- **Intelligent Indexing**: Index videos for spoken words and/or visual scenes
- **AI-Powered Clip Generation**: Generate clips using natural language queries
- **Multimodal Processing**: Support for both spoken content and visual scene analysis
- **LLM Ranking**: AI-powered ranking of clip segments for better results
- **Custom Overlays**: Add images and audio overlays to generated clips
- **Modern UI**: Beautiful Streamlit interface with real-time feedback
- **Analytics**: Track video and clip statistics

## 📋 Prerequisites

- Python 3.8+
- VideoDB API key
- OpenAI API key
- Internet connection for video processing

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ai-powered-video-clip-generator
   ```

2. **Set up environment variables**
   ```bash
   cp env.sample .env
   ```
   
   Edit `.env` file with your API keys:
   ```env
   OPENAI_API_KEY="your-openai-api-key"
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

### Backend (FastAPI)

1. **Start the backend server**
   ```bash
   python start_backend.py
   ```
   
   The API will be available at `http://localhost:8000`
   
   You can also use uvicorn directly:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Access API documentation**
   - Swagger UI: `http://localhost:8000/docs`

### Frontend (Streamlit)

1. **Start the Streamlit app**
   ```bash
   python start_frontend.py
   ```
   
   The app will be available at `http://localhost:8501`

## 📖 Usage Guide

### 1. Video Upload

1. Navigate to the "📹 Video Upload" page
2. Enter a video URL (YouTube, etc.)
3. Choose indexing type:
   - **Spoken Words Only**: Index only the spoken content
   - **Multimodal**: Index both spoken content and visual scenes
4. Optionally add a custom scene indexing prompt
5. Click "🚀 Upload Video"

### 2. Clip Generation

1. Navigate to the "🎬 Clip Generation" page
2. Select an indexed video from the dropdown
3. Enter your query (e.g., "find all funny jokes done by kapil sharma")
4. Configure advanced options:
   - Index type (spoken words or multimodal)
   - Include LLM ranking
   - Max duration and top N results
5. Click "🎬 Generate Clip"

### 3. Customization

1. Navigate to the "🎨 Customization" page
2. Upload overlay images with positioning settings
3. Generate clips with custom overlays
4. Add audio overlays for enhanced clips

### 4. Analytics

1. Navigate to the "📊 Analytics" page
2. View video and clip statistics
3. Monitor recent activity
4. Track performance metrics

## 🔧 API Endpoints

### Video Management
- `POST /videos/upload` - Upload a new video
- `GET /videos/` - Get all videos
- `GET /videos/{video_id}` - Get specific video details
- `DELETE /videos/{video_id}` - Delete a video

### Clip Generation
- `POST /clips/create` - Generate a clip from query
- `POST /clips/create-with-overlay` - Generate clip with overlays
- `GET /clips/` - Get all generated clips
- `GET /clips/{clip_id}` - Get specific clip details

### Media Management
- `POST /videos/upload-image` - Upload image for overlay

### Health Check
- `GET /health` - API health status

## 🎯 Example Queries

Here are some specific example queries you can try:

- **Comedy Shows**: "find all jokes about marriage and relationships"
- **Educational Content**: "find explanations about machine learning algorithms"
- **Sports Highlights**: "find all scoring moments and celebrations"
- **Interviews**: "find questions about career advice and success stories"
- **Presentations**: "find key insights about artificial intelligence trends"
- **Music Performances**: "find guitar solos and instrumental breaks"
- **News Segments**: "find discussions about climate change policies"
- **Tutorial Videos**: "find step-by-step instructions for coding"
- **Product Reviews**: "find pros and cons mentioned about smartphones"
- **Cooking Shows**: "find recipe instructions for pasta dishes"

## 🔍 Advanced Features

### Multimodal Indexing
When using multimodal indexing, the system analyzes both:
- **Spoken content**: What people are saying
- **Visual scenes**: What's happening on screen

This provides more accurate results for queries that require understanding both audio and visual elements.

### LLM Ranking
The system uses AI to rank and filter clip segments based on:
- Relevance to the query
- Completeness of content
- Quality for viewing experience
- Contextual coherence

### Custom Overlays
Add professional touches to your clips:
- **Image overlays**: Logos, watermarks, graphics
- **Audio overlays**: Background music, sound effects
- **Positioning**: Precise control over overlay placement

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- [VideoDB](https://videodb.io) for video processing capabilities
- [Streamlit](https://streamlit.io) for the beautiful UI framework
- [FastAPI](https://fastapi.tiangolo.com) for the robust backend API
- [OpenAI](https://openai.com) for LLM capabilities

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation at `/docs` when running the backend
- Join our community discussions

---

**Happy Clip Generating! 🎬✨** 