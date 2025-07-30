#!/usr/bin/env python3
"""
Startup script for the VideoDB Clip Generator Backend
"""

import os
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Add the backend directory to Python path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

def main():
    """Start the FastAPI backend server"""
    
    # Load environment variables from .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print("✅ Loaded environment variables from .env file")
    else:
        print("⚠️  Warning: .env file not found!")
        print("Please create a .env file with your API keys:")
        print("OPENAI_API_KEY=your-openai-api-key")
        print("Note: VIDEO_DB_API_KEY will be provided through the Streamlit interface.")
        print()
    
    # Check required environment variables (only OpenAI API key is required from env)
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        print("Note: VIDEO_DB_API_KEY will be provided through the Streamlit interface.")
        return 1
    
    print("🚀 Starting VideoDB Clip Generator Backend...")
    print("📡 API will be available at: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 