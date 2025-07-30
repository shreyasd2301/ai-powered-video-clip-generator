#!/usr/bin/env python3
"""
Startup script for the VideoDB Clip Generator Frontend
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_backend():
    """Check if backend is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Start the Streamlit frontend"""
    
    print("🎬 VideoDB Clip Generator Frontend")
    print("=" * 50)
    
    # Check if backend is running
    print("🔍 Checking backend connection...")
    if not check_backend():
        print("❌ Backend is not running!")
        print("Please start the backend first:")
        print("   python start_backend.py")
        print("   or")
        print("   cd backend && python main.py")
        return 1
    
    print("✅ Backend is running")
    print()
    
    # Check if frontend dependencies are installed
    frontend_dir = Path(__file__).parent / "frontend"
    requirements_file = frontend_dir / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ Frontend requirements not found!")
        return 1
    
    print("🚀 Starting Streamlit frontend...")
    print("🌐 App will be available at: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Change to frontend directory and start streamlit
        os.chdir(frontend_dir)
        
        # Start streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 