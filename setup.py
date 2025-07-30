#!/usr/bin/env python3
"""
Setup script for VideoDB Clip Generator
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install all dependencies"""
    print("\n📦 Installing dependencies...")
    
    # Install backend dependencies
    print("Installing backend dependencies...")
    backend_dir = Path(__file__).parent / "backend"
    if backend_dir.exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(backend_dir / "requirements.txt")
            ], check=True)
            print("✅ Backend dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install backend dependencies: {e}")
            return False
    else:
        print("❌ Backend directory not found")
        return False
    
    # Install frontend dependencies
    print("Installing frontend dependencies...")
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", 
                str(frontend_dir / "requirements.txt")
            ], check=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install frontend dependencies: {e}")
            return False
    else:
        print("❌ Frontend directory not found")
        return False
    
    return True

def setup_environment():
    """Set up environment file"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(__file__).parent / ".env"
    env_sample = Path(__file__).parent / "env.sample"
    
    if not env_file.exists():
        if env_sample.exists():
            # Copy sample env file
            import shutil
            shutil.copy(env_sample, env_file)
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file with your API keys")
        else:
            # Create basic env file
            with open(env_file, 'w') as f:
                f.write("""# VideoDB Clip Generator Environment Variables
# Get your API keys from:
# OpenAI: https://platform.openai.com/api-keys
# VideoDB API key will be collected through the Streamlit interface

OPENAI_API_KEY="your-openai-api-key"
# ANTHROPIC_KEY="your-anthropic-key"  # Optional
# GEMINI_API_KEY="your-gemini-key"    # Optional
""")
            print("✅ Created .env file")
            print("⚠️  Please edit .env file with your API keys")
    else:
        print("✅ .env file already exists")
    
    return True

def check_api_keys():
    """Check if API keys are configured"""
    print("\n🔑 Checking API keys...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key) or os.getenv(key) == f"your-{key.lower().replace('_', '-')}":
            missing_keys.append(key)
    
    if missing_keys:
        print("❌ Missing or invalid API keys:")
        for key in missing_keys:
            print(f"   - {key}")
        print("\nPlease edit .env file with your actual API keys")
        print("Note: VIDEO_DB_API_KEY will be provided through the Streamlit interface.")
        return False
    
    print("✅ API keys are configured")
    print("ℹ️  VIDEO_DB_API_KEY will be collected through the Streamlit interface")
    return True

def test_imports():
    """Test if all imports work"""
    print("\n🧪 Testing imports...")
    
    try:
        import fastapi
        import streamlit
        import videodb
        import openai
        print("✅ All required packages imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🎬 VideoDB Clip Generator Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Setup environment
    if not setup_environment():
        return 1
    
    # Test imports
    if not test_imports():
        return 1
    
    # Check API keys
    if not check_api_keys():
        print("\n⚠️  Setup completed with warnings")
        print("Please configure your API keys before running the application")
    else:
        print("\n✅ Setup completed successfully!")
    
    print("\n🚀 Next steps:")
    print("1. Edit .env file with your API keys")
    print("2. Start the backend: python start_backend.py")
    print("3. Start the frontend: python start_frontend.py")
    print("4. Or run the demo: python demo.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 