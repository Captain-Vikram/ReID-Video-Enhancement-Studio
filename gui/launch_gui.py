#!/usr/bin/env python3
"""
Launch Script for ReID Video Enhancement Studio
Simplified startup with automatic dependency checking and sample data generation.
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'streamlit', 'opencv-python', 'pandas', 'numpy', 
        'plotly', 'scipy', 'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_dependencies(packages):
    """Install missing dependencies."""
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)

def generate_sample_data():
    pass

def launch_streamlit():
    """Launch the Streamlit application."""
    print("🚀 Launching ReID Video Enhancement Studio...")
    print("🌐 Opening in your default web browser...")
    print("📍 URL: http://localhost:8501")
    print("\n💡 Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def main():
    """Main launch sequence."""
    print("🎬 ReID Video Enhancement Studio Launcher")
    print("=" * 50)
    
    # Check current directory (app.py should be in the same gui folder)
    if not Path("app.py").exists():
        print("❌ Error: app.py not found in current directory")
        print("💡 Please run this script from the gui directory")
        return
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    missing = check_dependencies()
    
    if missing:
        try:
            install_dependencies(missing)
            print("✅ Dependencies installed successfully")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("💡 Please run: pip install -r requirements.txt")
            return
    else:
        print("✅ All dependencies satisfied")
    
    # Launch application
    print("\n" + "=" * 50)
    launch_streamlit()

if __name__ == "__main__":
    main()
