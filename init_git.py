#!/usr/bin/env python3
"""
Git Repository Initialization Script for ReID Video Enhancement Studio

This script helps initialize a Git repository with proper configuration,
creates initial commit, and sets up branching structure.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {description}: {e}")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return None

def check_git_installed():
    """Check if Git is installed."""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Git is not installed or not found in PATH")
        print("   Please install Git from: https://git-scm.com/")
        return False

def initialize_git_repo():
    """Initialize Git repository with proper configuration."""
    if not check_git_installed():
        return False
    
    print("🚀 Initializing ReID Video Enhancement Studio Git Repository")
    print("=" * 60)
    
    # Check if already a git repository
    if os.path.exists(".git"):
        print("📁 Git repository already exists!")
        response = input("   Do you want to continue? (y/n): ").lower()
        if response != 'y':
            print("   Aborted by user")
            return False
    
    # Initialize repository
    run_command("git init", "Initializing Git repository")
    
    # Configure Git (optional - user can skip)
    print("\n🔧 Git Configuration (optional)")
    configure = input("   Configure Git user name and email? (y/n): ").lower()
    if configure == 'y':
        name = input("   Enter your name: ").strip()
        email = input("   Enter your email: ").strip()
        
        if name:
            run_command(f'git config user.name "{name}"', f"Setting user name to {name}")
        if email:
            run_command(f'git config user.email "{email}"', f"Setting user email to {email}")
    
    # Set up Git configuration for the repository
    print("\n⚙️  Configuring repository settings...")
    
    # Set default branch name
    run_command("git config init.defaultBranch main", "Setting default branch to 'main'")
    
    # Configure line endings
    if sys.platform == "win32":
        run_command("git config core.autocrlf true", "Configuring line endings for Windows")
    else:
        run_command("git config core.autocrlf input", "Configuring line endings for Unix/Mac")
    
    # Set up useful aliases
    aliases = [
        ("git config alias.st status", "Adding 'git st' alias for status"),
        ("git config alias.co checkout", "Adding 'git co' alias for checkout"),
        ("git config alias.br branch", "Adding 'git br' alias for branch"),
        ("git config alias.cm commit", "Adding 'git cm' alias for commit"),
        ("git config alias.lg 'log --oneline --graph --all'", "Adding 'git lg' alias for pretty log"),
    ]
    
    for command, description in aliases:
        run_command(command, description)
    
    # Create initial commit
    print("\n📝 Creating initial commit...")
    run_command("git add .", "Staging all files")
    
    commit_message = """Initial commit: ReID Video Enhancement Studio

✨ Features:
- AI-powered person re-identification and tracking
- Strategic camera angle mapping and enhancement
- Professional video rendering with quality improvements
- Interactive Streamlit GUI interface
- Command-line interface for batch processing
- Comprehensive documentation and user manual

🏗️ Project Structure:
- Core enhancement algorithms in src/
- Interactive GUI in gui/
- Pre-trained models support
- Sample data and comprehensive outputs
- Detailed documentation and guides

🛠️ Technical Stack:
- Python 3.8+ with OpenCV, PyTorch
- YOLO-based object detection and tracking
- Streamlit for web interface
- Advanced video processing and rendering

📚 Documentation:
- README.md with video demonstrations
- USER_MANUAL.md with comprehensive guide
- CONTRIBUTING.md for collaboration
- Technical reports and API documentation

Ready for development and collaboration! 🚀"""
    
    run_command(f'git commit -m "{commit_message}"', "Creating initial commit")
    
    # Create development branches
    print("\n🌿 Setting up branch structure...")
    branches = [
        ("develop", "Development branch for ongoing work"),
        ("feature/gui-improvements", "Branch for GUI enhancements"),
        ("feature/performance-optimization", "Branch for performance improvements"),
    ]
    
    for branch, description in branches:
        run_command(f"git checkout -b {branch}", f"Creating {branch}")
        run_command("git checkout main", "Returning to main branch")
    
    # Create .gitkeep files for empty directories
    print("\n📁 Creating .gitkeep files for empty directories...")
    empty_dirs = [
        "models",
        "outputs/enhanced_videos",
        "outputs/reports",
        "outputs/data",
        "logs",
    ]
    
    for dir_path in empty_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        gitkeep_path = Path(dir_path) / ".gitkeep"
        if not gitkeep_path.exists():
            gitkeep_path.write_text("# This file keeps the directory in Git\n")
            print(f"   Created {gitkeep_path}")
    
    # Display repository status
    print("\n📊 Repository Status:")
    print("=" * 40)
    run_command("git status", "Checking repository status")
    run_command("git branch -a", "Listing all branches")
    
    print("\n🎉 Git repository initialization complete!")
    print("\n📋 Next Steps:")
    print("   1. Review the files that were committed")
    print("   2. Set up a remote repository (GitHub, GitLab, etc.)")
    print("   3. Push to remote: git remote add origin <url> && git push -u origin main")
    print("   4. Start developing on the 'develop' branch: git checkout develop")
    print("   5. Create feature branches from 'develop' for new features")
    print("\n🔗 Useful Commands:")
    print("   git st                    # Check status")
    print("   git lg                    # View commit history")
    print("   git co develop           # Switch to develop branch")
    print("   git co -b feature/name   # Create new feature branch")
    
    return True

if __name__ == "__main__":
    try:
        success = initialize_git_repo()
        if success:
            print("\n✨ Repository ready for development!")
        else:
            print("\n💥 Repository initialization failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Initialization cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
