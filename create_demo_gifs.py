#!/usr/bin/env python3
"""
Convert MP4 videos to GIFs for GitHub display
GitHub doesn't play MP4 files directly, but GIFs work perfectly.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True)
        return True
    except FileNotFoundError:
        return False

def convert_video_to_gif(input_path, output_path, max_width=800, fps=10):
    """Convert MP4 to optimized GIF."""
    print(f"Converting {input_path} to {output_path}...")
    
    # FFmpeg command for high-quality GIF conversion
    cmd = [
        'ffmpeg', '-i', str(input_path),
        '-vf', f'fps={fps},scale={max_width}:-1:flags=lanczos,palettegen=reserve_transparent=0',
        '-y', 'palette.png'
    ]
    
    # Generate palette
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error generating palette: {result.stderr}")
        return False
    
    # Convert to GIF using palette
    cmd = [
        'ffmpeg', '-i', str(input_path), '-i', 'palette.png',
        '-lavfi', f'fps={fps},scale={max_width}:-1:flags=lanczos[x];[x][1:v]paletteuse',
        '-y', str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Clean up palette
    if os.path.exists('palette.png'):
        os.remove('palette.png')
    
    if result.returncode == 0:
        print(f"✅ Successfully created {output_path}")
        return True
    else:
        print(f"❌ Error converting: {result.stderr}")
        return False

def main():
    print("🎬 Converting videos to GIFs for GitHub display")
    print("=" * 50)
    
    # Check for ffmpeg
    if not check_ffmpeg():
        print("❌ FFmpeg not found!")
        print("\nTo install FFmpeg:")
        print("Windows: Download from https://ffmpeg.org/download.html")
        print("Linux: sudo apt install ffmpeg")
        print("Mac: brew install ffmpeg")
        return
    
    # Create gifs directory
    gif_dir = Path('gifs')
    gif_dir.mkdir(exist_ok=True)
    
    # Video files to convert
    video_dir = Path('outputs/enhanced_videos')
    video_files = list(video_dir.glob('*.mp4'))
    
    if not video_files:
        print("❌ No MP4 files found in outputs/enhanced_videos/")
        return
    
    print(f"Found {len(video_files)} video files to convert:")
    for video in video_files:
        print(f"  - {video.name}")
    
    print("\n🔄 Starting conversion...")
    
    success_count = 0
    for video_path in video_files:
        gif_name = video_path.stem + '.gif'
        gif_path = gif_dir / gif_name
        
        if convert_video_to_gif(video_path, gif_path):
            success_count += 1
    
    print(f"\n🎉 Conversion complete!")
    print(f"✅ Successfully converted {success_count}/{len(video_files)} videos")
    print(f"📁 GIFs saved in: {gif_dir.absolute()}")
    
    # Generate markdown for README
    if success_count > 0:
        print("\n📝 Add this to your README.md:")
        print("```markdown")
        print("## 🎥 Video Demonstrations")
        print()
        
        for video_path in video_files:
            gif_name = video_path.stem + '.gif'
            gif_path = gif_dir / gif_name
            if gif_path.exists():
                print(f"### {video_path.stem.replace('_', ' ').title()}")
                print(f"![{video_path.stem}](gifs/{gif_name})")
                print()
        
        print("```")

if __name__ == "__main__":
    main()
