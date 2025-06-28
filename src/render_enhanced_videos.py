"""
Enhanced Video Rendering Script
Run this to generate professional-quality annotated videos with smooth tracking.
"""

import sys
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from enhanced_video_renderer import EnhancedVideoRenderer

def main():
    """Main function to run enhanced video rendering."""
    print("[SYSTEM] Enhanced Video Rendering System")
    print("=" * 60)
    print("Creating professional-quality annotated tracking videos...")
    print()
    
    # Configuration
    tracking_data_path = "outputs/data/enhanced_strategic_player_tracking.csv"
    video_paths = {
        "broadcast": "data/broadcast.mp4",
        "tacticam": "data/tacticam.mp4"
    }
    output_dir = "outputs/enhanced_videos"
    
    # Verify input files
    if not Path(tracking_data_path).exists():
        print(f"[ERROR] Tracking data not found at {tracking_data_path}")
        print("Please run main.py first to generate tracking data.")
        return
    
    for camera, video_path in video_paths.items():
        if not Path(video_path).exists():
            print(f"[ERROR] Video file not found at {video_path}")
            return
    
    print("[SUCCESS] Input files verified")
    print()
    
    try:
        # Create enhanced video renderer
        print("[INIT] Initializing Enhanced Video Renderer...")
        renderer = EnhancedVideoRenderer(tracking_data_path, video_paths, output_dir)
        
        print("[INFO] Enhancement Features:")
        print("   • 7-frame sliding window for smooth bounding boxes")
        print("   • Kalman filter prediction for motion smoothing")
        print("   • Rolling average confidence display")
        print("   • Crowded region detection and handling")
        print("   • Uncertain identity labeling")
        print("   • Professional visual styling")
        print()
        
        # Render enhanced videos for all cameras
        print("[RENDER] Rendering enhanced videos...")
        results = renderer.render_all_cameras()
        
        # Generate quality report
        print("[REPORT] Generating quality report...")
        report = renderer.generate_quality_report()
        
        # Save quality report
        report_path = Path(output_dir) / "enhancement_quality_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Display results
        print("\n[SUCCESS] Enhanced Video Rendering Completed!")
        print("=" * 60)
        
        success_count = 0
        for camera, path in results.items():
            if path:
                print(f"[VIDEO] {camera.upper()}: {path}")
                success_count += 1
            else:
                print(f"[ERROR] {camera.upper()}: Processing failed")
        
        print(f"\n[STATS] Quality Metrics:")
        for camera, metrics in report["quality_metrics"].items():
            conf = metrics["average_confidence"] * 100
            stability = metrics["confidence_stability"] * 100
            detections = metrics["avg_detections_per_frame"]
            print(f"   {camera.upper()}:")
            print(f"     • Average Confidence: {conf:.1f}%")
            print(f"     • Confidence Stability: {stability:.1f}%")
            print(f"     • Avg Detections/Frame: {detections:.1f}")
        
        print(f"\n[OUTPUT] Output Files:")
        print(f"   • Enhanced videos: {output_dir}/")
        print(f"   • Quality report: {report_path}")
        
        if success_count == len(video_paths):
            print("\n[COMPLETE] All enhanced videos generated successfully!")
            print("[FEATURES] Features implemented:")
            print("   [SUCCESS] Smooth bounding box tracking")
            print("   [SUCCESS] Stable confidence display")
            print("   [SUCCESS] Crowded region handling")
            print("   [SUCCESS] Motion prediction")
            print("   [SUCCESS] Professional visual quality")
        else:
            print(f"\n[WARNING] {success_count}/{len(video_paths)} videos processed successfully")
            
    except Exception as e:
        print("[ERROR] Enhanced rendering failed:")
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
