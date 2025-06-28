"""
Cross-Camera Player Mapping - Professional Main Runner
Enhanced strategic approach with professional quality outputs.
"""

import sys
import os
import json
from pathlib import Path
import logging

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the enhanced strategic mapping system
from enhanced_strategic_mapping import EnhancedStrategicMapping
from enhanced_video_renderer import EnhancedVideoRenderer
import config

def setup_environment():
    """Setup the environment and verify requirements."""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        config.OUTPUT_DIR,
        config.ANNOTATED_VIDEOS_DIR,
        config.DATA_OUTPUT_DIR,
        config.REPORTS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Verify input files
    if not config.BROADCAST_VIDEO.exists():
        print(f"❌ Error: Broadcast video not found at {config.BROADCAST_VIDEO}")
        return False
    
    if not config.TACTICAM_VIDEO.exists():
        print(f"❌ Error: Tacticam video not found at {config.TACTICAM_VIDEO}")
        return False
    
    if not Path(config.MODEL_PATH).exists():
        print(f"❌ Error: YOLO model not found at {config.MODEL_PATH}")
        return False
    
    print("✅ Environment setup complete")
    return True

def main():
    """Main function to run the cross-camera player mapping system."""
    print("🚀 Cross-Camera Player Mapping System")
    print("=" * 70)
    print("Professional implementation with enhanced strategic approach")
    print()
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Please check your input files.")
        return
    
    try:
        # Initialize and run the enhanced strategic system
        print("🎯 Initializing Enhanced Strategic Mapping System...")
        mapper = EnhancedStrategicMapping()
        
        print("🎬 Processing videos with cross-camera player mapping...")
        results = mapper.process_videos_enhanced()
        
        # Display results
        if results["status"] == "success":
            print("\n✅ Cross-Camera Player Mapping completed successfully!")
            print(f"⏱️  Processing time: {results['processing_time']:.2f} seconds")
            print(f"📊 Total detections: {results['total_detections']}")
            print(f"👥 Unique global IDs: {results['unique_global_ids']}")
            print(f"🔗 Cross-camera matches: {results['cross_camera_matches']}")
            
            print("\n📁 Basic tracking outputs generated:")
            print(f"   • Broadcast video: {config.ANNOTATED_VIDEOS_DIR}/broadcast_enhanced_strategic.mp4")
            print(f"   • Tacticam video: {config.ANNOTATED_VIDEOS_DIR}/tacticam_enhanced_strategic.mp4")
            print(f"   • Tracking data: {config.DATA_OUTPUT_DIR}/enhanced_strategic_player_tracking.csv")
            print(f"   • Processing report: {config.REPORTS_DIR}/enhanced_strategic_processing_report.json")
            
            # Now run enhanced video rendering for broadcast-quality output
            print("\n� Generating Broadcast-Quality Enhanced Videos...")
            print("=" * 50)
            
            try:
                # Configuration for enhanced rendering
                tracking_data_path = str(config.DATA_OUTPUT_DIR / "enhanced_strategic_player_tracking.csv")
                video_paths = {
                    "broadcast": str(config.BROADCAST_VIDEO),
                    "tacticam": str(config.TACTICAM_VIDEO)
                }
                enhanced_output_dir = str(Path(config.OUTPUT_DIR) / "enhanced_videos")
                
                # Create enhanced video renderer
                print("🔧 Initializing Broadcast-Quality Renderer...")
                renderer = EnhancedVideoRenderer(tracking_data_path, video_paths, enhanced_output_dir)
                
                print("🎯 Enhancement Features Active:")
                print("   • 7-frame temporal smoothing for stable bounding boxes")
                print("   • Kalman filter prediction for motion smoothing")
                print("   • Rolling average confidence display")
                print("   • Crowded region detection and handling")
                print("   • Uncertain identity labeling")
                print("   • Professional broadcast-quality styling")
                print()
                
                # Render enhanced videos
                print("🎥 Rendering broadcast-quality videos...")
                enhanced_results = renderer.render_all_cameras()
                
                # Generate quality report
                print("📊 Generating quality metrics...")
                quality_report = renderer.generate_quality_report()
                
                # Save quality report
                import json
                report_path = Path(enhanced_output_dir) / "broadcast_quality_report.json"
                with open(report_path, 'w') as f:
                    json.dump(quality_report, f, indent=2)
                
                # Display enhanced results
                print("\n�🎉 BROADCAST-QUALITY VIDEO RENDERING COMPLETE!")
                print("=" * 60)
                
                success_count = 0
                for camera, path in enhanced_results.items():
                    if path:
                        print(f"✅ {camera.upper()}: {path}")
                        success_count += 1
                    else:
                        print(f"❌ {camera.upper()}: Enhancement failed")
                
                if success_count > 0:
                    print(f"\n📈 Quality Metrics:")
                    for camera, metrics in quality_report["quality_metrics"].items():
                        conf = metrics["average_confidence"] * 100
                        stability = metrics["confidence_stability"] * 100
                        detections = metrics["avg_detections_per_frame"]
                        print(f"   {camera.upper()}:")
                        print(f"     • Average Confidence: {conf:.1f}%")
                        print(f"     • Confidence Stability: {stability:.1f}%")
                        print(f"     • Avg Detections/Frame: {detections:.1f}")
                    
                    print(f"\n📁 Final Output Files:")
                    print(f"   • Original videos: {config.ANNOTATED_VIDEOS_DIR}/")
                    print(f"   • Enhanced videos: {enhanced_output_dir}/")
                    print(f"   • Tracking data: {tracking_data_path}")
                    print(f"   • Quality report: {report_path}")
                    
                    print("\n🏆 COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
                    print("🎯 Both basic tracking AND broadcast-quality enhanced videos generated!")
                    
                else:
                    print("\n⚠️  Enhanced video rendering failed, but basic tracking succeeded.")
                    print("📋 Check the basic tracking outputs in the outputs/ directory.")
                
            except Exception as enhance_error:
                print(f"\n⚠️  Enhanced video rendering failed: {enhance_error}")
                print("📋 Basic tracking completed successfully. Enhanced rendering encountered an error.")
                print("💡 You can manually run 'python src/render_enhanced_videos.py' to retry enhancement.")
            
            print("\n📋 Check all generated files for detailed results and annotations.")
            
        else:
            print(f"\n❌ Processing failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
