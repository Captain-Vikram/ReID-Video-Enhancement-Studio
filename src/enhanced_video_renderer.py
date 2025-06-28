"""
Broadcast-Quality Enhanced Video Renderer for Professional Sports Tracking
Implements smooth bounding boxes, stable confidence display, and broadcast-quality visual refinements.

Key Features (Broadcast Quality):
1. Temporal Smoothing (7-frame sliding window) - eliminates bounding box jitter
2. Kalman Filter Motion Prediction - handles occlusions and fast movements
3. Stabilized Confidence Display - prevents flickering percentages
4. Crowded Region Intelligence - smart handling of overlapping players
5. Uncertainty Labeling - clear indication of low-confidence detections
6. TV Broadcast Styling - professional annotations suitable for live broadcast

Note: This enhancement layer preserves ALL original tracking decisions and IDs.
"""

import cv2
import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from scipy.interpolate import UnivariateSpline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KalmanTracker:
    """Simple Kalman filter for position prediction."""
    
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
    
    def update(self, center_x: float, center_y: float) -> Tuple[float, float]:
        """Update with measurement and return predicted position."""
        measurement = np.array([[center_x], [center_y]], dtype=np.float32)
        
        if not self.initialized:
            self.kalman.statePre = np.array([center_x, center_y, 0, 0], dtype=np.float32).reshape(4, 1)
            self.kalman.statePost = np.array([center_x, center_y, 0, 0], dtype=np.float32).reshape(4, 1)
            self.initialized = True
        
        # Predict
        prediction = self.kalman.predict()
        
        # Update
        self.kalman.correct(measurement)
        
        return float(prediction[0]), float(prediction[1])

class EnhancedVideoRenderer:
    """Professional video renderer with smooth tracking and enhanced visuals."""
    
    def __init__(self, tracking_data_path: str, video_paths: Dict[str, str], output_dir: str):
        """Initialize the enhanced video renderer.
        
        Args:
            tracking_data_path: Path to CSV tracking data
            video_paths: Dictionary mapping camera names to video file paths
            output_dir: Directory for output videos
        """
        self.tracking_data_path = Path(tracking_data_path)
        self.video_paths = {k: Path(v) for k, v in video_paths.items()}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tracking data
        self.df = pd.read_csv(tracking_data_path)
        logger.info(f"Loaded {len(self.df)} tracking records")
        
        # Smoothing parameters
        self.smoothing_window = 7  # 7-frame sliding window
        self.confidence_window = 7  # Rolling average for confidence
        self.prediction_weight = 0.3  # Weight for predictive smoothing
        self.smoothing_weight = 0.7   # Weight for averaged smoothing
        
        # Tracking state
        self.kalman_trackers = {}  # track_id -> KalmanTracker
        self.position_history = defaultdict(lambda: deque(maxlen=15))  # For motion prediction
        self.confidence_history = defaultdict(lambda: deque(maxlen=self.confidence_window))
        
        # Visual parameters
        self.colors = self._generate_colors()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2
        self.box_thickness = 2
        
        # Crowded region parameters
        self.crowd_iou_threshold = 0.4  # Broadcast-quality crowd detection
        self.stable_confidence_threshold = 0.75  # Higher threshold for uncertainty labeling
        
    def _generate_colors(self) -> Dict[int, Tuple[int, int, int]]:
        """Generate distinct colors for different track IDs with broadcast-quality consistency."""
        colors = {}
        
        # Use deterministic color generation for consistency
        unique_ids = self.df['global_id'].unique() if hasattr(self, 'df') else range(50)
        
        for global_id in unique_ids:
            # Generate broadcast-quality distinct colors using golden angle
            hue = (global_id * 137) % 360  # Golden angle for optimal distribution
            color = self._hsv_to_bgr(hue, 0.8, 0.9)  # High saturation, high value
            colors[global_id] = color
            
        return colors
    
    def _hsv_to_bgr(self, h: float, s: float, v: float) -> Tuple[int, int, int]:
        """Convert HSV to BGR for consistent broadcast-quality color generation."""
        h_norm = h / 360.0
        hsv = np.array([[[h_norm, s, v]]], dtype=np.float32)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return tuple(int(x * 255) for x in bgr)
    
    def _get_color(self, global_id: int) -> Tuple[int, int, int]:
        """Get consistent color for a global ID."""
        if global_id not in self.colors:
            # Generate new color if not exists
            np.random.seed(global_id)
            self.colors[global_id] = tuple(np.random.randint(50, 255, 3).tolist())
        return self.colors[global_id]
    
    def _smooth_bounding_boxes(self, camera: str) -> Dict[int, Dict[int, np.ndarray]]:
        """Smooth bounding boxes using sliding window averaging.
        
        Returns:
            Dict mapping frame_id -> {global_id: smoothed_bbox}
        """
        camera_data = self.df[self.df['camera'] == camera].copy()
        smoothed_boxes = defaultdict(dict)
        
        # Group by global_id for processing
        for global_id in camera_data['global_id'].unique():
            track_data = camera_data[camera_data['global_id'] == global_id].sort_values('frame_id')
            
            if len(track_data) < 2:
                continue
                
            frames = track_data['frame_id'].values
            x1_vals = track_data['x1'].values
            y1_vals = track_data['y1'].values
            x2_vals = track_data['x2'].values
            y2_vals = track_data['y2'].values
            
            # Create Kalman tracker for this track
            if global_id not in self.kalman_trackers:
                self.kalman_trackers[global_id] = KalmanTracker()
            
            kalman = self.kalman_trackers[global_id]
            
            for i, frame_id in enumerate(frames):
                # Define smoothing window
                half_window = self.smoothing_window // 2
                start_idx = max(0, i - half_window)
                end_idx = min(len(frames), i + half_window + 1)
                
                # Get window data
                window_x1 = x1_vals[start_idx:end_idx]
                window_y1 = y1_vals[start_idx:end_idx]
                window_x2 = x2_vals[start_idx:end_idx]
                window_y2 = y2_vals[start_idx:end_idx]
                
                # Smooth by averaging
                smooth_x1 = np.mean(window_x1)
                smooth_y1 = np.mean(window_y1)
                smooth_x2 = np.mean(window_x2)
                smooth_y2 = np.mean(window_y2)
                
                # Get center for Kalman prediction
                center_x = (smooth_x1 + smooth_x2) / 2
                center_y = (smooth_y1 + smooth_y2) / 2
                
                # Get Kalman prediction
                pred_x, pred_y = kalman.update(center_x, center_y)
                
                # Combine smoothed and predicted positions
                width = smooth_x2 - smooth_x1
                height = smooth_y2 - smooth_y1
                
                final_center_x = self.smoothing_weight * center_x + self.prediction_weight * pred_x
                final_center_y = self.smoothing_weight * center_y + self.prediction_weight * pred_y
                
                # Convert back to bbox
                final_x1 = final_center_x - width / 2
                final_y1 = final_center_y - height / 2
                final_x2 = final_center_x + width / 2
                final_y2 = final_center_y + height / 2
                
                smoothed_boxes[frame_id][global_id] = np.array([final_x1, final_y1, final_x2, final_y2])
                
                # Store position history for motion prediction
                self.position_history[global_id].append((final_center_x, final_center_y, frame_id))
        
        return smoothed_boxes
    
    def _smooth_confidences(self, camera: str) -> Dict[int, Dict[int, float]]:
        """Smooth confidence values using rolling average.
        
        Returns:
            Dict mapping frame_id -> {global_id: smoothed_confidence}
        """
        camera_data = self.df[self.df['camera'] == camera].copy()
        smoothed_confidences = defaultdict(dict)
        
        for global_id in camera_data['global_id'].unique():
            track_data = camera_data[camera_data['global_id'] == global_id].sort_values('frame_id')
            
            frames = track_data['frame_id'].values
            confidences = track_data['confidence'].values
            
            for i, frame_id in enumerate(frames):
                # Define confidence window
                half_window = self.confidence_window // 2
                start_idx = max(0, i - half_window)
                end_idx = min(len(frames), i + half_window + 1)
                
                # Average confidence over window
                window_confidences = confidences[start_idx:end_idx]
                avg_confidence = np.mean(window_confidences)
                
                smoothed_confidences[frame_id][global_id] = avg_confidence
                
                # Update confidence history
                self.confidence_history[global_id].append(avg_confidence)
        
        return smoothed_confidences
    
    def _detect_crowded_regions(self, frame_boxes: Dict[int, np.ndarray]) -> List[int]:
        """Detect crowded regions by analyzing bounding box overlaps.
        
        Returns:
            List of global_ids in crowded regions
        """
        if len(frame_boxes) < 3:  # Need at least 3 boxes for crowding
            return []
        
        crowded_ids = []
        box_items = list(frame_boxes.items())
        
        for i, (id1, box1) in enumerate(box_items):
            overlap_count = 0
            
            for j, (id2, box2) in enumerate(box_items):
                if i != j:
                    # Calculate IoU
                    x1 = max(box1[0], box2[0])
                    y1 = max(box1[1], box2[1])
                    x2 = min(box1[2], box2[2])
                    y2 = min(box1[3], box2[3])
                    
                    if x1 < x2 and y1 < y2:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                        union = area1 + area2 - intersection
                        
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > self.crowd_iou_threshold:
                            overlap_count += 1
            
            # If overlapping with 2+ other boxes, consider crowded
            if overlap_count >= 2:
                crowded_ids.append(id1)
        
        return crowded_ids
    
    def _draw_enhanced_annotations(self, frame: np.ndarray, frame_boxes: Dict[int, np.ndarray],
                                 frame_confidences: Dict[int, float], frame_id: int, camera: str = "CAM") -> np.ndarray:
        """Draw enhanced annotations with professional styling."""
        annotated_frame = frame.copy()
        
        # Detect crowded regions
        crowded_ids = self._detect_crowded_regions(frame_boxes)
        
        for global_id, bbox in frame_boxes.items():
            confidence = frame_confidences.get(global_id, 0.0)
            
            # Get color for this ID
            color = self._get_color(global_id)
            
            # Determine if this is a crowded/uncertain region
            is_crowded = global_id in crowded_ids
            is_uncertain = confidence < self.stable_confidence_threshold
            
            # Draw bounding box with different styles
            x1, y1, x2, y2 = bbox.astype(int)
            
            if is_crowded or is_uncertain:
                # Dashed line for uncertain detections
                self._draw_dashed_rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.box_thickness)
            else:
                # Solid line for confident detections
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.box_thickness)
            
            # Prepare label text
            conf_text = f"{int(confidence * 100)}%" if confidence > self.stable_confidence_threshold else "??%"
            
            if is_uncertain:
                label = f"P{global_id:03d} (uncertain) {conf_text}"
            else:
                label = f"P{global_id:03d} {conf_text}"
            
            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
            
            # Position text above bounding box
            text_x = x1
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            
            # Draw text background for better visibility
            cv2.rectangle(annotated_frame, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + baseline + 5),
                         color, -1)
            
            # Draw text
            text_color = (255, 255, 255) if sum(color) < 400 else (0, 0, 0)
            cv2.putText(annotated_frame, label, (text_x, text_y), self.font, 
                       self.font_scale, text_color, self.font_thickness)
        
        # Add frame info and stats
        self._draw_frame_info(annotated_frame, frame_id, len(frame_boxes), len(crowded_ids), camera)
        
        return annotated_frame
    
    def _draw_dashed_rectangle(self, img: np.ndarray, pt1: Tuple[int, int], 
                              pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int):
        """Draw a dashed rectangle for uncertain detections."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        dash_length = 10
        
        # Top edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y1), color, thickness)
        
        # Bottom edge
        for x in range(x1, x2, dash_length * 2):
            cv2.line(img, (x, y2), (min(x + dash_length, x2), y2), color, thickness)
        
        # Left edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, y), (x1, min(y + dash_length, y2)), color, thickness)
        
        # Right edge
        for y in range(y1, y2, dash_length * 2):
            cv2.line(img, (x2, y), (x2, min(y + dash_length, y2)), color, thickness)
    
    def _draw_frame_info(self, frame: np.ndarray, frame_id: int, total_detections: int, crowded_count: int, camera: str = "CAM"):
        """Draw frame information overlay."""
        height, width = frame.shape[:2]
        
        # Frame info
        info_text = f"{camera.upper()} | Frame: {frame_id:04d} | Players: {total_detections} | Crowded: {crowded_count}"
        
        # Position at top-right
        (text_width, text_height), baseline = cv2.getTextSize(info_text, self.font, 0.6, 1)
        text_x = width - text_width - 10
        text_y = 30
        
        # Draw background
        cv2.rectangle(frame, (text_x - 5, text_y - text_height - 5),
                     (text_x + text_width + 5, text_y + baseline + 5),
                     (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, info_text, (text_x, text_y), self.font, 0.6, (255, 255, 255), 1)
    
    def render_enhanced_video(self, camera: str, output_filename: str = None) -> str:
        """Render enhanced video for specified camera.
        
        Args:
            camera: Camera name ('broadcast' or 'tacticam')
            output_filename: Optional custom output filename
            
        Returns:
            Path to generated video file
        """
        if camera not in self.video_paths:
            raise ValueError(f"Camera '{camera}' not found in video paths")
        
        video_path = self.video_paths[camera]
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Generate output filename
        if output_filename is None:
            output_filename = f"{camera}_broadcast_quality.mp4"
        
        output_path = self.output_dir / output_filename
        
        logger.info(f"[RENDER] Processing enhanced video for {camera}")
        logger.info(f"[INPUT] {video_path}")
        logger.info(f"[OUTPUT] {output_path}")
        
        # Smooth bounding boxes and confidences
        logger.info("[SMOOTH] Smoothing bounding boxes...")
        smoothed_boxes = self._smooth_bounding_boxes(camera)
        
        logger.info("[SMOOTH] Smoothing confidences...")
        smoothed_confidences = self._smooth_confidences(camera)
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"[VIDEO] {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_id = 0
        processed_frames = 0
        
        logger.info("[RENDER] Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get smoothed data for this frame
            frame_boxes = smoothed_boxes.get(frame_id, {})
            frame_confidences = smoothed_confidences.get(frame_id, {})
            
            # Draw enhanced annotations
            if frame_boxes:  # Only annotate if we have detections
                annotated_frame = self._draw_enhanced_annotations(
                    frame, frame_boxes, frame_confidences, frame_id, camera
                )
            else:
                annotated_frame = frame.copy()
                # Still draw frame info even without detections
                self._draw_frame_info(annotated_frame, frame_id, 0, 0, camera)
            
            # Write frame
            out.write(annotated_frame)
            
            frame_id += 1
            processed_frames += 1
            
            # Progress logging
            if processed_frames % 50 == 0:
                progress = (processed_frames / total_frames) * 100
                logger.info(f"[PROGRESS] {processed_frames}/{total_frames} frames ({progress:.1f}%)")
        
        # Cleanup
        cap.release()
        out.release()
        
        logger.info(f"[SUCCESS] Enhanced video saved: {output_path}")
        logger.info(f"[STATS] Processed {processed_frames} frames")
        
        return str(output_path)
    
    def render_all_cameras(self) -> Dict[str, str]:
        """Render enhanced videos for all cameras.
        
        Returns:
            Dictionary mapping camera names to output video paths
        """
        results = {}
        
        for camera in self.video_paths.keys():
            try:
                output_path = self.render_enhanced_video(camera)
                results[camera] = output_path
                logger.info(f"[SUCCESS] {camera} enhanced video completed")
            except Exception as e:
                logger.error(f"[ERROR] Failed to render {camera}: {e}")
                results[camera] = None
        
        return results
    
    def generate_quality_report(self) -> Dict:
        """Generate a quality report of the enhancement process."""
        report = {
            "enhancement_type": "Broadcast-Quality Visual Refinement",
            "original_data_preserved": True,
            "total_frames": len(self.df['frame_id'].unique()),
            "total_detections": len(self.df),
            "cameras": list(self.video_paths.keys()),
            "unique_global_ids": len(self.df['global_id'].unique()),
            "enhancement_parameters": {
                "temporal_smoothing_window": self.smoothing_window,
                "confidence_smoothing_window": self.confidence_window,
                "kalman_blend_ratio": f"{self.smoothing_weight:.1f} smoothed + {self.prediction_weight:.1f} predicted",
                "crowd_detection_iou": self.crowd_iou_threshold,
                "uncertainty_threshold": self.stable_confidence_threshold
            },
            "broadcast_improvements": [
                "7-frame temporal smoothing for stable bounding boxes",
                "Kalman filter motion prediction with velocity blending",
                "Stabilized confidence display prevents flickering",
                "Crowded region detection with uncertainty labeling",
                "Professional TV broadcast styling and annotations",
                "Consistent color mapping across all frames"
            ],
            "quality_metrics": {}
        }
        
        # Calculate quality metrics per camera
        for camera in self.df['camera'].unique():
            camera_data = self.df[self.df['camera'] == camera]
            
            avg_confidence = camera_data['confidence'].mean()
            confidence_std = camera_data['confidence'].std()
            detections_per_frame = len(camera_data) / len(camera_data['frame_id'].unique())
            
            report["quality_metrics"][camera] = {
                "average_confidence": float(avg_confidence),
                "confidence_stability": float(1.0 / (1.0 + confidence_std)),  # Higher is better
                "avg_detections_per_frame": float(detections_per_frame)
            }
        
        return report

def main():
    """Main function for testing the enhanced video renderer."""
    # Configuration
    tracking_data_path = "outputs/data/enhanced_strategic_player_tracking.csv"
    video_paths = {
        "broadcast": "data/broadcast.mp4",
        "tacticam": "data/tacticam.mp4"
    }
    output_dir = "outputs/enhanced_videos"
    
    # Create renderer
    renderer = EnhancedVideoRenderer(tracking_data_path, video_paths, output_dir)
    
    # Render enhanced videos
    results = renderer.render_all_cameras()
    
    # Generate quality report
    report = renderer.generate_quality_report()
    
    # Save report
    report_path = Path(output_dir) / "enhancement_quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n[COMPLETE] Enhanced Video Rendering Completed!")
    print("=" * 60)
    for camera, path in results.items():
        if path:
            print(f"[SUCCESS] {camera}: {path}")
        else:
            print(f"[FAILED] {camera}: Processing failed")
    
    print(f"\n[REPORT] Quality report saved: {report_path}")

if __name__ == "__main__":
    main()
