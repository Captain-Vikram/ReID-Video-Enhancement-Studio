"""
Video Enhancement Backend Module
Separated logic for video processing, temporal smoothing, and quality enhancement.
"""

import cv2
import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class TemporalSmoother:
    """Handles temporal smoothing of bounding boxes and confidence scores."""
    
    def __init__(self, window_size: int = 7):
        self.window_size = window_size
        self.bbox_history = defaultdict(lambda: deque(maxlen=window_size))
        self.confidence_history = defaultdict(lambda: deque(maxlen=window_size))
    
    def add_detection(self, track_id: int, bbox: Tuple[float, float, float, float], 
                     confidence: float):
        """Add detection to smoothing history."""
        self.bbox_history[track_id].append(bbox)
        self.confidence_history[track_id].append(confidence)
    
    def get_smoothed_bbox(self, track_id: int) -> Tuple[float, float, float, float]:
        """Get temporally smoothed bounding box."""
        if track_id not in self.bbox_history or len(self.bbox_history[track_id]) == 0:
            return (0, 0, 0, 0)
        
        bboxes = list(self.bbox_history[track_id])
        if len(bboxes) == 1:
            return bboxes[0]
        
        # Average all coordinates
        x1_avg = sum(bbox[0] for bbox in bboxes) / len(bboxes)
        y1_avg = sum(bbox[1] for bbox in bboxes) / len(bboxes)
        x2_avg = sum(bbox[2] for bbox in bboxes) / len(bboxes)
        y2_avg = sum(bbox[3] for bbox in bboxes) / len(bboxes)
        
        return (x1_avg, y1_avg, x2_avg, y2_avg)
    
    def get_smoothed_confidence(self, track_id: int) -> float:
        """Get temporally smoothed confidence score."""
        if track_id not in self.confidence_history or len(self.confidence_history[track_id]) == 0:
            return 0.0
        
        confidences = list(self.confidence_history[track_id])
        return sum(confidences) / len(confidences)

class MotionPredictor:
    """Handles motion prediction using Kalman filter or velocity-based methods."""
    
    def __init__(self, method: str = "kalman"):
        self.method = method
        self.trackers = {}
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
        self.position_history = defaultdict(lambda: deque(maxlen=5))
    
    def predict_position(self, track_id: int, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Predict next position based on motion history."""
        
        if self.method == "kalman":
            return self._kalman_predict(track_id, current_pos)
        else:
            return self._velocity_predict(track_id, current_pos)
    
    def _kalman_predict(self, track_id: int, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Kalman filter prediction."""
        if track_id not in self.trackers:
            # Initialize Kalman filter
            self.trackers[track_id] = cv2.KalmanFilter(4, 2)
            kalman = self.trackers[track_id]
            
            kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                               [0, 1, 0, 0]], np.float32)
            kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                              [0, 1, 0, 1],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32)
            kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
            kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
            
            # Initialize state
            kalman.statePre = np.array([current_pos[0], current_pos[1], 0, 0], dtype=np.float32).reshape(4, 1)
            kalman.statePost = np.array([current_pos[0], current_pos[1], 0, 0], dtype=np.float32).reshape(4, 1)
            
            return current_pos
        
        kalman = self.trackers[track_id]
        
        # Predict
        prediction = kalman.predict()
        
        # Update with measurement
        measurement = np.array([[current_pos[0]], [current_pos[1]]], dtype=np.float32)
        kalman.correct(measurement)
        
        return (float(prediction[0]), float(prediction[1]))
    
    def _velocity_predict(self, track_id: int, current_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Velocity-based prediction."""
        self.position_history[track_id].append(current_pos)
        
        positions = list(self.position_history[track_id])
        if len(positions) < 2:
            return current_pos
        
        # Calculate average velocity
        velocities = []
        for i in range(1, len(positions)):
            vx = positions[i][0] - positions[i-1][0]
            vy = positions[i][1] - positions[i-1][1]
            velocities.append((vx, vy))
        
        if not velocities:
            return current_pos
        
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        
        # Predict next position
        predicted_x = current_pos[0] + avg_vx
        predicted_y = current_pos[1] + avg_vy
        
        return (predicted_x, predicted_y)

class CrowdedRegionDetector:
    """Detects and handles crowded regions with multiple overlapping players."""
    
    def __init__(self, overlap_threshold: float = 0.5, crowd_threshold: int = 4):
        self.overlap_threshold = overlap_threshold
        self.crowd_threshold = crowd_threshold
    
    def detect_crowded_regions(self, detections: List[Dict]) -> List[bool]:
        """Detect which detections are in crowded regions."""
        if len(detections) < self.crowd_threshold:
            return [False] * len(detections)
        
        crowded_flags = []
        
        for i, det in enumerate(detections):
            overlap_count = 0
            bbox1 = (det['x1'], det['y1'], det['x2'], det['y2'])
            
            for j, other_det in enumerate(detections):
                if i == j:
                    continue
                
                bbox2 = (other_det['x1'], other_det['y1'], other_det['x2'], other_det['y2'])
                
                if self._calculate_iou(bbox1, bbox2) > self.overlap_threshold:
                    overlap_count += 1
            
            crowded_flags.append(overlap_count >= 2)
        
        return crowded_flags
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union of two bounding boxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class VideoEnhancer:
    """Main video enhancement engine combining all enhancement techniques."""
    
    def __init__(self, enhancement_config: Dict = None):
        self.config = enhancement_config or self._default_config()
        
        # Initialize components
        self.temporal_smoother = TemporalSmoother(self.config['temporal_window'])
        self.motion_predictor = MotionPredictor(self.config['prediction_method'])
        self.crowd_detector = CrowdedRegionDetector()
        
        # Enhancement state
        self.frame_count = 0
        self.processing_stats = {
            'total_detections': 0,
            'crowded_events': 0,
            'smoothing_applied': 0,
            'predictions_made': 0
        }
    
    def _default_config(self) -> Dict:
        """Default enhancement configuration."""
        return {
            'temporal_window': 7,
            'prediction_method': 'kalman',
            'apply_smoothing': True,
            'apply_prediction': True,
            'handle_crowds': True,
            'stabilize_confidence': True,
            'visual_enhancements': True
        }
    
    def enhance_frame_detections(self, detections: List[Dict]) -> List[Dict]:
        """Enhance detections for a single frame."""
        if not detections:
            return detections
        
        enhanced_detections = []
        
        # Detect crowded regions
        crowded_flags = self.crowd_detector.detect_crowded_regions(detections)
        if any(crowded_flags):
            self.processing_stats['crowded_events'] += 1
        
        for i, detection in enumerate(detections):
            enhanced_det = detection.copy()
            track_id = detection.get('track_id', detection.get('global_id', i))
            
            # Extract bounding box and confidence
            bbox = (detection['x1'], detection['y1'], detection['x2'], detection['y2'])
            confidence = detection.get('confidence', 1.0)
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            
            # Add to smoothing history
            self.temporal_smoother.add_detection(track_id, bbox, confidence)
            
            # Apply temporal smoothing
            if self.config['apply_smoothing']:
                smoothed_bbox = self.temporal_smoother.get_smoothed_bbox(track_id)
                enhanced_det['x1'] = smoothed_bbox[0]
                enhanced_det['y1'] = smoothed_bbox[1]
                enhanced_det['x2'] = smoothed_bbox[2]
                enhanced_det['y2'] = smoothed_bbox[3]
                self.processing_stats['smoothing_applied'] += 1
            
            # Apply motion prediction
            if self.config['apply_prediction']:
                predicted_center = self.motion_predictor.predict_position(track_id, center)
                
                # Blend predicted position with smoothed position
                current_center = ((enhanced_det['x1'] + enhanced_det['x2']) / 2,
                                (enhanced_det['y1'] + enhanced_det['y2']) / 2)
                
                blend_weight = 0.3  # Weight for prediction
                final_center_x = (1 - blend_weight) * current_center[0] + blend_weight * predicted_center[0]
                final_center_y = (1 - blend_weight) * current_center[1] + blend_weight * predicted_center[1]
                
                # Update bbox to center on predicted position
                width = enhanced_det['x2'] - enhanced_det['x1']
                height = enhanced_det['y2'] - enhanced_det['y1']
                enhanced_det['x1'] = final_center_x - width / 2
                enhanced_det['y1'] = final_center_y - height / 2
                enhanced_det['x2'] = final_center_x + width / 2
                enhanced_det['y2'] = final_center_y + height / 2
                
                self.processing_stats['predictions_made'] += 1
            
            # Apply confidence stabilization
            if self.config['stabilize_confidence']:
                smoothed_confidence = self.temporal_smoother.get_smoothed_confidence(track_id)
                enhanced_det['confidence'] = smoothed_confidence
                enhanced_det['stability_score'] = 1.0 - abs(confidence - smoothed_confidence)
            
            # Mark if in crowded region
            enhanced_det['in_crowded_region'] = crowded_flags[i]
            enhanced_det['enhancement_applied'] = True
            
            enhanced_detections.append(enhanced_det)
            self.processing_stats['total_detections'] += 1
        
        self.frame_count += 1
        return enhanced_detections
    
    def enhance_video_file(self, input_video_path: str, tracking_data_path: str, 
                          output_video_path: str, progress_callback: Callable = None) -> Dict:
        """Enhance entire video file with tracking data."""
        
        # Load tracking data
        df = pd.read_csv(tracking_data_path)
        logger.info(f"Loaded {len(df)} tracking records")
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        start_time = time.time()
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get detections for this frame
            frame_detections = df[df['frame_id'] == frame_idx].to_dict('records')
            
            # Enhance detections
            enhanced_detections = self.enhance_frame_detections(frame_detections)
            
            # Draw enhanced annotations
            annotated_frame = self._draw_enhanced_annotations(frame, enhanced_detections)
            
            # Write frame
            out.write(annotated_frame)
            
            # Update progress
            if progress_callback and frame_idx % 30 == 0:
                progress = frame_idx / total_frames
                progress_callback(progress, f"Processing frame {frame_idx}/{total_frames}")
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        
        # Generate enhancement report
        report = {
            'processing_time': processing_time,
            'frames_processed': self.frame_count,
            'enhancement_stats': self.processing_stats,
            'config_used': self.config,
            'video_properties': {
                'fps': fps,
                'resolution': (width, height),
                'total_frames': total_frames,
                'duration': total_frames / fps if fps > 0 else 0
            }
        }
        
        return report
    
    def _draw_enhanced_annotations(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw enhanced annotations on frame."""
        annotated_frame = frame.copy()
        
        for detection in detections:
            # Extract coordinates
            x1, y1, x2, y2 = int(detection['x1']), int(detection['y1']), int(detection['x2']), int(detection['y2'])
            confidence = detection.get('confidence', 1.0)
            track_id = detection.get('track_id', detection.get('global_id', 0))
            
            # Choose color based on confidence and crowded status
            if detection.get('in_crowded_region', False):
                color = (0, 165, 255)  # Orange for crowded regions
            elif confidence > 0.8:
                color = (0, 255, 0)    # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (0, 0, 255)    # Red for low confidence
            
            # Draw bounding box with enhanced styling
            thickness = 3 if detection.get('in_crowded_region', False) else 2
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw enhanced labels
            label = f"ID:{track_id}"
            conf_text = f"{confidence:.2f}"
            
            # Add uncertainty indicator for low confidence
            if confidence < 0.7:
                label += " (?)"
            
            # Add crowded region indicator
            if detection.get('in_crowded_region', False):
                label += " [CROWD]"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw confidence below bbox
            cv2.putText(annotated_frame, conf_text, (x1, y2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Add enhancement status overlay
        status_text = f"Enhanced: {len(detections)} detections"
        cv2.putText(annotated_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return annotated_frame

# Quality assessment utilities
class QualityAnalyzer:
    """Analyzes video quality metrics before and after enhancement."""
    
    @staticmethod
    def analyze_tracking_quality(df: pd.DataFrame) -> Dict:
        """Analyze tracking data quality metrics."""
        
        metrics = {
            'total_detections': len(df),
            'unique_tracks': df.get('global_id', df.get('track_id', [])).nunique() if len(df) > 0 else 0,
            'frames_with_detections': df['frame_id'].nunique() if len(df) > 0 else 0,
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
            'confidence_std': df['confidence'].std() if 'confidence' in df.columns else 0,
            'confidence_range': {
                'min': df['confidence'].min() if 'confidence' in df.columns else 0,
                'max': df['confidence'].max() if 'confidence' in df.columns else 0
            }
        }
        
        # Calculate additional metrics
        if len(df) > 0:
            frame_detection_counts = df.groupby('frame_id').size()
            metrics['avg_detections_per_frame'] = frame_detection_counts.mean()
            metrics['max_detections_per_frame'] = frame_detection_counts.max()
            metrics['crowded_frames'] = (frame_detection_counts > 8).sum()
        else:
            metrics['avg_detections_per_frame'] = 0
            metrics['max_detections_per_frame'] = 0
            metrics['crowded_frames'] = 0
        
        return metrics
    
    @staticmethod
    def compare_enhancement_quality(original_df: pd.DataFrame, enhanced_df: pd.DataFrame) -> Dict:
        """Compare original vs enhanced tracking quality."""
        
        original_metrics = QualityAnalyzer.analyze_tracking_quality(original_df)
        enhanced_metrics = QualityAnalyzer.analyze_tracking_quality(enhanced_df)
        
        comparison = {
            'original': original_metrics,
            'enhanced': enhanced_metrics,
            'improvements': {}
        }
        
        # Calculate improvements
        for metric in ['avg_confidence', 'confidence_std']:
            if metric in original_metrics and metric in enhanced_metrics:
                original_val = original_metrics[metric]
                enhanced_val = enhanced_metrics[metric]
                
                if original_val != 0:
                    improvement = ((enhanced_val - original_val) / original_val) * 100
                    comparison['improvements'][metric] = improvement
        
        return comparison
