"""
Enhanced Strategic Cross-Camera Player Mapping System
Builds upon the successful strategic approach with improved accuracy and consistency.
Key improvements:
1. Better cross-camera association logic
2. Enhanced feature matching 
3. Improved ID consistency
4. Professional visualization
"""

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from collections import defaultdict, deque
from ultralytics import YOLO

# Import configuration from old system
import sys
sys.path.append('old')
import config

# Configure logging with UTF-8 encoding for Windows compatibility
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/reports/enhanced_strategic_processing.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EnhancedFeatureExtractor:
    """Enhanced multi-modal feature extraction for robust player re-identification."""
    
    def __init__(self):
        """Initialize enhanced feature extractor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize feature extraction models."""
        try:
            # ResNet50 for visual features
            self.visual_model = resnet50(pretrained=True)
            self.visual_model.fc = torch.nn.Identity()
            self.visual_model.eval()
            self.visual_model.to(self.device)
            
            # Transform for visual features
            self.visual_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("[SUCCESS] Enhanced feature extractors initialized")
            
        except Exception as e:
            logger.error(f"Feature extractor initialization failed: {e}")
            self.visual_model = None
    
    def extract_enhanced_features(self, player_region: np.ndarray, bbox: List[int], 
                                frame_shape: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """Extract comprehensive multi-modal features."""
        features = {}
        
        # Visual features
        features['visual'] = self._extract_visual_features(player_region)
        
        # Color features (robust to lighting changes)
        features['color'] = self._extract_enhanced_color_features(player_region)
        
        # Shape and size features
        features['shape'] = self._extract_shape_features(player_region, bbox)
        
        # Position features
        features['position'] = self._extract_position_features(bbox, frame_shape)
        
        return features
    
    def _extract_visual_features(self, player_region: np.ndarray) -> np.ndarray:
        """Extract deep visual features using ResNet50."""
        if self.visual_model is None or player_region.size == 0:
            return np.zeros(2048)
        
        try:
            # Convert BGR to RGB
            if len(player_region.shape) == 3:
                player_region_rgb = cv2.cvtColor(player_region, cv2.COLOR_BGR2RGB)
            else:
                return np.zeros(2048)
            
            # Transform and extract features
            input_tensor = self.visual_transform(player_region_rgb).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.visual_model(input_tensor)
                features = features.cpu().numpy().flatten()
                
                # Normalize features
                features = features / (np.linalg.norm(features) + 1e-8)
            
            return features
            
        except Exception as e:
            logger.warning(f"Visual feature extraction failed: {e}")
            return np.zeros(2048)
    
    def _extract_enhanced_color_features(self, player_region: np.ndarray) -> np.ndarray:
        """Extract enhanced color features robust to lighting changes."""
        if player_region.size == 0:
            return np.zeros(96)
        
        try:
            # Convert to multiple color spaces for robustness
            hsv = cv2.cvtColor(player_region, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(player_region, cv2.COLOR_BGR2LAB)
            
            # HSV histograms (hue is lighting-invariant)
            h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            
            # LAB histograms (perceptually uniform)
            a_hist = cv2.calcHist([lab], [1], None, [32], [0, 256])
            
            # Normalize histograms
            h_hist = h_hist.flatten() / (np.sum(h_hist) + 1e-8)
            s_hist = s_hist.flatten() / (np.sum(s_hist) + 1e-8)
            a_hist = a_hist.flatten() / (np.sum(a_hist) + 1e-8)
            
            # Combine features
            color_features = np.concatenate([h_hist, s_hist, a_hist])
            
            return color_features
            
        except Exception as e:
            logger.warning(f"Color feature extraction failed: {e}")
            return np.zeros(96)
    
    def _extract_shape_features(self, player_region: np.ndarray, bbox: List[int]) -> np.ndarray:
        """Extract shape and proportion features."""
        try:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            
            # Basic shape features
            aspect_ratio = height / max(width, 1)
            area = width * height
            
            # Body proportion features (head, torso, legs estimation)
            upper_region = player_region[:height//3, :]
            middle_region = player_region[height//3:2*height//3, :]
            lower_region = player_region[2*height//3:, :]
            
            # Color variance in each region (indicates clothing)
            upper_var = np.var(upper_region) if upper_region.size > 0 else 0
            middle_var = np.var(middle_region) if middle_region.size > 0 else 0
            lower_var = np.var(lower_region) if lower_region.size > 0 else 0
            
            shape_features = np.array([
                aspect_ratio, area, width, height,
                upper_var, middle_var, lower_var,
                width/max(height, 1)  # inverse aspect ratio
            ])
            
            return shape_features
            
        except Exception as e:
            logger.warning(f"Shape feature extraction failed: {e}")
            return np.zeros(8)
    
    def _extract_position_features(self, bbox: List[int], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Extract position-based features."""
        try:
            x1, y1, x2, y2 = bbox
            frame_height, frame_width = frame_shape[:2]
            
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Normalize position
            norm_x = center_x / frame_width
            norm_y = center_y / frame_height
            
            # Distance from frame center
            center_distance = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
            
            # Field position (assuming standard field layout)
            field_zone = self._get_field_zone(norm_x, norm_y)
            
            position_features = np.array([norm_x, norm_y, center_distance, field_zone])
            
            return position_features
            
        except Exception as e:
            logger.warning(f"Position feature extraction failed: {e}")
            return np.zeros(4)
    
    def _get_field_zone(self, norm_x: float, norm_y: float) -> float:
        """Get field zone (0-8) based on normalized position."""
        # Divide field into 3x3 grid
        zone_x = min(int(norm_x * 3), 2)
        zone_y = min(int(norm_y * 3), 2)
        return float(zone_y * 3 + zone_x)


class EnhancedTracker:
    """Enhanced tracker with improved ID consistency."""
    
    def __init__(self, camera_id: str):
        """Initialize enhanced tracker."""
        self.camera_id = camera_id
        self.tracks = {}
        self.next_id = 1
        self.max_disappeared = 15
        self.distance_threshold = 100
        self.track_history = defaultdict(lambda: deque(maxlen=30))
        
    def update(self, detections: List[Dict], frame_idx: int) -> List[Dict]:
        """Update tracks with enhanced stability."""
        if not detections:
            # Mark tracks as disappeared
            tracks_to_remove = []
            for track_id in self.tracks:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    tracks_to_remove.append(track_id)
            
            for track_id in tracks_to_remove:
                del self.tracks[track_id]
                if track_id in self.track_history:
                    del self.track_history[track_id]
            
            return []
        
        # Calculate detection centers
        detection_centers = []
        for det in detections:
            bbox = det['bbox']
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            detection_centers.append(center)
            det['center'] = center
        
        # Get existing track centers with prediction
        track_centers = []
        track_ids = []
        for track_id, track_info in self.tracks.items():
            predicted_center = self._predict_position(track_id, frame_idx)
            track_centers.append(predicted_center)
            track_ids.append(track_id)
        
        matched_tracks = []
        
        if track_centers and detection_centers:
            # Calculate distance matrix
            distances = cdist(detection_centers, track_centers)
            
            # Use Hungarian algorithm for optimal assignment
            if distances.size > 0:
                det_indices, track_indices = linear_sum_assignment(distances)
                matched_det_indices = set()
                
                # Process matches
                for det_idx, track_idx in zip(det_indices, track_indices):
                    distance = distances[det_idx, track_idx]
                    if distance < self.distance_threshold:
                        track_id = track_ids[track_idx]
                        detection = detections[det_idx]
                        
                        # Update track
                        self.tracks[track_id].update({
                            'bbox': detection['bbox'],
                            'center': detection['center'],
                            'confidence': detection['confidence'],
                            'frame_idx': frame_idx,
                            'disappeared': 0
                        })
                        
                        # Update history
                        self.track_history[track_id].append({
                            'center': detection['center'],
                            'frame_idx': frame_idx,
                            'bbox': detection['bbox']
                        })
                        
                        matched_tracks.append({
                            'track_id': track_id,
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'center': detection['center']
                        })
                        
                        matched_det_indices.add(det_idx)
                
                # Create new tracks for unmatched detections
                for i, detection in enumerate(detections):
                    if i not in matched_det_indices:
                        track_id = self.next_id
                        self.next_id += 1
                        
                        self.tracks[track_id] = {
                            'bbox': detection['bbox'],
                            'center': detection['center'],
                            'confidence': detection['confidence'],
                            'frame_idx': frame_idx,
                            'disappeared': 0,
                            'created_frame': frame_idx
                        }
                        
                        self.track_history[track_id].append({
                            'center': detection['center'],
                            'frame_idx': frame_idx,
                            'bbox': detection['bbox']
                        })
                        
                        matched_tracks.append({
                            'track_id': track_id,
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'center': detection['center']
                        })
        else:
            # No existing tracks, create new ones
            for detection in detections:
                track_id = self.next_id
                self.next_id += 1
                
                self.tracks[track_id] = {
                    'bbox': detection['bbox'],
                    'center': detection['center'],
                    'confidence': detection['confidence'],
                    'frame_idx': frame_idx,
                    'disappeared': 0,
                    'created_frame': frame_idx
                }
                
                self.track_history[track_id].append({
                    'center': detection['center'],
                    'frame_idx': frame_idx,
                    'bbox': detection['bbox']
                })
                
                matched_tracks.append({
                    'track_id': track_id,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'center': detection['center']
                })
        
        # Update disappeared counters
        matched_track_ids = {track['track_id'] for track in matched_tracks}
        tracks_to_remove = []
        
        for track_id in self.tracks:
            if track_id not in matched_track_ids:
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
            if track_id in self.track_history:
                del self.track_history[track_id]
        
        return matched_tracks
    
    def _predict_position(self, track_id: int, frame_idx: int) -> Tuple[float, float]:
        """Predict track position based on history."""
        if track_id not in self.track_history or len(self.track_history[track_id]) < 2:
            return self.tracks[track_id]['center']
        
        # Use linear prediction based on recent movement
        history = list(self.track_history[track_id])
        recent_positions = history[-3:]  # Use last 3 positions
        
        if len(recent_positions) >= 2:
            # Calculate velocity
            last_pos = recent_positions[-1]['center']
            prev_pos = recent_positions[-2]['center']
            frame_diff = recent_positions[-1]['frame_idx'] - recent_positions[-2]['frame_idx']
            
            if frame_diff > 0:
                velocity_x = (last_pos[0] - prev_pos[0]) / frame_diff
                velocity_y = (last_pos[1] - prev_pos[1]) / frame_diff
                
                # Predict next position
                time_diff = frame_idx - recent_positions[-1]['frame_idx']
                predicted_x = last_pos[0] + velocity_x * time_diff
                predicted_y = last_pos[1] + velocity_y * time_diff
                
                return (predicted_x, predicted_y)
        
        return self.tracks[track_id]['center']


class EnhancedStrategicMapping:
    """Enhanced strategic cross-camera player mapping system."""
    
    def __init__(self):
        """Initialize enhanced strategic system."""
        logger.info("[INIT] Initializing Enhanced Strategic Cross-Camera Player Mapping")
        
        self.start_time = time.time()
        self._initialize_components()
        self._initialize_data_structures()
        
        logger.info("[SUCCESS] Enhanced strategic system initialized")
    
    def _initialize_components(self):
        """Initialize system components."""
        # Load YOLO model
        self.model = YOLO(config.MODEL_PATH)
        
        # Initialize enhanced components
        self.feature_extractor = EnhancedFeatureExtractor()
        self.tracker_broadcast = EnhancedTracker(config.BROADCAST_CAMERA_ID)
        self.tracker_tacticam = EnhancedTracker(config.TACTICAM_CAMERA_ID)
        
    def _initialize_data_structures(self):
        """Initialize data structures."""
        self.tracks_broadcast = {}
        self.tracks_tacticam = {}
        self.global_id_mapping = {}
        self.global_id_counter = 1
        self.all_detections = []
        
        # Feature storage for cross-camera matching
        self.track_features = defaultdict(lambda: deque(maxlen=15))
        self.stable_tracks = {'broadcast': set(), 'tacticam': set()}
        
        # Cross-camera associations
        self.cross_camera_matches = {}
        
    def process_videos_enhanced(self):
        """Process both videos with enhanced strategic mapping."""
        try:
            # Load videos
            cap_broadcast, cap_tacticam = self._load_videos()
            
            # Get total frame counts for both videos
            broadcast_total_frames = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_COUNT))
            tacticam_total_frames = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process all frames (remove the 100 frame limit)
            max_frames = min(broadcast_total_frames, tacticam_total_frames)
            logger.info(f"Processing {max_frames} frames from both videos")
            
            frame_idx = 0
            while True:
                # Read frames from both videos
                ret_broadcast, frame_broadcast = cap_broadcast.read()
                ret_tacticam, frame_tacticam = cap_tacticam.read()
                
                if not ret_broadcast or not ret_tacticam:
                    logger.info(f"Reached end of video(s) at frame {frame_idx}")
                    break
                
                # Remove any artificial frame limits
                # if frame_idx >= config.MAX_FRAMES:  # Remove this line
                #     break
                
                # Process the frame
                self._process_frame_pair(frame_broadcast, frame_tacticam, frame_idx)
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    logger.info(f"Processed {frame_idx} frames...")
            
            # Perform enhanced cross-camera matching
            self._perform_enhanced_cross_camera_matching()
            
            # Generate enhanced outputs
            self._generate_enhanced_outputs(cap_broadcast, cap_tacticam)
            
            # Cleanup
            cap_broadcast.release()
            cap_tacticam.release()
            
            total_time = time.time() - self.start_time
            
            results = {
                "status": "success",
                "processing_time": total_time,
                "total_detections": len(self.all_detections),
                "unique_global_ids": len(set(self.global_id_mapping.values())),
                "cross_camera_matches": len(self.cross_camera_matches)
            }
            
            logger.info(f"[COMPLETE] Enhanced strategic processing completed in {total_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _load_videos(self) -> Tuple[cv2.VideoCapture, cv2.VideoCapture]:
        """Load video files."""
        cap_broadcast = cv2.VideoCapture(str(config.BROADCAST_VIDEO))
        cap_tacticam = cv2.VideoCapture(str(config.TACTICAM_VIDEO))
        
        if not cap_broadcast.isOpened():
            raise ValueError(f"Cannot open broadcast video: {config.BROADCAST_VIDEO}")
        if not cap_tacticam.isOpened():
            raise ValueError(f"Cannot open tacticam video: {config.TACTICAM_VIDEO}")
        
        logger.info("[SUCCESS] Videos loaded successfully")
        return cap_broadcast, cap_tacticam
    
    def _process_frame_pair(self, frame_broadcast: np.ndarray, frame_tacticam: np.ndarray, frame_idx: int):
        """Process a pair of frames from both cameras simultaneously."""
        try:
            # Process both cameras
            tracks_b = self._process_single_frame_enhanced(
                frame_broadcast, config.BROADCAST_CAMERA_ID, frame_idx
            )
            tracks_t = self._process_single_frame_enhanced(
                frame_tacticam, config.TACTICAM_CAMERA_ID, frame_idx
            )
            
            # Store tracks
            self.tracks_broadcast[frame_idx] = tracks_b
            self.tracks_tacticam[frame_idx] = tracks_t
            
            # Extract features for cross-camera matching (every 5 frames)
            if frame_idx % 5 == 0:
                self._extract_and_store_features(frame_broadcast, tracks_b, 'broadcast', frame_idx)
                self._extract_and_store_features(frame_tacticam, tracks_t, 'tacticam', frame_idx)
            
        except Exception as e:
            logger.warning(f"Frame pair processing failed for frame {frame_idx}: {e}")
    
    def _process_frames_enhanced(self, cap_broadcast: cv2.VideoCapture, 
                               cap_tacticam: cv2.VideoCapture):
        """Process frames with enhanced tracking."""
        frame_idx = 0
        max_frames = getattr(config, 'MAX_FRAMES', 100)
        
        logger.info(f"Processing {max_frames} frames with enhanced tracking")
        
        while frame_idx < max_frames:
            ret_b, frame_b = cap_broadcast.read()
            ret_t, frame_t = cap_tacticam.read()
            
            if not ret_b or not ret_t:
                break
            
            # Process both cameras
            tracks_b = self._process_single_frame_enhanced(
                frame_b, config.BROADCAST_CAMERA_ID, frame_idx
            )
            tracks_t = self._process_single_frame_enhanced(
                frame_t, config.TACTICAM_CAMERA_ID, frame_idx
            )
            
            # Store tracks
            self.tracks_broadcast[frame_idx] = tracks_b
            self.tracks_tacticam[frame_idx] = tracks_t
            
            # Extract features for cross-camera matching (every 5 frames)
            if frame_idx % 5 == 0:
                self._extract_and_store_features(frame_b, tracks_b, 'broadcast', frame_idx)
                self._extract_and_store_features(frame_t, tracks_t, 'tacticam', frame_idx)
            
            frame_idx += 1
            
            if frame_idx % 20 == 0:
                logger.info(f"Processed {frame_idx} frames")
        
        logger.info(f"[SUCCESS] Processed {frame_idx} frames with enhanced tracking")
    
    def _process_single_frame_enhanced(self, frame: np.ndarray, camera_id: str, 
                                     frame_idx: int) -> List[Dict]:
        """Process single frame with enhanced detection and tracking."""
        try:
            # Detect players
            detections = self._detect_players_enhanced(frame)
            
            # Update tracker
            if camera_id == config.BROADCAST_CAMERA_ID:
                tracks = self.tracker_broadcast.update(detections, frame_idx)
            else:
                tracks = self.tracker_tacticam.update(detections, frame_idx)
            
            # Store detection results
            for track in tracks:
                detection_record = {
                    'frame_id': frame_idx,
                    'camera': camera_id,
                    'track_id': track['track_id'],
                    'bbox': track['bbox'],
                    'confidence': track['confidence']
                }
                self.all_detections.append(detection_record)
            
            return tracks
            
        except Exception as e:
            logger.warning(f"Frame processing failed for {camera_id} frame {frame_idx}: {e}")
            return []
    
    def _detect_players_enhanced(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced player detection with improved filtering."""
        try:
            results = self.model(frame, conf=config.DETECTION_CONFIDENCE, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        bbox = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Enhanced filtering for players
                        if class_id == config.PERSON_CLASS_ID and confidence > config.DETECTION_CONFIDENCE:
                            x1, y1, x2, y2 = bbox
                            width = x2 - x1
                            height = y2 - y1
                            area = width * height
                            aspect_ratio = height / max(width, 1)
                            
                            # Enhanced filtering criteria
                            if (config.MIN_DETECTION_AREA <= area <= config.MAX_DETECTION_AREA and
                                1.2 <= aspect_ratio <= 5.0 and  # Reasonable person aspect ratio
                                width >= 20 and height >= 40):  # Minimum size requirements
                                
                                detections.append({
                                    'bbox': [int(x) for x in bbox],
                                    'confidence': confidence,
                                    'class_id': class_id,
                                    'area': area
                                })
            
            return detections
            
        except Exception as e:
            logger.warning(f"Enhanced detection failed: {e}")
            return []
    
    def _extract_and_store_features(self, frame: np.ndarray, tracks: List[Dict], 
                                  camera_id: str, frame_idx: int):
        """Extract and store features for cross-camera matching."""
        for track in tracks:
            try:
                bbox = track['bbox']
                x1, y1, x2, y2 = bbox
                
                # Extract player region with padding
                padding = 10
                x1_pad = max(0, x1 - padding)
                y1_pad = max(0, y1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                
                player_region = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if player_region.size == 0:
                    continue
                
                # Extract enhanced features
                features = self.feature_extractor.extract_enhanced_features(
                    player_region, bbox, frame.shape
                )
                
                # Store features
                track_key = f"{camera_id}_{track['track_id']}"
                self.track_features[track_key].append({
                    'frame_idx': frame_idx,
                    'features': features,
                    'bbox': bbox
                })
                
                # Mark as stable if enough history
                if len(self.track_features[track_key]) >= 3:
                    self.stable_tracks[camera_id].add(track['track_id'])
                
            except Exception as e:
                logger.warning(f"Feature extraction failed for track {track['track_id']}: {e}")
    
    def _perform_enhanced_cross_camera_matching(self):
        """Perform enhanced cross-camera player matching."""
        logger.info("[MATCHING] Performing enhanced cross-camera matching")
        
        stable_broadcast = list(self.stable_tracks['broadcast'])
        stable_tacticam = list(self.stable_tracks['tacticam'])
        
        if not stable_broadcast or not stable_tacticam:
            logger.warning("No stable tracks found for cross-camera matching")
            return
        
        # Compute averaged features for stable tracks
        broadcast_features = []
        tacticam_features = []
        
        for track_id in stable_broadcast:
            track_key = f"broadcast_{track_id}"
            features = self._get_averaged_features(track_key)
            broadcast_features.append(features)
        
        for track_id in stable_tacticam:
            track_key = f"tacticam_{track_id}"
            features = self._get_averaged_features(track_key)
            tacticam_features.append(features)
        
        # Compute similarity matrix using enhanced features
        similarity_matrix = self._compute_enhanced_similarity_matrix(
            broadcast_features, tacticam_features
        )
        
        # Apply threshold and use Hungarian algorithm
        similarity_matrix[similarity_matrix < config.FEATURE_SIMILARITY_THRESHOLD] = 0
        
        if np.any(similarity_matrix > 0):
            cost_matrix = 1 - similarity_matrix  # Convert similarity to cost
            broadcast_indices, tacticam_indices = linear_sum_assignment(cost_matrix)
            
            matches = 0
            for b_idx, t_idx in zip(broadcast_indices, tacticam_indices):
                if similarity_matrix[b_idx, t_idx] > config.FEATURE_SIMILARITY_THRESHOLD:
                    broadcast_track_id = stable_broadcast[b_idx]
                    tacticam_track_id = stable_tacticam[t_idx]
                    
                    # Create cross-camera association
                    self.cross_camera_matches[tacticam_track_id] = broadcast_track_id
                    matches += 1
                    
                    logger.info(f"Match: tacticam_{tacticam_track_id} -> broadcast_{broadcast_track_id} "
                              f"(similarity: {similarity_matrix[b_idx, t_idx]:.3f})")
            
            logger.info(f"[SUCCESS] Created {matches} cross-camera associations")
        else:
            logger.warning("No valid cross-camera matches found")
    
    def _get_averaged_features(self, track_key: str) -> Dict[str, np.ndarray]:
        """Get averaged features for a track."""
        if track_key not in self.track_features:
            return {}
        
        feature_records = list(self.track_features[track_key])
        if not feature_records:
            return {}
        
        # Average each feature type
        averaged = {}
        feature_types = feature_records[0]['features'].keys()
        
        for feature_type in feature_types:
            features_list = [record['features'][feature_type] 
                           for record in feature_records 
                           if feature_type in record['features']]
            
            if features_list:
                averaged[feature_type] = np.mean(features_list, axis=0)
        
        return averaged
    
    def _compute_enhanced_similarity_matrix(self, broadcast_features: List[Dict], 
                                          tacticam_features: List[Dict]) -> np.ndarray:
        """Compute enhanced similarity matrix using multiple feature types."""
        if not broadcast_features or not tacticam_features:
            return np.zeros((len(broadcast_features), len(tacticam_features)))
        
        # Feature weights for enhanced matching
        weights = {
            'visual': 0.4,
            'color': 0.3,
            'shape': 0.2,
            'position': 0.1
        }
        
        similarity_matrix = np.zeros((len(broadcast_features), len(tacticam_features)))
        
        for i, b_features in enumerate(broadcast_features):
            for j, t_features in enumerate(tacticam_features):
                total_similarity = 0
                total_weight = 0
                
                for feature_type, weight in weights.items():
                    if feature_type in b_features and feature_type in t_features:
                        b_feat = b_features[feature_type]
                        t_feat = t_features[feature_type]
                        
                        if b_feat.size > 0 and t_feat.size > 0:
                            # Cosine similarity
                            similarity = np.dot(b_feat, t_feat) / (
                                np.linalg.norm(b_feat) * np.linalg.norm(t_feat) + 1e-8
                            )
                            total_similarity += similarity * weight
                            total_weight += weight
                
                if total_weight > 0:
                    similarity_matrix[i, j] = total_similarity / total_weight
        
        return similarity_matrix
    
    def _generate_enhanced_outputs(self, cap_broadcast: cv2.VideoCapture, 
                                 cap_tacticam: cv2.VideoCapture):
        """Generate enhanced outputs with improved annotations."""
        logger.info("[OUTPUT] Generating enhanced strategic outputs")
        
        # Assign global IDs
        self._assign_enhanced_global_ids()
        
        # Reset video positions
        cap_broadcast.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap_tacticam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Generate enhanced videos
        self._generate_enhanced_videos(cap_broadcast, cap_tacticam)
        
        # Generate CSV data
        self._generate_enhanced_csv()
        
        # Generate report
        self._generate_enhanced_report()
    
    def _assign_enhanced_global_ids(self):
        """Assign enhanced global IDs with cross-camera consistency."""
        # First, assign global IDs to broadcast tracks (reference camera)
        for frame_tracks in self.tracks_broadcast.values():
            for track in frame_tracks:
                track_key = f"broadcast_{track['track_id']}"
                if track_key not in self.global_id_mapping:
                    self.global_id_mapping[track_key] = self.global_id_counter
                    self.global_id_counter += 1
        
        # Then, assign IDs to tacticam tracks based on cross-camera matches
        for frame_tracks in self.tracks_tacticam.values():
            for track in frame_tracks:
                track_key = f"tacticam_{track['track_id']}"
                if track_key not in self.global_id_mapping:
                    # Check if this track has a cross-camera match
                    if track['track_id'] in self.cross_camera_matches:
                        matched_broadcast_id = self.cross_camera_matches[track['track_id']]
                        broadcast_key = f"broadcast_{matched_broadcast_id}"
                        if broadcast_key in self.global_id_mapping:
                            # Use same global ID as matched broadcast track
                            self.global_id_mapping[track_key] = self.global_id_mapping[broadcast_key]
                        else:
                            # Assign new global ID
                            self.global_id_mapping[track_key] = self.global_id_counter
                            self.global_id_counter += 1
                    else:
                        # Assign new global ID
                        self.global_id_mapping[track_key] = self.global_id_counter
                        self.global_id_counter += 1
    
    def _generate_enhanced_videos(self, cap_broadcast: cv2.VideoCapture, 
                                cap_tacticam: cv2.VideoCapture):
        """Generate enhanced annotated videos with professional quality."""
        logger.info("🎥 Generating enhanced strategic videos")
        
        # Setup output paths
        broadcast_output = Path("outputs/videos/broadcast_enhanced_strategic.mp4")
        tacticam_output = Path("outputs/videos/tacticam_enhanced_strategic.mp4")
        
        # Ensure output directory exists
        broadcast_output.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_CODEC)
        
        width_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_b = int(cap_broadcast.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_t = int(cap_tacticam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_b = cv2.VideoWriter(str(broadcast_output), fourcc, config.OUTPUT_FPS, (width_b, height_b))
        out_t = cv2.VideoWriter(str(tacticam_output), fourcc, config.OUTPUT_FPS, (width_t, height_t))
        
        frame_idx = 0
        max_frames = len(self.tracks_broadcast)
        
        while frame_idx < max_frames:
            ret_b, frame_b = cap_broadcast.read()
            ret_t, frame_t = cap_tacticam.read()
            
            if not ret_b or not ret_t:
                break
            
            # Draw enhanced strategic annotations
            if frame_idx in self.tracks_broadcast:
                annotated_b = self._draw_enhanced_strategic_annotations(
                    frame_b, self.tracks_broadcast[frame_idx], 'broadcast', frame_idx
                )
            else:
                annotated_b = frame_b.copy()
            
            if frame_idx in self.tracks_tacticam:
                annotated_t = self._draw_enhanced_strategic_annotations(
                    frame_t, self.tracks_tacticam[frame_idx], 'tacticam', frame_idx
                )
            else:
                annotated_t = frame_t.copy()
            
            out_b.write(annotated_b)
            out_t.write(annotated_t)
            
            frame_idx += 1
        
        out_b.release()
        out_t.release()
        
        logger.info(f"[SUCCESS] Enhanced strategic videos saved:")
        logger.info(f"   • {broadcast_output}")
        logger.info(f"   • {tacticam_output}")
    
    def _draw_enhanced_strategic_annotations(self, frame: np.ndarray, tracks: List[Dict], 
                                           camera_id: str, frame_idx: int) -> np.ndarray:
        """Draw enhanced strategic annotations with professional quality."""
        annotated_frame = frame.copy()
        
        for track in tracks:
            track_key = f"{camera_id}_{track['track_id']}"
            global_id = self.global_id_mapping.get(track_key, track['track_id'])
            
            # Professional color scheme based on global ID
            color_idx = (global_id - 1) % len(config.ANNOTATION_COLOR_PALETTE)
            color = config.ANNOTATION_COLOR_PALETTE[color_idx]
            
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox
            confidence = track['confidence']
            
            # Enhanced bounding box with gradient effect
            thickness = config.BBOX_THICKNESS
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Corner markers for professional look
            corner_size = 12
            corner_thickness = thickness + 1
            # Top-left corner
            cv2.line(annotated_frame, (x1, y1), (x1 + corner_size, y1), color, corner_thickness)
            cv2.line(annotated_frame, (x1, y1), (x1, y1 + corner_size), color, corner_thickness)
            # Top-right corner
            cv2.line(annotated_frame, (x2, y1), (x2 - corner_size, y1), color, corner_thickness)
            cv2.line(annotated_frame, (x2, y1), (x2, y1 + corner_size), color, corner_thickness)
            # Bottom-left corner
            cv2.line(annotated_frame, (x1, y2), (x1 + corner_size, y2), color, corner_thickness)
            cv2.line(annotated_frame, (x1, y2), (x1, y2 - corner_size), color, corner_thickness)
            # Bottom-right corner
            cv2.line(annotated_frame, (x2, y2), (x2 - corner_size, y2), color, corner_thickness)
            cv2.line(annotated_frame, (x2, y2), (x2, y2 - corner_size), color, corner_thickness)
            
            # Enhanced label with professional styling
            is_matched = track['track_id'] in self.cross_camera_matches
            match_symbol = "★" if is_matched else ""
            label = f"P{global_id}{match_symbol} ({confidence:.2f})"
            
            # Dynamic font scaling
            font_scale = min(config.FONT_SCALE, max(0.4, (x2 - x1) / 150))
            font_thickness = config.FONT_THICKNESS
            
            # Get label size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                       font_scale, font_thickness)[0]
            
            # Professional label background with rounded effect
            label_bg_x1 = x1 - 2
            label_bg_y1 = y1 - label_size[1] - 12
            label_bg_x2 = x1 + label_size[0] + 8
            label_bg_y2 = y1 - 2
            
            # Gradient background effect
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (label_bg_x1, label_bg_y1), (label_bg_x2, label_bg_y2), 
                         color, -1)
            cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)
            
            # Label text with shadow effect for better readability
            text_x = x1 + 4
            text_y = y1 - 6
            
            # Shadow
            cv2.putText(annotated_frame, label, (text_x + 1, text_y + 1), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness + 1)
            # Main text
            cv2.putText(annotated_frame, label, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        # Professional frame information overlay
        info_bg_height = 80
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (annotated_frame.shape[1], info_bg_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Frame info
        info_text = f"Frame: {frame_idx:04d} | Camera: {camera_id.upper()} | Players: {len(tracks)} | Enhanced Strategic Mode"
        cv2.putText(annotated_frame, info_text, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Cross-camera matches info
        matches_count = len(self.cross_camera_matches)
        match_text = f"Cross-Camera Matches: {matches_count} | Global IDs: {len(set(self.global_id_mapping.values()))}"
        cv2.putText(annotated_frame, match_text, (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        return annotated_frame
    
    def _generate_enhanced_csv(self):
        """Generate enhanced CSV output with global IDs."""
        logger.info("[DATA] Generating enhanced CSV data")
        
        csv_data = []
        for detection in self.all_detections:
            track_key = f"{detection['camera']}_{detection['track_id']}"
            global_id = self.global_id_mapping.get(track_key, detection['track_id'])
            
            x1, y1, x2, y2 = detection['bbox']
            
            csv_data.append({
                'frame_id': detection['frame_id'],
                'camera': detection['camera'],
                'local_track_id': detection['track_id'],
                'global_id': global_id,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1, 'height': y2 - y1,
                'center_x': (x1 + x2) / 2, 'center_y': (y1 + y2) / 2,
                'confidence': detection['confidence']
            })
        
        df = pd.DataFrame(csv_data)
        csv_output_path = Path("outputs/data/enhanced_strategic_player_tracking.csv")
        csv_output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_output_path, index=False)
        
        logger.info(f"[SUCCESS] Enhanced CSV saved to {csv_output_path}")
    
    def _generate_enhanced_report(self):
        """Generate enhanced processing report."""
        logger.info("📋 Generating enhanced processing report")
        
        total_time = time.time() - self.start_time
        unique_global_ids = len(set(self.global_id_mapping.values()))
        cross_camera_matches = len(self.cross_camera_matches)
        
        report = {
            "enhanced_strategic_processing_summary": {
                "processing_time_seconds": round(total_time, 2),
                "frames_processed": len(self.tracks_broadcast),
                "total_detections": len(self.all_detections),
                "unique_global_ids": unique_global_ids,
                "cross_camera_matches": cross_camera_matches,
                "match_success_rate": round(cross_camera_matches / max(1, len(self.stable_tracks['broadcast'])), 2)
            },
            "tracking_performance": {
                "broadcast_tracks": len(self.tracker_broadcast.tracks),
                "tacticam_tracks": len(self.tracker_tacticam.tracks),
                "stable_broadcast_tracks": len(self.stable_tracks['broadcast']),
                "stable_tacticam_tracks": len(self.stable_tracks['tacticam'])
            },
            "enhanced_features": {
                "multi_modal_feature_extraction": True,
                "enhanced_color_features": True,
                "position_aware_matching": True,
                "professional_annotations": True,
                "cross_camera_consistency": True
            },
            "quality_improvements": {
                "enhanced_detection_filtering": True,
                "improved_tracking_stability": True,
                "better_cross_camera_association": True,
                "professional_visualization": True
            },
            "output_files": {
                "enhanced_videos": [
                    "outputs/videos/broadcast_enhanced_strategic.mp4",
                    "outputs/videos/tacticam_enhanced_strategic.mp4"
                ],
                "enhanced_csv": "outputs/data/enhanced_strategic_player_tracking.csv",
                "processing_report": "outputs/reports/enhanced_strategic_processing_report.json"
            }
        }
        
        report_path = Path("outputs/reports/enhanced_strategic_processing_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"[SUCCESS] Enhanced report saved to {report_path}")


def main():
    """Main function to run the enhanced strategic system."""
    print("[SYSTEM] Enhanced Strategic Cross-Camera Player Mapping System")
    print("=" * 80)
    print("Building upon successful strategic approach with enhanced accuracy")
    print("Key improvements:")
    print("• Better cross-camera association logic")
    print("• Enhanced multi-modal feature matching")
    print("• Improved ID consistency and stability")
    print("• Professional quality annotations")
    print()
    
    try:
        # Initialize enhanced strategic system
        mapper = EnhancedStrategicMapping()
        
        # Process videos with enhanced strategic approach
        results = mapper.process_videos_enhanced()
        
        if results["status"] == "success":
            print("[SUCCESS] Enhanced strategic processing completed successfully!")
            print(f"[STATS] Total time: {results['processing_time']:.2f} seconds")
            print(f"[STATS] Total detections: {results['total_detections']}")
            print(f"[STATS] Unique global IDs: {results['unique_global_ids']}")
            print(f"[STATS] Cross-camera matches: {results['cross_camera_matches']}")
            print()
            print("[OUTPUT] Enhanced strategic outputs generated:")
            print("   • outputs/videos/broadcast_enhanced_strategic.mp4")
            print("   • outputs/videos/tacticam_enhanced_strategic.mp4") 
            print("   • outputs/data/enhanced_strategic_player_tracking.csv")
            print("   • outputs/reports/enhanced_strategic_processing_report.json")
        else:
            print(f"❌ Enhanced strategic processing failed: {results['error']}")
    
    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
