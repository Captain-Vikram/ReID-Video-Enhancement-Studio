"""
Configuration Parameters for Enhanced Cross-Camera Player Mapping System
Professional settings optimized for reliability and accuracy.
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Video paths
BROADCAST_VIDEO = DATA_DIR / "broadcast.mp4"
TACTICAM_VIDEO = DATA_DIR / "tacticam.mp4"

# Output directories
ANNOTATED_VIDEOS_DIR = OUTPUT_DIR / "videos"
DATA_OUTPUT_DIR = OUTPUT_DIR / "data"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Camera IDs
BROADCAST_CAMERA_ID = "broadcast"
TACTICAM_CAMERA_ID = "tacticam"

# =============================================================================
# DETECTION PARAMETERS
# =============================================================================

# YOLOv11 Detection Settings
DETECTION_CONFIDENCE = 0.6      # Higher confidence for reliability
NMS_THRESHOLD = 0.4             # Non-maximum suppression threshold
MODEL_PATH = MODELS_DIR / "best.pt"  # Path to YOLOv11 model

# Detection Filtering
MIN_DETECTION_AREA = 1000       # Minimum bounding box area (pixels²)
MAX_DETECTION_AREA = 50000      # Maximum bounding box area (pixels²)
PERSON_CLASS_ID = 2             # Class ID for 'player' in YOLO model

# =============================================================================
# TRACKING PARAMETERS
# =============================================================================

# Enhanced Tracking Settings
MAX_DISAPPEARED = 15            # Frames before track deletion
MIN_HITS = 3                   # Frames before track confirmation  
IOU_THRESHOLD = 0.3            # Intersection over Union threshold
MAX_DISTANCE = 0.2             # Maximum feature distance for association

# Track Management
TRACK_BUFFER_SIZE = 100        # Maximum tracks to maintain
MIN_TRACK_LENGTH = 10          # Minimum track length for inclusion in results

# =============================================================================
# RE-IDENTIFICATION PARAMETERS
# =============================================================================

# Feature Extraction
FEATURE_SIMILARITY_THRESHOLD = 0.4    # Cross-camera matching threshold
VISUAL_FEATURE_DIM = 2048             # ResNet50 feature dimension
TEMPORAL_WINDOW = 15                  # Frames for feature averaging

# Enhanced Feature Weights for Multi-Modal Matching
VISUAL_WEIGHT = 0.4
COLOR_WEIGHT = 0.3
SHAPE_WEIGHT = 0.2
POSITION_WEIGHT = 0.1

# Cross-Camera Matching
MATCHING_ALGORITHM = "hungarian"       # Optimal assignment algorithm
MAX_MATCHING_DISTANCE = 0.5           # Maximum distance for valid matches

# =============================================================================
# OUTPUT PARAMETERS
# =============================================================================

# Video Output Settings
OUTPUT_FPS = 25                # Output video frame rate
OUTPUT_CODEC = "mp4v"         # Video codec for output
OUTPUT_QUALITY = 90           # Video quality (0-100)

# Annotation Style
BBOX_THICKNESS = 3            # Bounding box line thickness
FONT_SCALE = 0.8             # Text font scale
FONT_THICKNESS = 2           # Text line thickness
TEXT_BACKGROUND_ALPHA = 0.7  # Text background transparency

# Professional Color Palette for Player IDs
ANNOTATION_COLOR_PALETTE = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green  
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Light Green
    (255, 128, 128),  # Light Red
]

# =============================================================================
# SYSTEM SETTINGS
# =============================================================================

# Processing
MAX_FRAMES = None              # Maximum frames to process (for demo)
PROCESS_EVERY_N_FRAMES = 1  # Process every frame for maximum accuracy
BATCH_SIZE = 1               # Processing batch size
USE_GPU = True               # Use GPU acceleration if available

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

# Performance
MEMORY_LIMIT = 8             # Maximum memory usage in GB
THREAD_COUNT = 4             # Number of processing threads

# =============================================================================
# INITIALIZATION
# =============================================================================

# Ensure output directories exist
for directory in [OUTPUT_DIR, ANNOTATED_VIDEOS_DIR, DATA_OUTPUT_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
