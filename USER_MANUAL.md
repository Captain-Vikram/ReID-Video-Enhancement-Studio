# 📖 ReID Video Enhancement Studio - User Instructions Manual

## 🎯 Table of Contents

1. [Quick Start Guide](#-quick-start-guide)
2. [Installation & Setup](#-installation--setup)
3. [Project Structure](#-project-structure)
4. [GUI Application Guide](#-gui-application-guide)
5. [Command Line Usage](#-command-line-usage)
6. [Video Processing Workflow](#-video-processing-workflow)
7. [Troubleshooting](#-troubleshooting)
8. [Advanced Features](#-advanced-features)
9. [Output Files Guide](#-output-files-guide)
10. [FAQ](#-frequently-asked-questions)

---

## � Quick Start Guide

### Option 1: One-Click Launch (Windows)
1. **Double-click** `start_gui.bat` in the project root
2. The application will automatically check dependencies and launch
3. Open your web browser to the provided URL (usually `http://localhost:8501`)

### Option 2: Python Launch
```bash
# Navigate to project directory
cd "path/to/FINAL_SUBMISSION"

# Launch the application
python launch_gui.py
```

### Option 3: Advanced Launch
```bash
# Navigate to GUI directory
cd gui

# Launch directly with Streamlit
streamlit run app.py
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8 or later**
- **8GB+ RAM** (16GB recommended for large videos)
- **GPU support** (optional but recommended for faster processing)

### Automatic Installation
The application will automatically install missing dependencies when launched. If you prefer manual installation:

```bash
# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "import streamlit, cv2, torch; print('All dependencies installed!')"
```

### Required Files
Ensure these files are present:
- `models/best.pt` - YOLO detection model
- `src/enhanced_strategic_mapping.py` - Cross-camera mapping module
- `src/enhanced_video_renderer.py` - Video enhancement module

---

## 📁 Project Structure

```
FINAL_SUBMISSION/
├── 🎬 gui/                    # GUI Application Files
│   ├── app.py                 # Main Streamlit application
│   ├── launch_gui.py          # GUI launcher with dependency check
│   ├── start_gui.bat          # Windows batch launcher
│   └── README.md              # GUI-specific documentation
│
├── 🧠 src/                    # Core Processing Modules
│   ├── enhanced_strategic_mapping.py    # Cross-camera player mapping
│   ├── enhanced_video_renderer.py       # Professional video enhancement
│   └── config.py              # Configuration settings
│
├── 🎯 models/                 # AI Models
│   └── best.pt                # YOLOv11 player detection model
│
├── 📹 data/                   # Input Videos
│   ├── broadcast.mp4          # Broadcast camera footage
│   └── tacticam.mp4           # Tactical camera footage
│
├── 📊 outputs/                # Generated Results
│   ├── videos/                # Basic enhanced videos
│   ├── enhanced_videos/       # Professional quality videos
│   ├── data/                  # Player tracking data (CSV)
│   └── reports/               # Quality reports (JSON)
│
├── 🚀 Launch Files (Root)
│   ├── launch_gui.py          # Main Python launcher
│   ├── start_gui.bat          # Main Windows launcher
│   └── create_demo_gifs.py    # GIF generator for demos
│
└── 📖 Documentation
    ├── README.md              # Main project documentation
    ├── requirements.txt       # Python dependencies
    └── docs/                  # Additional documentation
```

---

## 🎬 GUI Application Guide

### Step 1: Upload Videos
1. **Navigate to Upload Tab**
2. **Upload Broadcast Camera Video**:
   - Click "Upload broadcast camera video"
   - Select your wide-angle broadcast footage (MP4/AVI/MOV)
   - Verify file size and format in the success message

3. **Upload Tactical Camera Video**:
   - Click "Upload tactical camera video" 
   - Select your close-up tactical footage (MP4/AVI/MOV)
   - Ensure videos are roughly synchronized

### Step 2: Configure Processing
1. **Navigate to Process Tab**
2. **Check System Status**:
   - ✅ Green checkmarks = Full AI functionality available
   - ❌ Red X marks = Component missing (will use fallback mode)

3. **Processing Options** (in Sidebar):
   - **Enhancement Method**: Choose `kalman` (recommended) or `velocity`
   - **Temporal Smoothing**: Enable for stable bounding boxes
   - **Motion Prediction**: Enable for handling occlusions
   - **Confidence Stabilization**: Enable to reduce flickering
   - **Video Quality**: Set output quality (1-100, recommend 95)

### Step 3: Run Enhancement
1. **Click "Run Enhancement"** button
2. **Monitor Progress**:
   - Real-time progress bar with detailed status messages
   - Processing typically takes 1-5 minutes depending on video length
   - Watch for different processing phases:
     - Setting up environment
     - Cross-camera mapping
     - Video enhancement
     - Quality report generation

### Step 4: Review Results
1. **Navigate to Results Tab**
2. **Check Processing Mode**:
   - **Full AI Backend**: Complete cross-camera mapping succeeded
   - **Fallback Mode**: Basic processing due to missing components

3. **Review Enhancement Details**:
   - Active features and their status
   - Cross-camera mapping statistics
   - Processing performance metrics

4. **Analyze Quality Metrics**:
   - Confidence distribution charts
   - Detection statistics
   - Enhancement effectiveness scores

### Step 5: Download Results
1. **Navigate to Download Tab**
2. **Video Comparison**:
   - **Enhanced Only**: View final results
   - **Side-by-Side**: Compare original vs enhanced
   - **Original Only**: Review source material

3. **Download Files**:
   - **Enhanced Videos**: Professional annotated output
   - **Quality Reports**: Detailed processing statistics (JSON)
   - **Tracking Data**: Player movement data (CSV)

---

## 💻 Command Line Usage

### Basic Processing
```bash
# Run complete cross-camera mapping pipeline
python main.py

# Generate enhanced professional videos
python render_enhanced_videos.py

# Run complete demo with sample data
python demo_complete.py
```

### Advanced Options
```bash
# Process specific video files
python main.py --broadcast data/my_broadcast.mp4 --tactical data/my_tactical.mp4

# Generate GIFs for demonstration
python create_demo_gifs.py

# View output videos
python view_videos.py
```

---

## 🎥 Video Processing Workflow

### Input Requirements
- **Two synchronized video files** (broadcast + tactical camera)
- **Supported formats**: MP4, AVI, MOV
- **Recommended**: 720p or higher resolution
- **Duration**: 30 seconds to 10 minutes (longer videos supported)

### Processing Pipeline
1. **Video Upload & Validation**
   - Format verification
   - Resolution analysis
   - Duration compatibility check

2. **Player Detection**
   - YOLOv11 model detects players in both videos
   - Bounding box extraction
   - Confidence score assignment

3. **Tracking Within Camera**
   - Assigns consistent IDs within each video
   - Temporal smoothing of detections
   - Motion prediction for occlusions

4. **Cross-Camera Mapping**
   - Feature extraction (visual, positional, color)
   - Player correspondence matching
   - Global ID assignment

5. **Enhancement Application**
   - 7-frame temporal smoothing
   - Kalman filter motion prediction
   - Confidence score stabilization
   - Professional visual styling

6. **Output Generation**
   - Enhanced video rendering
   - Quality report compilation
   - Statistics calculation

### Expected Processing Times
| Video Duration | Processing Time | Output Size |
|---------------|----------------|-------------|
| 30 seconds    | 1-2 minutes    | ~50MB      |
| 2 minutes     | 3-5 minutes    | ~200MB     |
| 5 minutes     | 8-12 minutes   | ~500MB     |
| 10 minutes    | 15-25 minutes  | ~1GB       |

---

## � Troubleshooting

### Common Issues

#### "YOLO Model Not Found"
**Solution:**
```bash
# Verify model exists
ls models/best.pt

# If missing, contact support for model file
```

#### "Import Error: Enhanced Strategic Mapping"
**Solution:**
```bash
# Verify source files exist
ls src/enhanced_strategic_mapping.py
ls src/enhanced_video_renderer.py

# Check Python path
python -c "import sys; print(sys.path)"
```

#### "Streamlit Won't Start"
**Solution:**
```bash
# Install/reinstall Streamlit
pip uninstall streamlit
pip install streamlit

# Try alternative port
streamlit run app.py --server.port 8502
```

#### "Videos Won't Upload"
**Solutions:**
1. **Check file size**: Videos >100MB may take time to upload
2. **Verify format**: Only MP4, AVI, MOV supported
3. **Check permissions**: Ensure read access to video files
4. **Try smaller video**: Test with shorter clips first

#### "Processing Fails Immediately"
**Diagnostic Steps:**
1. Check system status in Process tab
2. Verify all components show green checkmarks
3. Review error messages in browser console
4. Try with sample videos first

### System Status Indicators

| Component | Status | Meaning |
|-----------|--------|---------|
| ✅ YOLO Model | Ready | AI detection available |
| ✅ Enhancer | Ready | Video enhancement available |
| ✅ Strategic Mapping | Ready | Cross-camera mapping available |
| ✅ PyTorch | Ready | GPU acceleration available |
| ✅ Ultralytics | Ready | YOLO framework available |
| ❌ Any Component | Missing | Will use fallback mode |

### Performance Optimization

#### For Faster Processing:
1. **Use GPU**: Install CUDA-compatible PyTorch
2. **Reduce video resolution**: Process at 720p then upscale
3. **Shorter clips**: Process in segments for very long videos
4. **Close other applications**: Free up RAM and CPU

#### For Better Quality:
1. **Higher quality settings**: Set video quality to 95-100
2. **Enable all enhancements**: Use all smoothing options
3. **Good source material**: Well-lit, clear videos work best
4. **Proper synchronization**: Ensure videos are time-aligned

---

## 🚀 Advanced Features

### Custom Configuration
Edit `src/config.py` to modify:
- Detection confidence thresholds
- Tracking smoothing parameters
- Cross-camera matching weights
- Output video settings

### Batch Processing
```bash
# Process multiple video pairs
python batch_process.py --input_dir videos/ --output_dir results/
```

### API Integration
```python
# Use programmatically
from gui.app import ReIDVideoEnhancer

enhancer = ReIDVideoEnhancer()
results = enhancer.enhance_video(video1_path, video2_path)
```

### Custom Models
Replace `models/best.pt` with your own trained YOLO model for specialized detection.

---

## 📊 Output Files Guide

### Enhanced Videos
- **Location**: `outputs/enhanced_videos/`
- **Naming**: `{camera}_enhanced_professional.mp4`
- **Content**: Professional broadcast-quality with annotations
- **Features**: Smooth tracking, stable IDs, clean styling

### Strategic Videos
- **Location**: `outputs/videos/`
- **Naming**: `{camera}_enhanced_strategic.mp4`
- **Content**: Cross-camera mapped with global IDs
- **Features**: Consistent player IDs across both cameras

### Data Files
- **Location**: `outputs/data/`
- **Format**: CSV with columns: frame_id, track_id, x1, y1, x2, y2, confidence
- **Usage**: Can be imported into other analysis tools

### Quality Reports
- **Location**: `outputs/reports/`
- **Format**: JSON with processing statistics
- **Content**: 
  - Processing time and performance
  - Detection statistics
  - Quality metrics
  - Enhancement effectiveness

---

## ❓ Frequently Asked Questions

### Q: What video formats are supported?
**A:** MP4, AVI, and MOV formats are fully supported. MP4 is recommended for best compatibility.

### Q: How long should my videos be?
**A:** Optimal length is 1-5 minutes. Shorter videos may have insufficient data for cross-camera mapping. Longer videos (>10 minutes) require more processing time and memory.

### Q: Do videos need to be exactly synchronized?
**A:** Videos should be roughly synchronized (within 1-2 seconds). The system includes synchronization algorithms to handle minor timing differences.

### Q: Can I process only one video?
**A:** The system is designed for cross-camera mapping and requires two videos. For single-video enhancement, use the fallback mode or other video processing tools.

### Q: What if the system shows "Fallback Mode"?
**A:** Fallback mode means AI components are unavailable, so the system performs basic video copying with minimal enhancement. Check system status and install missing dependencies.

### Q: How accurate is the cross-camera mapping?
**A:** Accuracy depends on video quality, lighting, and player distinctiveness. Typical accuracy is 85-95% for well-recorded football footage.

### Q: Can I use this for other sports?
**A:** The system is optimized for football but may work for other team sports. You may need to retrain the YOLO model for optimal results.

### Q: What are the hardware requirements?
**A:** Minimum: 8GB RAM, modern CPU. Recommended: 16GB+ RAM, dedicated GPU for faster processing.

### Q: How do I report issues or get support?
**A:** Check the troubleshooting section first. For additional support, review the logs in the terminal/console where you launched the application.

### Q: Can I modify the processing parameters?
**A:** Yes, many parameters can be adjusted in the GUI sidebar or by editing `src/config.py` for advanced customization.

---

## 📞 Support & Resources

- **Documentation**: Check `docs/` folder for detailed technical documentation
- **Sample Data**: Use provided sample videos in `data/` for testing
- **Logs**: Check terminal output for detailed processing information
- **Configuration**: Modify `src/config.py` for advanced settings

---

*📝 Last Updated: June 2025*
*🔄 Version: 2.0 - GUI Reorganized Edition*
