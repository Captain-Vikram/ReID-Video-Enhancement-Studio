# 🚀 Quick Start Guide

## Instant Launch Options

### 🎬 GUI Application (Recommended)
```bash
# Option 1: Smart Launcher (Auto-installs dependencies + generates sample data)
python launch_gui.py

# Option 2: Windows One-Click
start_gui.bat  # Double-click this file

# Option 3: Direct Streamlit
pip install -r requirements.txt
streamlit run app.py
```

### 📋 Command Line System
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete tracking system
python main.py

# Generate enhanced videos
python render_enhanced_videos.py
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **GPU**: CUDA-compatible (optional, for faster processing)
- **Browser**: Any modern browser for GUI

## 🧪 Testing with Sample Data

### Generate Test Files
```bash
# Creates realistic sample data for testing
python create_sample_data.py
```

**Generated Files:**
- `sample_data/sample_annotated_video.mp4` (1920x1080, 10s)
- `sample_data/sample_tracking_data.csv` (2,954 records, 300 frames)
- `sample_data/sample_tracking_data.json` (alternative format)

### Test the GUI
1. Run `python launch_gui.py`
2. Upload `sample_annotated_video.mp4` as video file
3. Upload `sample_tracking_data.csv` as tracking data
4. Click "🚀 Run Enhancement" and test all features

## 📊 Expected Results

### GUI Application
- Opens in browser at `http://localhost:8501`
- Real-time processing with progress tracking
- Interactive video comparison and analytics
- Download enhanced videos and quality reports

### Command Line System
- Generates annotated videos in `outputs/videos/`
- Creates tracking data CSV in `outputs/data/`
- Produces processing reports in `outputs/reports/`

## ⚠️ Troubleshooting

### Common Issues
```bash
# Missing dependencies
pip install streamlit plotly opencv-python pandas numpy

# Python version check
python --version  # Should be 3.8+

# GPU availability (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### File Upload Issues
- Ensure video files are under 200MB
- Use MP4 format for best compatibility
- Check CSV columns match required format

### Performance Issues
- Close other applications to free memory
- Use shorter video clips for testing
- Reduce video resolution if processing is slow

---

🔗 **For detailed documentation**: See [README.md](README.md) and [GUI_README.md](GUI_README.md)
