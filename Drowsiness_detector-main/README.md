# Enhanced Drowsiness Detection System

A real-time drowsiness detection system with advanced features for driver safety monitoring.

## Features

### Core Detection
- **Eye Aspect Ratio (EAR)**: Accurate eye closure detection using facial landmark analysis
- **Mouth Aspect Ratio (MAR)**: Yawn detection for fatigue monitoring
- **Multi-level Alerts**: Graduated warning system (warning → alert)
- **Facial Landmark Detection**: Support for both dlib (high accuracy) and OpenCV (fallback)

### Advanced Features
- Real-time statistics tracking (blinks, yawns, drowsy events)
- Session logging with JSON export
- Smoothed detection using rolling buffers
- FPS optimization
- Histogram equalization for varying lighting conditions

## Installation

### 1. Install Python Dependencies

```bash
pip install opencv-python numpy pygame scipy
```

### 2. Install dlib (Optional but Recommended)

For highest accuracy, install dlib with facial landmarks:

**Windows:**
```bash
pip install cmake
pip install dlib
```

**Linux/Mac:**
```bash
pip install dlib
```

### 3. Download Facial Landmark Model (for dlib)

Download `shape_predictor_68_face_landmarks.dat` from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Extract and place in the project directory.

**Note:** The system will work without dlib using OpenCV Haar Cascades, but with lower accuracy.

## Usage

### Run Enhanced Version
```bash
python drowsiness_detector_enhanced.py
```

### Run Original Version
```bash
python drowsiness_detector.py
```

### Controls
- **q**: Quit the application
- **r**: Reset counters
- **s**: Save session log

## Configuration

Edit `Config` class in `drowsiness_detector_enhanced.py`:

```python
EAR_THRESHOLD = 0.25           # Eye closure threshold
EAR_CONSEC_FRAMES = 20         # Frames before alert
MAR_THRESHOLD = 0.6            # Yawn detection threshold
```

## How It Works

### Eye Aspect Ratio (EAR)
EAR measures eye openness using 6 facial landmarks per eye:
- EAR > 0.25: Eyes open
- EAR < 0.25: Eyes closed

Formula: `EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)`

### Alert Levels
1. **Normal** (Green): EAR above threshold
2. **Warning** (Orange): Eyes closed 12-19 frames
3. **Alert** (Red): Eyes closed 20+ frames

### Yawn Detection
Detects yawning using mouth aspect ratio (MAR):
- MAR > 0.6: Yawning detected

## Output

### On-Screen Display
- Real-time EAR and MAR values
- Frame counter
- Blink and yawn statistics
- FPS and runtime
- Visual facial landmarks

### Session Logs
Saved to `drowsiness_log.json`:
- Drowsy events with timestamps
- Yawn events
- Session statistics
- Total frames analyzed

## Performance

- **FPS**: 25-30 fps (depends on hardware)
- **Latency**: < 50ms per frame
- **Detection Method**: dlib (high accuracy) or OpenCV (fast)

## Improvements Over Original

1. ✅ **Accurate EAR-based detection** vs simple threshold
2. ✅ **Yawn detection** for additional fatigue indicators
3. ✅ **Graduated alerts** vs binary alarm
4. ✅ **Smoothed detection** reduces false positives
5. ✅ **Session logging** for analysis
6. ✅ **Better visualization** with landmark overlay
7. ✅ **Statistics tracking** (blinks, yawns)
8. ✅ **Adaptive lighting** with histogram equalization

## Troubleshooting

### Camera Not Opening
- Check camera permissions
- Try different camera index: `cv2.VideoCapture(1)`

### Dlib Installation Fails
- Use OpenCV fallback (automatic)
- The system works without dlib

### Low FPS
- Reduce resolution in Config
- Use OpenCV method instead of dlib
- Close other applications

### False Positives
- Adjust EAR_THRESHOLD (increase for less sensitive)
- Increase EAR_CONSEC_FRAMES

## Requirements

- Python 3.7+
- Webcam
- 4GB RAM minimum
- Windows/Linux/MacOS

## License

MIT License

## Author

Enhanced drowsiness detection system
