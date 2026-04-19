# Real-Time Drowsiness Detection System

A Python-based drowsiness detection system that uses computer vision and machine learning to detect signs of drowsiness in real-time from a webcam feed. The system monitors Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) to detect closed eyes and yawning.

## Features

- **Real-time face detection** using dlib's frontal face detector
- **68-point facial landmark detection** for precise eye and mouth tracking
- **Eye Aspect Ratio (EAR) calculation** to detect eye closure
- **Mouth Aspect Ratio (MAR) calculation** to detect yawning
- **Configurable thresholds** for fine-tuning sensitivity
- **Visual alerts** with on-screen notifications
- **Facial landmark visualization** for debugging and monitoring

## How It Works

### Eye Aspect Ratio (EAR)
The system calculates the Eye Aspect Ratio using the formula:

```
EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
```

- When eyes are open: EAR ≈ 0.25-0.30
- When eyes are closed: EAR ≈ 0.0-0.15
- Alert triggers if EAR stays below threshold for 20+ consecutive frames

### Mouth Aspect Ratio (MAR)
The system calculates the Mouth Aspect Ratio using the formula:

```
MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
```

- Normal state: MAR ≈ 0.3-0.5
- Yawning: MAR > 0.75
- Alert triggers immediately when MAR exceeds threshold

## Prerequisites

- Python 3.7 or higher
- Webcam
- Windows/Linux/macOS

## Installation

### Step 1: Clone or Download the Repository

```bash
cd dd
```

### Step 2: Install Python Dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

**Note for Windows users:** If you encounter issues installing dlib, you may need to:
1. Install Visual Studio Build Tools
2. Or use a pre-built wheel: `pip install dlib-19.24.2-cp310-cp310-win_amd64.whl`
3. Or use conda: `conda install -c conda-forge dlib`

### Step 3: Download the Facial Landmark Model

The system requires the dlib 68-point facial landmark predictor model.

#### Option 1: Direct Download
1. Download the compressed file:
   ```
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   ```

2. Extract the `.bz2` file to get `shape_predictor_68_face_landmarks.dat` (~99MB)

3. Create a `models` directory in the project folder:
   ```bash
   mkdir models
   ```

4. Place the extracted `shape_predictor_68_face_landmarks.dat` file in the `models/` directory

#### Option 2: Using Command Line

**On Linux/macOS:**
```bash
mkdir -p models
cd models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
cd ..
```

**On Windows (PowerShell):**
```powershell
mkdir models -Force
cd models
Invoke-WebRequest -Uri "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" -OutFile "shape_predictor_68_face_landmarks.dat.bz2"
# Extract using 7-Zip or WinRAR, or install bzip2 via chocolatey
cd ..
```

### Final Directory Structure

Your project should look like this:

```
dd/
├── drowsiness_detector.py
├── requirements.txt
├── README.md
└── models/
    └── shape_predictor_68_face_landmarks.dat
```

## Usage

### Running the System

Simply run the main Python script:

```bash
python drowsiness_detector.py
```

### Controls

- **Press 'q'** to quit the application

### On-Screen Display

The system displays:
- **EAR (Eye Aspect Ratio)**: Current eye openness value
- **MAR (Mouth Aspect Ratio)**: Current mouth openness value
- **Status**: Shows "ACTIVE" when monitoring, "DROWSINESS ALERT" when drowsiness is detected
- **Green landmarks**: Facial feature points on the face
- **Red border**: Appears when drowsiness is detected

## Configuration

You can adjust the detection sensitivity by modifying the constants at the top of `drowsiness_detector.py`:

```python
# Eye Aspect Ratio threshold - decrease for more sensitive eye closure detection
EAR_THRESHOLD = 0.25

# Number of consecutive frames before triggering alert - decrease for faster alerts
EAR_CONSEC_FRAMES = 20

# Mouth Aspect Ratio threshold - decrease for more sensitive yawn detection
MAR_THRESHOLD = 0.75
```

### Recommended Settings

| Sensitivity | EAR_THRESHOLD | EAR_CONSEC_FRAMES | MAR_THRESHOLD |
|-------------|---------------|-------------------|---------------|
| Low         | 0.20          | 30                | 0.80          |
| **Medium (Default)** | **0.25** | **20** | **0.75** |
| High        | 0.28          | 15                | 0.70          |

## Troubleshooting

### Webcam Not Working
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index in the code: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

### Model File Not Found
- Verify that `shape_predictor_68_face_landmarks.dat` is in the `models/` directory
- Check that the file is fully extracted (not still .bz2)

### No Face Detected
- Ensure good lighting conditions
- Position your face clearly in front of the camera
- Try adjusting the camera angle

### False Positives (Too Many Alerts)
- Increase `EAR_THRESHOLD` (try 0.27 or 0.28)
- Increase `EAR_CONSEC_FRAMES` (try 25 or 30)
- Increase `MAR_THRESHOLD` (try 0.80)

### False Negatives (Missing Drowsiness)
- Decrease `EAR_THRESHOLD` (try 0.23 or 0.22)
- Decrease `EAR_CONSEC_FRAMES` (try 15)
- Decrease `MAR_THRESHOLD` (try 0.70)

## Technical Details

### Dependencies
- **OpenCV**: Video capture and image processing
- **dlib**: Face detection and facial landmark detection
- **NumPy**: Numerical operations
- **SciPy**: Distance calculations for EAR and MAR
- **imutils**: Convenience functions for facial landmarks

### Algorithm Flow
1. Capture frame from webcam
2. Convert to grayscale
3. Detect faces using HOG-based detector
4. Predict 68 facial landmarks
5. Extract eye and mouth landmarks
6. Calculate EAR and MAR
7. Check thresholds and trigger alerts
8. Display annotated frame

### Facial Landmark Indices
- **Right eye**: Landmarks 36-41 (6 points)
- **Left eye**: Landmarks 42-47 (6 points)
- **Mouth**: Landmarks 48-67 (20 points)

## Performance

- **Frame rate**: ~20-30 FPS on modern hardware
- **Detection latency**: ~0.67 seconds (20 frames at 30 FPS)
- **CPU usage**: Moderate (10-30% on modern processors)

## Limitations

- Requires good lighting conditions
- May not work well with glasses or partial face occlusion
- Performance depends on CPU speed (no GPU acceleration)
- Single face detection at a time

## Future Enhancements

- Multi-face detection support
- Audio alert system
- Logging and statistics
- GPU acceleration
- Mobile deployment

## License

This project is provided as-is for educational and personal use.

## Credits

- **dlib**: Davis King - http://dlib.net
- **EAR/MAR algorithms**: Based on research by Soukupová and Čech (2016)

## References

- Soukupová, T., & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. In CVWW.
- King, D. E. (2009). Dlib-ml: A Machine Learning Toolkit. JMLR.
