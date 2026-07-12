# SmartLens: Advanced Spatial Analytics & Vehicle Tracking

SmartLens is a high-performance computer vision application built on top of YOLOv11 and PySide6. It goes beyond traditional object detection by providing an interactive spatial analytics canvas, allowing users to measure speeds, classify vehicle colors using AI clustering, and define custom tracking zones dynamically.

## Key Features

### 1. Interactive Spatial Analytics (Custom Segmentation)
Users can perform freehand drawing directly on the video stream to define custom polygonal regions (e.g., sidewalks, no-parking zones, or specific lanes).
- **Point-in-Polygon Testing:** The inference engine calculates the bottom-center coordinate of each vehicle (representing its physical footprint) and uses OpenCV's ray-casting to determine if the vehicle intersects with any custom-drawn region.
- **Dynamic Reporting Cards:** Real-time UI cards automatically generate to track the number of vehicles currently occupying each defined spatial region.

### 2. Interactive Speed Calibration
Rather than hardcoded detection lines, SmartLens provides a fully interactive UI where users can drag and drop "Start" (Line A) and "End" (Line B) thresholds across the video canvas. The system calculates crossing times delta and translates it into real-world speeds (km/h) based on a configurable distance parameter.

### 3. K-Means Color Clustering in HSV Space
Traditional color matching via Euclidean distance in BGR space is highly susceptible to lighting changes, shadows, and windshield reflections. SmartLens employs a sophisticated approach:
- Crops the lower-middle section of the vehicle to avoid asphalt and roof reflections.
- Converts the region of interest to HSV (Hue, Saturation, Value) space.
- Applies K-Means Clustering to extract the most dominant pigment.
- Categorizes the Hue angle into precise color labels (Red, Blue, White, Silver, Black, etc.).

### 4. Stationary Vehicle Detection (Stop Logic)
The system maintains a rolling vector history (last 15 frames) of every tracked object. If a vehicle's absolute displacement falls below a critical threshold (e.g., 5 pixels), it is flagged as `[STOP]` in the UI, enabling instant detection of traffic jams or illegal parking.

### 5. Multi-Threaded Architecture
Built with PySide6, the application completely decouples the heavy YOLO/OpenCV inference loop (running on a dedicated `QThread`) from the main GUI thread, ensuring zero UI blocking even during intense frame processing. Supports hardware acceleration via NVIDIA TensorRT (`.engine`) for zero-delay edge inference.

### 6. SQLite Logging
Every tracked vehicle, along with its color, class, and calculated speed, is asynchronously logged into a local SQLite database for historical data analysis.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python main.py`
3. Click "Browse" to select a video source or enter `0` for a local webcam.
4. Draw regions or adjust lines directly on the video canvas while the video is paused or stopped.
