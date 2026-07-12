# SmartLens: Advanced Spatial Analytics & Vehicle Tracking

![SmartLens Dashboard](screenshot.png)

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

### 5. Parallel Multi-Model Inference & Zero-Code Reflection
SmartLens dynamically scans the `models/` directory for any YOLO `.pt` or `.engine` files. You can run multiple models simultaneously (e.g., Vehicle Detection + License Plate Detection) on the same video stream without ID collisions. The UI automatically generates class checkboxes based on the internal class names of your custom models.

### 6. Data Analytics Dashboard & Violation Auto-Snapshot
Every tracking session generates a unique database session (e.g., `inf_001`). 
- **Dashboard:** Features a built-in PySide6 graphical dashboard using Matplotlib to visualize vehicle class distributions and color stats, along with one-click CSV exporting.
- **Violation Snapshot:** If a vehicle is flagged as `[STOP]` inside a user-defined custom region (e.g., "No Parking Zone"), the system automatically captures a cropped snapshot of the vehicle, saves it to the `evidences/` folder, and logs the violation.

---

## 🚀 Getting Started (Step-by-Step Guide)

### 1. Clone the Repository
```bash
git clone https://github.com/Limjongun/Smartlens.git
cd Smartlens
```

### 2. Set Up Virtual Environment (Recommended)
It is highly recommended to use a virtual environment to avoid dependency conflicts.

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install all required computer vision and data science libraries.
```bash
pip install -r requirements.txt
```

### 4. Prepare Your Models
SmartLens supports dynamic model loading.
- Place your custom YOLO weights (`.pt` or `.engine` files) into the `models/` folder. 
- *(Note: A default `yolo11n.pt` will be downloaded automatically on first run if you don't provide one).*

### 5. Launch the Application
Start the graphical user interface:
```bash
python main.py
```

### 6. Run Inference & Analytics
1. **Source:** Enter `0` for your webcam, or click **Browse** to load a video file.
2. **Select Models:** On the top menu, tick the checkboxes for the models you want to run simultaneously.
3. **Configure:** Adjust the confidence slider, distance mapping, and color thresholds on the left sidebar. 
4. **Draw Zones:** Draw freehand polygons on the video canvas to mark restricted areas. Give them a name (e.g., "Sidewalk").
5. **Start:** Click **Start** to run the AI engine. 
6. **Dashboard:** Click the **Dashboard** button at any time to view graphical analytics, review parking violations, and export the session data to CSV!
