import cv2
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
from database import VehicleLog, init_db
import numpy as np

# Custom Colors (B, G, R) for OpenCV per Class
CLASS_COLORS = {
    2: (255, 0, 0),    # Car: Blue
    3: (0, 255, 0),    # Motorcycle: Green
    5: (0, 255, 255),  # Bus: Yellow
    7: (0, 0, 255)     # Truck: Red
}

def get_dominant_color(frame, bbox, threshold=None):
    """Estimate dominant color of the vehicle using K-Means and HSV space."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return "Unknown"
    
    # Smart Crop: Lower middle section to avoid windshields and tires/shadows
    cy1 = y1 + int(h * 0.4)
    cy2 = y1 + int(h * 0.8)
    cx1 = x1 + int(w * 0.25)
    cx2 = x1 + int(w * 0.75)
    
    roi = frame[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return "Unknown"
        
    pixels = np.float32(roi.reshape(-1, 3))
    if len(pixels) < 5:
        return "Unknown"
        
    # K-Means clustering to find 2 dominant colors (avoids noise)
    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 3, flags)
    
    # Find the most frequent color cluster
    _, counts = np.unique(labels, return_counts=True)
    dominant_bgr = palette[np.argmax(counts)]
    
    # Convert dominant color to HSV for robust classification
    bgr_patch = np.uint8([[dominant_bgr]])
    hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    h_val, s_val, v_val = hsv_patch[0][0]
    
    # HSV Tuning (Hue 0-179, Saturation 0-255, Value 0-255)
    if s_val < 40 and v_val > 180: return "White"
    if v_val < 50: return "Black"
    if s_val < 60 and 50 <= v_val <= 180: return "Silver/Grey"
    
    if (0 <= h_val <= 10) or (160 <= h_val <= 179): return "Red"
    elif 11 <= h_val <= 25: return "Orange"
    elif 26 <= h_val <= 34: return "Yellow"
    elif 35 <= h_val <= 85: return "Green"
    elif 86 <= h_val <= 130: return "Blue"
    elif 131 <= h_val <= 159: return "Purple"
    
    return "Unknown"

class InferenceThread(QThread):
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(dict)

    def __init__(self, model_path="yolo11n.pt", source=0):
        super().__init__()
        self.model_path = model_path
        self.source = source
        self.running = True
        self.paused = False
        self.db_session = init_db()()
        
        self.counted_ids = set()
        
        # Track history for line crossing & speed logic
        self.track_history = {} # track_id -> list of (cx, cy)
        self.entry_frames = {}  # track_id -> frame number when entering Speed Zone (Line A)
        self.vehicle_speeds = {} # track_id -> calculated speed (km/h)
        self.vehicle_colors = {} # track_id -> detected color
        
        # Mapping global track_id to class-specific ID
        self.class_counts = {2: 0, 3: 0, 5: 0, 7: 0} 
        self.global_to_class_id = {}
        
        # Detection Configs (Set by UI)
        self.conf_threshold = 0.25
        self.active_classes = [2, 3, 5, 7]
        self.real_distance_m = 10.0 
        self.color_threshold = 20000.0 # Obsolete with KMeans, kept for UI compatibility
        self.line_a_pct = 0.3
        self.line_b_pct = 0.7
        self.custom_regions = []

    def set_detection_config(self, conf, active_classes, real_distance_m, color_thresh, line_a, line_b, custom_regions=None):
        self.conf_threshold = conf
        self.active_classes = active_classes
        self.real_distance_m = real_distance_m
        self.color_threshold = color_thresh
        self.line_a_pct = line_a
        self.line_b_pct = line_b
        if custom_regions is not None:
            self.custom_regions = custom_regions

    def pause_resume(self):
        self.paused = not self.paused
        return self.paused

    def run(self):
        print(f"Loading model from {self.model_path}...")
        model = YOLO(self.model_path)
        
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Failed to open video source {self.source}")
            return
            
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0 
            
        if width == 0 or height == 0:
            width, height = 640, 480 

        frame_count = 0

        while self.running and cap.isOpened():
            if self.paused:
                self.msleep(50)
                continue
                
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Define 2 Speed Lines based on dynamic percentage
            line_a_y = int(height * self.line_a_pct)
            line_b_y = int(height * self.line_b_pct)
            
            # Setup Custom Regions
            region_counts = {r["name"]: 0 for r in self.custom_regions}
            scaled_regions = []
            for r in self.custom_regions:
                pts = np.array([(int(x*width), int(y*height)) for x,y in r["points"]], np.int32)
                scaled_regions.append({"name": r["name"], "color": r["color"], "pts": pts})
                
                # Draw Semi-Transparent Polygon
                cv2.polylines(frame, [pts], True, r["color"], 2)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], r["color"])
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            
            results = model.track(frame, persist=True, conf=self.conf_threshold, verbose=False)
            
            current_in_frame = 0
            
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                
                for box, track_id, cls in zip(boxes, track_ids, classes):
                    if cls in self.active_classes:
                        current_in_frame += 1
                        x1, y1, x2, y2 = box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        # Map to Class Specific ID
                        if track_id not in self.global_to_class_id:
                            self.class_counts[cls] += 1
                            self.global_to_class_id[track_id] = self.class_counts[cls]
                        class_specific_id = self.global_to_class_id[track_id]
                        
                        # Detect Color once per vehicle using KMeans
                        if track_id not in self.vehicle_colors:
                            self.vehicle_colors[track_id] = get_dominant_color(frame, box, threshold=self.color_threshold)
                        
                        # History tracking (increased to 15 frames for better STOP detection)
                        if track_id not in self.track_history:
                            self.track_history[track_id] = []
                        self.track_history[track_id].append((cx, cy))
                        if len(self.track_history[track_id]) > 15:
                            self.track_history[track_id].pop(0)
                        
                        is_stopped = False
                        
                        # Speed & Crossing & Stop Logic
                        if len(self.track_history[track_id]) >= 2:
                            prev_cx, prev_cy = self.track_history[track_id][-2]
                            
                            # Check crossing Line A (Top)
                            if (prev_cy < line_a_y and cy >= line_a_y) or (prev_cy > line_a_y and cy <= line_a_y):
                                self.entry_frames[track_id] = frame_count
                            
                            # Check crossing Line B (Bottom)
                            if (prev_cy < line_b_y and cy >= line_b_y) or (prev_cy > line_b_y and cy <= line_b_y):
                                if track_id in self.entry_frames and track_id not in self.counted_ids:
                                    frames_taken = abs(frame_count - self.entry_frames[track_id])
                                    time_taken = frames_taken / fps
                                    
                                    if time_taken > 0:
                                        speed_ms = self.real_distance_m / time_taken
                                        speed_kmh = int(speed_ms * 3.6)
                                    else:
                                        speed_kmh = 0
                                        
                                    self.vehicle_speeds[track_id] = speed_kmh
                                    self.counted_ids.add(track_id)
                                    
                                    # Log to DB
                                    class_name = model.names[cls]
                                    v_color = self.vehicle_colors[track_id]
                                    log = VehicleLog(
                                        track_id=int(track_id), 
                                        vehicle_class=class_name,
                                        color=v_color,
                                        speed_kmh=speed_kmh
                                    )
                                    self.db_session.add(log)
                                    self.db_session.commit()

                            # Stop Logic: Check if it moved less than 5 pixels over the last 15 frames
                            if len(self.track_history[track_id]) >= 10:
                                first_cx, first_cy = self.track_history[track_id][0]
                                distance = np.sqrt((cx - first_cx)**2 + (cy - first_cy)**2)
                                if distance < 5.0:
                                    is_stopped = True
                        
                        # Region Intersection Logic
                        bottom_center = (float(cx), float(y2))
                        for r in scaled_regions:
                            if cv2.pointPolygonTest(r["pts"], bottom_center, False) >= 0:
                                region_counts[r["name"]] += 1

                        # Draw Custom BBox and Colors
                        color = CLASS_COLORS.get(cls, (255, 255, 255))
                        if is_stopped:
                            color = (0, 0, 255) # Red for stopped vehicle
                            
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw center crosshair
                        cv2.line(frame, (cx - 10, cy), (cx + 10, cy), color, 2)
                        cv2.line(frame, (cx, cy - 10), (cx, cy + 10), color, 2)
                        
                        # Draw Label (ID, Color, Speed)
                        class_name = model.names[cls].capitalize()
                        v_color = self.vehicle_colors.get(track_id, "")
                        v_speed = self.vehicle_speeds.get(track_id, "?")
                        
                        label_1 = f"{class_name} {class_specific_id} ({v_color})"
                        if is_stopped:
                            label_1 += " [STOP]"
                            
                        label_2 = f"{v_speed} km/h" if track_id in self.vehicle_speeds else "Measuring..."
                        
                        cv2.putText(frame, label_1, (x1, max(y1 - 25, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(frame, label_2, (x1, max(y1 - 10, 35)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Emit to UI
            annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(annotated_frame_rgb)
            
            # Send stats
            self.stats_ready.emit({
                "current_in_frame": current_in_frame,
                "total_counted": len(self.counted_ids),
                "counts_per_class": {
                    "car": self.class_counts.get(2, 0),
                    "motorcycle": self.class_counts.get(3, 0),
                    "bus": self.class_counts.get(5, 0),
                    "truck": self.class_counts.get(7, 0)
                },
                "region_counts": region_counts
            })
            
        cap.release()
        self.db_session.close()

    def stop(self):
        self.running = False
        self.wait()
