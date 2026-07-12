import time
import cv2
import os
from PySide6.QtCore import QThread, Signal
from ultralytics import YOLO
from database import VehicleLog, ViolationLog, init_db
import numpy as np

from collections import defaultdict

def get_class_color(cls_id):
    """Generate a deterministic color for any class ID."""
    np.random.seed(int(cls_id) * 123)
    return tuple(int(x) for x in np.random.randint(50, 255, 3))

def get_dominant_color(frame, bbox, threshold=None):
    """Estimate dominant color of the vehicle using K-Means and HSV space."""
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clip to frame boundaries
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # Crop the lower-middle portion of the vehicle (avoids windows and shadows)
    crop_h = y2 - y1
    crop_w = x2 - x1
    
    if crop_h < 10 or crop_w < 10:
        return "Unknown"
        
    start_y = y1 + int(crop_h * 0.4)
    end_y = y1 + int(crop_h * 0.9)
    start_x = x1 + int(crop_w * 0.2)
    end_x = x1 + int(crop_w * 0.8)
    
    vehicle_crop = frame[start_y:end_y, start_x:end_x]
    
    if vehicle_crop.size == 0:
        return "Unknown"

    # Convert to HSV for better color invariant clustering
    hsv_crop = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
    
    # Reshape for K-Means
    pixels = hsv_crop.reshape((-1, 3))
    pixels = np.float32(pixels)

    # Define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Get the most dominant cluster
    counts = np.bincount(labels.flatten())
    dominant_cluster_idx = np.argmax(counts)
    dominant_hsv = centers[dominant_cluster_idx]
    
    hue = dominant_hsv[0]
    sat = dominant_hsv[1]
    val = dominant_hsv[2]

    # Map Hue to Color Names
    if sat < 40 and val > 180:
        return "White"
    elif sat < 50 and 50 < val <= 180:
        return "Silver/Grey"
    elif val <= 50:
        return "Black"
    elif 0 <= hue <= 10 or 160 <= hue <= 180:
        return "Red"
    elif 11 <= hue <= 25:
        return "Orange"
    elif 26 <= hue <= 34:
        return "Yellow"
    elif 35 <= hue <= 85:
        return "Green"
    elif 86 <= hue <= 125:
        return "Blue"
    elif 126 <= hue <= 159:
        return "Purple"
    
    return "Unknown"

class InferenceThread(QThread):
    frame_ready = Signal(np.ndarray)
    stats_ready = Signal(dict)

    def __init__(self, model_paths, source, session_name, parent=None):
        super().__init__(parent)
        self.model_paths = model_paths
        self.source = source
        self.session_name = session_name
        self.is_stopped = False
        self.paused = False
        self.logged_violations = set()
        os.makedirs("evidences", exist_ok=True)
        
        # State tracking
        self.track_history = {} # track_id -> [(cx, cy), ...]
        self.entry_times = {} # track_id -> time crossed line A
        self.counted_ids = set() # track_id -> bool
        self.vehicle_speeds = {} # track_id -> speed
        self.vehicle_colors = {} # track_id -> detected color
        
        self.class_counts = defaultdict(int) 
        
        # Detection Configs (Set by UI)
        self.conf_threshold = 0.25
        self.active_classes = []
        self.real_distance_m = 10.0 
        self.color_threshold = 20000.0
        self.line_a_pct = 0.3
        self.line_b_pct = 0.7
        self.custom_regions = []
        self.model_to_global = {}

        # Init Database
        self.SessionLocal = init_db()

    def set_detection_config(self, conf, active_classes, real_distance_m, color_thresh, line_a, line_b, custom_regions=None, model_to_global=None):
        self.conf_threshold = conf
        self.active_classes = active_classes
        self.real_distance_m = real_distance_m
        self.color_threshold = color_thresh
        self.line_a_pct = line_a
        self.line_b_pct = line_b
        if custom_regions is not None:
            self.custom_regions = custom_regions
        if model_to_global is not None:
            self.model_to_global = model_to_global

    def pause_resume(self):
        self.paused = not self.paused
        return self.paused

    def stop(self):
        self.is_stopped = True
        self.quit()
        self.wait()

    def run(self):
        db = self.SessionLocal()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"Failed to open source: {self.source}")
            return
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0

        # Load all models dynamically
        models = []
        for path in self.model_paths:
            print(f"Thread Loading: {path}")
            models.append(YOLO(path, task='detect'))

        while cap.isOpened() and not self.is_stopped:
            if self.paused:
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret:
                break
                
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
            
            # PARALLEL MULTI-MODEL EXECUTION
            all_boxes = []
            all_cls = []
            all_conf = []
            all_tids = []
            
            for m_idx, model in enumerate(models):
                m_path = self.model_paths[m_idx]
                results = model.track(frame, persist=True, conf=self.conf_threshold, verbose=False)
                
                if results and results[0].boxes:
                    for box in results[0].boxes:
                        local_cls = int(box.cls[0].item())
                        
                        # Map to global ID
                        if m_path in self.model_to_global and local_cls in self.model_to_global[m_path]:
                            global_cls = self.model_to_global[m_path][local_cls]
                        else:
                            continue
                            
                        if global_cls not in self.active_classes:
                            continue
                            
                        all_boxes.append(box.xyxy[0].cpu().numpy())
                        all_cls.append(global_cls)
                        all_conf.append(box.conf[0].item())
                        
                        if box.id is not None:
                            t_id = int(box.id[0].item())
                            global_t_id = f"{m_idx}_{t_id}"
                            all_tids.append(global_t_id)
                        else:
                            all_tids.append(None)
            
            current_in_frame = len(all_boxes)
            
            # Process all detected objects
            for i in range(current_in_frame):
                b = all_boxes[i]
                cls = all_cls[i]
                conf = all_conf[i]
                t_id = all_tids[i]
                
                x1, y1, x2, y2 = map(int, b)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                color = get_class_color(cls)
                is_stopped = False
                speed_text = ""
                color_text = ""
                stop_text = ""
                label = ""
                
                if t_id is not None:
                    if t_id not in self.track_history:
                        self.track_history[t_id] = []
                    self.track_history[t_id].append((cx, cy))
                    
                    if len(self.track_history[t_id]) > 15:
                        self.track_history[t_id].pop(0)

                    # Determine Color if not yet detected
                    if t_id not in self.vehicle_colors:
                        if (x2 - x1) > 20 and (y2 - y1) > 20: # Must be large enough to sample
                            dominant_color = get_dominant_color(frame, b)
                            if dominant_color != "Unknown":
                                self.vehicle_colors[t_id] = dominant_color
                            else:
                                self.vehicle_colors[t_id] = "(Calculating)"
                        else:
                            self.vehicle_colors[t_id] = "(Too far)"

                    # Stop logic (if distance moved in last 10 frames is very small)
                    if len(self.track_history[t_id]) >= 10:
                        first_cx, first_cy = self.track_history[t_id][0]
                        distance = np.sqrt((cx - first_cx)**2 + (cy - first_cy)**2)
                        if distance < 5.0:
                            is_stopped = True
                
                    # Region Intersection Logic
                    bottom_center = (float(cx), float(y2))
                    in_region_name = None
                    for r in scaled_regions:
                        if cv2.pointPolygonTest(r["pts"], bottom_center, False) >= 0:
                            region_counts[r["name"]] += 1
                            in_region_name = r["name"]

                    if is_stopped:
                        color = (0, 0, 255) # Red for stopped vehicle
                        
                        # Trigger Violation Snapshot
                        if in_region_name and t_id not in self.logged_violations:
                            self.logged_violations.add(t_id)
                            filename = f"evidences/{self.session_name}_{t_id}_{int(time.time())}.jpg"
                            # Expand crop slightly
                            crop = frame[max(0, y1-20):min(frame.shape[0], y2+20), max(0, x1-20):min(frame.shape[1], x2+20)]
                            if crop.size > 0:
                                cv2.imwrite(filename, crop)
                                new_violation = ViolationLog(
                                    session_name=self.session_name,
                                    track_id=str(t_id),
                                    vehicle_class=str(cls),
                                    region_name=in_region_name,
                                    image_path=filename
                                )
                                db.add(new_violation)
                                db.commit()
                        
                    # Track Crossing A to B for Speed Calculation
                    if t_id not in self.entry_times and cy > line_a_y - 20 and cy < line_a_y + 20:
                        self.entry_times[t_id] = time.time()
                        
                    if t_id in self.entry_times and t_id not in self.vehicle_speeds and cy > line_b_y - 20 and cy < line_b_y + 20:
                        delta_t = time.time() - self.entry_times[t_id]
                        if delta_t > 0:
                            speed_mps = self.real_distance_m / delta_t
                            speed_kmh = speed_mps * 3.6
                            self.vehicle_speeds[t_id] = speed_kmh
                            
                            # Logging to DB when speed is finalized
                            new_log = VehicleLog(
                                session_name=self.session_name,
                                track_id=str(t_id),
                                vehicle_class=str(cls),
                                color=self.vehicle_colors.get(t_id, "Unknown"),
                                speed_kmh=speed_kmh
                            )
                            db.add(new_log)
                            db.commit()
                            
                    # Counting Logic
                    if t_id not in self.counted_ids and cy > line_b_y:
                        self.counted_ids.add(t_id)
                        self.class_counts[cls] += 1
                        
                    # Annotations for tracked object
                    speed_text = f"{self.vehicle_speeds[t_id]:.0f} km/h" if t_id in self.vehicle_speeds else "Measuring..."
                    color_text = self.vehicle_colors.get(t_id, "")
                    stop_text = "[STOP]" if is_stopped else ""
                    label = f"ID:{t_id} ({color_text}) {stop_text}"
                else:
                    # Fallback for untracked objects
                    label = f"Detected (No ID)"
                    
                # DRAW EVERYTHING (Tracked and Untracked)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if speed_text:
                    cv2.putText(frame, speed_text, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            annotated_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.frame_ready.emit(annotated_frame_rgb)
            
            # Send stats
            counts_per_class = {cls_id: count for cls_id, count in self.class_counts.items()}
            
            self.stats_ready.emit({
                "current_in_frame": current_in_frame,
                "total_counted": len(self.counted_ids),
                "counts_per_class": counts_per_class,
                "region_counts": region_counts
            })
            
        db.close()
        cap.release()
