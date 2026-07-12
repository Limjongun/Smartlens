import sys
import cv2
import numpy as np
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                               QHBoxLayout, QWidget, QPushButton, QComboBox, QLineEdit, QFrame, QFileDialog, QSlider, QCheckBox, QScrollArea, QInputDialog)
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QPolygon, QBrush
from PySide6.QtCore import Qt, QPoint
from inference import InferenceThread

MODERN_STYLE = """
QMainWindow {
    background-color: #1e1e2e;
}
QLabel {
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame#card {
    background-color: #313244;
    border-radius: 10px;
    padding: 10px;
}
QLineEdit, QComboBox {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #45475a;
    border-radius: 5px;
    padding: 8px;
    font-size: 14px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1px solid #89b4fa;
}
QPushButton {
    border: none;
    border-radius: 5px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#browseBtn {
    background-color: #89b4fa;
    color: #11111b;
}
QPushButton#browseBtn:hover {
    background-color: #b4befe;
}
QPushButton#startBtn {
    background-color: #a6e3a1;
    color: #11111b;
}
QPushButton#startBtn:hover {
    background-color: #94e2d5;
}
QPushButton#stopBtn {
    background-color: #f38ba8;
    color: #11111b;
}
QPushButton#stopBtn:hover {
    background-color: #eba0ac;
}
QPushButton#pauseBtn {
    background-color: #f9e2af;
    color: #11111b;
}
QPushButton#pauseBtn:hover {
    background-color: #fab387;
}
QPushButton#clearBtn {
    background-color: #cba6f7;
    color: #11111b;
}
QPushButton#clearBtn:hover {
    background-color: #b4befe;
}
QPushButton:disabled {
    background-color: #45475a;
    color: #a6adc8;
}
QSlider::groove:horizontal {
    border: 1px solid #45475a;
    height: 8px;
    background: #181825;
    margin: 2px 0;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    border: 1px solid #89b4fa;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}
QCheckBox {
    color: #cdd6f4;
    font-size: 14px;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1px solid #45475a;
    background-color: #181825;
}
QCheckBox::indicator:checked {
    background-color: #89b4fa;
    border: 1px solid #89b4fa;
}
"""

class InteractiveVideoLabel(QLabel):
    def __init__(self, app_ref):
        super().__init__()
        self.app = app_ref
        self.dragging = None
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #11111b; border-radius: 5px; color: #7f849c; font-size: 20px;")
        
    def mousePressEvent(self, event):
        if not self.pixmap(): return
        y_pct = event.position().y() / self.height()
        x_pct = event.position().x() / self.width()
        
        dist_a = abs(y_pct - self.app.line_a_pct)
        dist_b = abs(y_pct - self.app.line_b_pct)
        
        if dist_a < 0.05 and dist_a <= dist_b:
            self.dragging = 'A'
        elif dist_b < 0.05:
            self.dragging = 'B'
        else:
            self.dragging = 'Polygon'
            self.app.current_polygon = [(x_pct, y_pct)]
            
    def mouseMoveEvent(self, event):
        if not self.pixmap(): return
        y_pct = max(0.0, min(1.0, event.position().y() / self.height()))
        x_pct = max(0.0, min(1.0, event.position().x() / self.width()))
        
        if self.dragging == 'A':
            self.app.line_a_pct = y_pct
            self.app.update_detection_config()
            self.update() 
        elif self.dragging == 'B':
            self.app.line_b_pct = y_pct
            self.app.update_detection_config()
            self.update()
        elif self.dragging == 'Polygon':
            self.app.current_polygon.append((x_pct, y_pct))
            self.update()
        else:
            dist_a = abs(y_pct - self.app.line_a_pct)
            dist_b = abs(y_pct - self.app.line_b_pct)
            if dist_a < 0.05 or dist_b < 0.05:
                self.setCursor(Qt.SizeVerCursor)
            else:
                self.setCursor(Qt.CrossCursor)
                
    def mouseReleaseEvent(self, event):
        if self.dragging == 'Polygon':
            if len(self.app.current_polygon) > 10: # ensure it's an actual drawing, not a misclick
                name, ok = QInputDialog.getText(self, "Area Segmentation", "Masukan nama area ini (Contoh: Trotoar, Bahu Jalan):")
                if ok and name.strip():
                    color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
                    self.app.custom_regions.append({
                        "name": name.strip(),
                        "points": self.app.current_polygon.copy(),
                        "color": color
                    })
                    self.app.refresh_region_cards()
                    self.app.update_detection_config()
            self.app.current_polygon = []
            self.update()
            
        self.dragging = None
        self.setCursor(Qt.ArrowCursor)
        
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap(): return
        
        painter = QPainter(self)
        
        # Draw Line A and B
        pen_a = QPen(QColor(255, 0, 255))
        pen_a.setWidth(2)
        painter.setPen(pen_a)
        y_a = int(self.height() * self.app.line_a_pct)
        painter.drawLine(0, y_a, self.width(), y_a)
        painter.drawText(10, y_a - 5, "Line A (Start)")
        
        pen_b = QPen(QColor(0, 255, 255))
        pen_b.setWidth(2)
        painter.setPen(pen_b)
        y_b = int(self.height() * self.app.line_b_pct)
        painter.drawLine(0, y_b, self.width(), y_b)
        painter.drawText(10, y_b - 5, "Line B (End)")

        # Draw Saved Custom Regions if inference is NOT running (otherwise OpenCV draws them on the frame)
        if self.app.inference_thread is None or self.app.inference_thread.paused:
            for region in self.app.custom_regions:
                r, g, b = region["color"]
                pen = QPen(QColor(b, g, r))
                pen.setWidth(2)
                painter.setPen(pen)
                
                brush_color = QColor(b, g, r, 70) # Semi transparent
                painter.setBrush(QBrush(brush_color))
                
                qpoly = QPolygon()
                for px, py in region["points"]:
                    qpoly.append(QPoint(int(px * self.width()), int(py * self.height())))
                painter.drawPolygon(qpoly)

        # Draw Currently Drawing Polygon
        if self.app.current_polygon:
            pen = QPen(QColor(255, 255, 0)) # Yellow while drawing
            pen.setWidth(2)
            painter.setPen(pen)
            for i in range(1, len(self.app.current_polygon)):
                p1 = self.app.current_polygon[i-1]
                p2 = self.app.current_polygon[i]
                painter.drawLine(int(p1[0] * self.width()), int(p1[1] * self.height()),
                                 int(p2[0] * self.width()), int(p2[1] * self.height()))


class SmartLensApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Lens - Interactive Spatial & Speed Tracker")
        self.resize(1200, 950)
        self.setStyleSheet(MODERN_STYLE)
        
        self.line_a_pct = 0.3
        self.line_b_pct = 0.7
        self.custom_regions = []
        self.current_polygon = []
        self.region_labels = {} # name -> QLabel

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # ================= SIDEBAR =================
        sidebar = QFrame()
        sidebar.setObjectName("card")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop)
        
        settings_label = QLabel("Settings")
        settings_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        settings_label.setStyleSheet("color: #f5c2e7;")
        sidebar_layout.addWidget(settings_label)
        
        sidebar_layout.addSpacing(20)
        
        conf_label = QLabel("Confidence Threshold:")
        conf_label.setFont(QFont("Segoe UI", 12))
        sidebar_layout.addWidget(conf_label)
        
        self.conf_val_label = QLabel("25%")
        self.conf_val_label.setStyleSheet("color: #89b4fa; font-weight: bold;")
        
        conf_layout = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(1)
        self.conf_slider.setMaximum(100)
        self.conf_slider.setValue(25)
        self.conf_slider.valueChanged.connect(self.update_detection_config)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_val_label)
        sidebar_layout.addLayout(conf_layout)
        
        sidebar_layout.addSpacing(10)
        
        color_label = QLabel("Color Match Threshold:")
        color_label.setFont(QFont("Segoe UI", 12))
        sidebar_layout.addWidget(color_label)
        
        self.color_val_label = QLabel("20000")
        self.color_val_label.setStyleSheet("color: #a6e3a1; font-weight: bold;")
        
        color_layout = QHBoxLayout()
        self.color_slider = QSlider(Qt.Horizontal)
        self.color_slider.setMinimum(1000)
        self.color_slider.setMaximum(100000)
        self.color_slider.setValue(20000)
        self.color_slider.valueChanged.connect(self.update_detection_config)
        color_layout.addWidget(self.color_slider)
        color_layout.addWidget(self.color_val_label)
        sidebar_layout.addLayout(color_layout)
        
        sidebar_layout.addSpacing(20)
        
        classes_label = QLabel("Vehicle Classes:")
        classes_label.setFont(QFont("Segoe UI", 12))
        sidebar_layout.addWidget(classes_label)
        
        self.chk_car = QCheckBox("Car (Mobil)")
        self.chk_car.setChecked(True)
        self.chk_car.stateChanged.connect(self.update_detection_config)
        
        self.chk_motorcycle = QCheckBox("Motorcycle (Motor)")
        self.chk_motorcycle.setChecked(True)
        self.chk_motorcycle.stateChanged.connect(self.update_detection_config)
        
        self.chk_bus = QCheckBox("Bus")
        self.chk_bus.setChecked(True)
        self.chk_bus.stateChanged.connect(self.update_detection_config)
        
        self.chk_truck = QCheckBox("Truck (Truk)")
        self.chk_truck.setChecked(True)
        self.chk_truck.stateChanged.connect(self.update_detection_config)
        
        sidebar_layout.addWidget(self.chk_car)
        sidebar_layout.addWidget(self.chk_motorcycle)
        sidebar_layout.addWidget(self.chk_bus)
        sidebar_layout.addWidget(self.chk_truck)
        
        sidebar_layout.addSpacing(20)
        
        dist_label = QLabel("Real Distance A to B (meters):")
        dist_label.setFont(QFont("Segoe UI", 12))
        sidebar_layout.addWidget(dist_label)
        
        self.dist_val_label = QLabel("10m")
        self.dist_val_label.setStyleSheet("color: #f38ba8; font-weight: bold;")
        
        dist_layout = QHBoxLayout()
        self.dist_slider = QSlider(Qt.Horizontal)
        self.dist_slider.setMinimum(1)
        self.dist_slider.setMaximum(100)
        self.dist_slider.setValue(10)
        self.dist_slider.valueChanged.connect(self.update_detection_config)
        dist_layout.addWidget(self.dist_slider)
        dist_layout.addWidget(self.dist_val_label)
        sidebar_layout.addLayout(dist_layout)
        
        main_layout.addWidget(sidebar)


        # ================= MAIN CONTENT =================
        content_layout = QVBoxLayout()
        main_layout.addLayout(content_layout, stretch=1)

        header_label = QLabel("Smart Lens Dashboard")
        header_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header_label.setStyleSheet("color: #89b4fa;")
        content_layout.addWidget(header_label)

        controls_card = QFrame()
        controls_card.setObjectName("card")
        top_layout = QHBoxLayout(controls_card)
        
        self.source_input = QLineEdit()
        self.source_input.setPlaceholderText("Select a video file or enter '0' for webcam")
        self.source_input.setText("0") 
        
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.setObjectName("browseBtn")
        self.browse_btn.clicked.connect(self.browse_video)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n.pt", 
            "yolo11n.engine"
        ])

        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("startBtn")
        self.start_btn.clicked.connect(self.start_inference)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setObjectName("pauseBtn")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.clicked.connect(self.stop_inference)
        self.stop_btn.setEnabled(False)

        self.clear_btn = QPushButton("Clear Regions")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_custom_regions)

        top_layout.addWidget(QLabel("Source:"))
        top_layout.addWidget(self.source_input, stretch=2)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.model_combo, stretch=1)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.pause_btn)
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.clear_btn)
        
        content_layout.addWidget(controls_card)
        
        # Video Display Card
        video_card = QFrame()
        video_card.setObjectName("card")
        video_layout = QVBoxLayout(video_card)
        video_layout.setContentsMargins(5, 5, 5, 5)

        self.video_label = InteractiveVideoLabel(self)
        self.video_label.setText("Video Stream will appear here (Draw segments & Drag lines)")
        video_layout.addWidget(self.video_label)
        
        content_layout.addWidget(video_card, stretch=1)

        # Stats Dashboard Card
        stats_card = QFrame()
        stats_card.setObjectName("card")
        stats_layout = QHBoxLayout(stats_card)
        
        self.lbl_car = QLabel("🚗 Car: 0")
        self.lbl_motor = QLabel("🏍️ Motor: 0")
        self.lbl_bus = QLabel("🚌 Bus: 0")
        self.lbl_truck = QLabel("🚚 Truck: 0")
        self.lbl_total = QLabel("Total In Frame: 0")
        
        for lbl in [self.lbl_car, self.lbl_motor, self.lbl_bus, self.lbl_truck, self.lbl_total]:
            lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("color: #a6e3a1;")
        
        self.lbl_total.setStyleSheet("color: #f9e2af; font-size: 18px;")

        stats_layout.addWidget(self.lbl_car)
        stats_layout.addWidget(self.lbl_motor)
        stats_layout.addWidget(self.lbl_bus)
        stats_layout.addWidget(self.lbl_truck)
        stats_layout.addWidget(self.lbl_total)
        
        content_layout.addWidget(stats_card)
        
        # Region Tracking Cards
        self.region_container = QFrame()
        self.region_container.setObjectName("card")
        self.region_container_layout = QHBoxLayout(self.region_container)
        self.region_container_layout.setAlignment(Qt.AlignLeft)
        content_layout.addWidget(self.region_container)
        self.region_container.hide() # Hidden until regions are drawn

        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)

        self.inference_thread = None

    def refresh_region_cards(self):
        # Clear layout
        while self.region_container_layout.count():
            item = self.region_container_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        self.region_labels.clear()
        
        if not self.custom_regions:
            self.region_container.hide()
            return
            
        self.region_container.show()
        
        report_label = QLabel("Spatial Reports:")
        report_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        report_label.setStyleSheet("color: #f38ba8;")
        self.region_container_layout.addWidget(report_label)
        
        for region in self.custom_regions:
            card = QFrame()
            r,g,b = region["color"]
            card.setStyleSheet(f"background-color: rgba({b},{g},{r}, 0.2); border: 1px solid rgb({b},{g},{r}); border-radius: 5px;")
            card_layout = QVBoxLayout(card)
            
            title = QLabel(region["name"])
            title.setFont(QFont("Segoe UI", 12, QFont.Bold))
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet("color: #cdd6f4;")
            
            val_lbl = QLabel("0 Vehicles")
            val_lbl.setFont(QFont("Segoe UI", 14, QFont.Bold))
            val_lbl.setAlignment(Qt.AlignCenter)
            val_lbl.setStyleSheet("color: #a6e3a1;")
            
            card_layout.addWidget(title)
            card_layout.addWidget(val_lbl)
            
            self.region_labels[region["name"]] = val_lbl
            self.region_container_layout.addWidget(card)

    def clear_custom_regions(self):
        self.custom_regions = []
        self.refresh_region_cards()
        self.update_detection_config()
        self.video_label.update()

    def browse_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
        if file_path:
            self.source_input.setText(file_path)
            
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.update_frame(frame)
                cap.release()

    def update_detection_config(self):
        conf_val = self.conf_slider.value()
        self.conf_val_label.setText(f"{conf_val}%")
        
        color_val = self.color_slider.value()
        self.color_val_label.setText(str(color_val))
        
        dist_val = self.dist_slider.value()
        self.dist_val_label.setText(f"{dist_val}m")
        
        active_classes = []
        if self.chk_car.isChecked(): active_classes.append(2)
        if self.chk_motorcycle.isChecked(): active_classes.append(3)
        if self.chk_bus.isChecked(): active_classes.append(5)
        if self.chk_truck.isChecked(): active_classes.append(7)
        
        if self.inference_thread is not None:
            self.inference_thread.set_detection_config(
                conf_val / 100.0, 
                active_classes, 
                float(dist_val),
                float(color_val),
                self.line_a_pct,
                self.line_b_pct,
                self.custom_regions
            )

    def start_inference(self):
        source = self.source_input.text()
        if source.isdigit():
            source = int(source)

        model_path = "yolo11n.engine" if "engine" in self.model_combo.currentText() else "yolo11n.pt"

        self.inference_thread = InferenceThread(model_path=model_path, source=source)
        
        self.update_detection_config()
        
        self.inference_thread.frame_ready.connect(self.update_frame)
        self.inference_thread.stats_ready.connect(self.update_stats)
        
        self.inference_thread.start()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("Pause")
        self.source_input.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.model_combo.setEnabled(False)

    def stop_inference(self):
        if self.inference_thread is not None:
            self.inference_thread.stop()
            self.inference_thread = None
            
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("Pause")
        self.source_input.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.video_label.setText("Video Stream Stopped")
        self.video_label.setPixmap(QPixmap())

    def toggle_pause(self):
        if self.inference_thread is not None:
            is_paused = self.inference_thread.pause_resume()
            if is_paused:
                self.pause_btn.setText("Resume")
            else:
                self.pause_btn.setText("Pause")

    def update_frame(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        self.video_label.update() 

    def update_stats(self, stats):
        self.lbl_total.setText(f"In Frame: {stats['current_in_frame']}")
        counts = stats.get("counts_per_class", {})
        self.lbl_car.setText(f"🚗 Car: {counts.get('car', 0)}")
        self.lbl_motor.setText(f"🏍️ Motor: {counts.get('motorcycle', 0)}")
        self.lbl_bus.setText(f"🚌 Bus: {counts.get('bus', 0)}")
        self.lbl_truck.setText(f"🚚 Truck: {counts.get('truck', 0)}")
        
        region_counts = stats.get("region_counts", {})
        for name, lbl in self.region_labels.items():
            cnt = region_counts.get(name, 0)
            lbl.setText(f"{cnt} Vehicles")

    def closeEvent(self, event):
        self.stop_inference()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartLensApp()
    window.show()
    sys.exit(app.exec())
