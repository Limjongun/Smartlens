import sys
import os
import cv2
import numpy as np
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                               QHBoxLayout, QWidget, QPushButton, QLineEdit, QFrame, QFileDialog, QSlider, QCheckBox, QScrollArea, QInputDialog, QListWidget, QListWidgetItem)
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QColor, QPolygon, QBrush
from PySide6.QtCore import Qt, QPoint
from inference import InferenceThread
from ultralytics import YOLO
from database import init_db, InferenceSession
from dashboard import DashboardWindow

MODERN_STYLE = """
QMainWindow {
    background-color: #11111b; /* Darker background */
}
QLabel {
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QFrame#card {
    background-color: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 15px;
}
QLineEdit, QListWidget {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
}
QLineEdit:focus, QListWidget:focus {
    border: 1px solid #8caaee;
    background-color: #1e1e2e;
}
QListWidget::item {
    padding: 8px;
    border-radius: 5px;
    margin-bottom: 2px;
}
QListWidget::item:hover {
    background-color: #313244;
}
QListWidget::item:selected {
    background-color: #45475a;
    color: #8caaee;
    border-left: 3px solid #8caaee;
}
QPushButton {
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 14px;
    font-weight: bold;
}
QPushButton#browseBtn {
    background-color: #8caaee;
    color: #232634;
}
QPushButton#browseBtn:hover {
    background-color: #99d1db;
}
QPushButton#startBtn {
    background-color: #a6d189;
    color: #232634;
}
QPushButton#startBtn:hover {
    background-color: #81c8be;
}
QPushButton#stopBtn {
    background-color: #e78284;
    color: #232634;
}
QPushButton#stopBtn:hover {
    background-color: #ea999c;
}
QPushButton#pauseBtn {
    background-color: #e5c890;
    color: #232634;
}
QPushButton#pauseBtn:hover {
    background-color: #ef9f76;
}
QPushButton#clearBtn {
    background-color: #ca9ee6;
    color: #232634;
}
QPushButton#clearBtn:hover {
    background-color: #babbf1;
}
QPushButton:disabled {
    background-color: #414559;
    color: #a5adce;
}
QSlider::groove:horizontal {
    border: 1px solid #313244;
    height: 6px;
    background: #181825;
    margin: 2px 0;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #8caaee;
    border: 1px solid #8caaee;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #99d1db;
    border: 1px solid #99d1db;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}
QCheckBox {
    color: #cdd6f4;
    font-size: 14px;
    spacing: 10px;
    padding: 5px;
}
QCheckBox:hover {
    background-color: #1e1e2e;
    border-radius: 5px;
}
QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid #45475a;
    background-color: #11111b;
}
QCheckBox::indicator:hover {
    border: 2px solid #8caaee;
}
QCheckBox::indicator:checked {
    background-color: #8caaee;
    border: 2px solid #8caaee;
}
QScrollBar:vertical {
    border: none;
    background: #11111b;
    width: 8px;
    margin: 0px 0px 0px 0px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #45475a;
    min-height: 30px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #585b70;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
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
            if len(self.app.current_polygon) > 10:
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

        if self.app.inference_thread is None or self.app.inference_thread.paused:
            for region in self.app.custom_regions:
                r, g, b = region["color"]
                pen = QPen(QColor(b, g, r))
                pen.setWidth(2)
                painter.setPen(pen)
                
                brush_color = QColor(b, g, r, 70) 
                painter.setBrush(QBrush(brush_color))
                
                qpoly = QPolygon()
                for px, py in region["points"]:
                    qpoly.append(QPoint(int(px * self.width()), int(py * self.height())))
                painter.drawPolygon(qpoly)

        if self.app.current_polygon:
            pen = QPen(QColor(255, 255, 0))
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
        self.resize(1300, 950)
        self.setStyleSheet(MODERN_STYLE)
        
        self.line_a_pct = 0.3
        self.line_b_pct = 0.7
        self.custom_regions = []
        self.current_polygon = []
        self.region_labels = {} 
        self.class_checkboxes = {}
        self.class_stat_labels = {}
        
        self.active_model_paths = []
        self.global_names = {}
        self.model_to_global = {}

        self.dashboard_window = None
        self.SessionLocal = init_db()

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
        sidebar.setFixedWidth(300)
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
        
        classes_label = QLabel("Detection Classes:")
        classes_label.setFont(QFont("Segoe UI", 12))
        sidebar_layout.addWidget(classes_label)
        
        self.classes_scroll = QScrollArea()
        self.classes_scroll.setWidgetResizable(True)
        self.classes_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        self.classes_widget = QWidget()
        self.classes_layout = QVBoxLayout(self.classes_widget)
        self.classes_scroll.setWidget(self.classes_widget)
        sidebar_layout.addWidget(self.classes_scroll, stretch=1)
        
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

        self.models_list = QListWidget()
        self.models_list.setMaximumHeight(80)
        self.models_list.setMinimumWidth(200)
        self.models_list.itemChanged.connect(self.on_models_changed)

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
        
        self.dashboard_btn = QPushButton("Dashboard")
        self.dashboard_btn.setObjectName("startBtn")
        self.dashboard_btn.clicked.connect(self.open_dashboard)

        top_layout.addWidget(QLabel("Source:"))
        top_layout.addWidget(self.source_input, stretch=2)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(QLabel("Models:"))
        top_layout.addWidget(self.models_list, stretch=1)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.pause_btn)
        top_layout.addWidget(self.stop_btn)
        top_layout.addWidget(self.clear_btn)
        top_layout.addWidget(self.dashboard_btn)
        
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

        # ================= RIGHT SIDEBAR =================
        right_sidebar = QFrame()
        right_sidebar.setObjectName("card")
        right_sidebar.setFixedWidth(300)
        right_sidebar_layout = QVBoxLayout(right_sidebar)
        right_sidebar_layout.setAlignment(Qt.AlignTop)
        
        stats_label = QLabel("Detection Stats")
        stats_label.setFont(QFont("Segoe UI", 18, QFont.Bold))
        stats_label.setStyleSheet("color: #f5c2e7;")
        right_sidebar_layout.addWidget(stats_label)
        
        self.lbl_total = QLabel("Total In Frame: 0")
        self.lbl_total.setFont(QFont("Segoe UI", 16, QFont.Bold))
        self.lbl_total.setStyleSheet("color: #f9e2af;")
        self.lbl_total.setAlignment(Qt.AlignLeft)
        right_sidebar_layout.addWidget(self.lbl_total)
        
        right_sidebar_layout.addSpacing(10)
        
        self.stats_scroll = QScrollArea()
        self.stats_scroll.setWidgetResizable(True)
        self.stats_scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")
        
        self.stats_widget = QWidget()
        self.stats_layout = QVBoxLayout(self.stats_widget)
        self.stats_layout.setAlignment(Qt.AlignTop)
        self.stats_scroll.setWidget(self.stats_widget)
        right_sidebar_layout.addWidget(self.stats_scroll, stretch=1)
        
        main_layout.addWidget(right_sidebar)
        
        # Region Tracking Cards
        self.region_container = QFrame()
        self.region_container.setObjectName("card")
        self.region_container_layout = QHBoxLayout(self.region_container)
        self.region_container_layout.setAlignment(Qt.AlignLeft)
        content_layout.addWidget(self.region_container)
        self.region_container.hide() 

        scroll_area.setWidget(main_widget)
        self.setCentralWidget(scroll_area)

        self.inference_thread = None
        
        # Load Models Directory
        self.load_available_models()

    def load_available_models(self):
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)
        
        for f in ["yolo11n.pt", "yolo11n.engine"]:
            if os.path.exists(f) and not os.path.exists(os.path.join(models_dir, f)):
                import shutil
                try:
                    shutil.move(f, os.path.join(models_dir, f))
                except Exception as e:
                    print(e)
                    
        files = []
        if os.path.exists(models_dir):
            files = [f for f in os.listdir(models_dir) if f.endswith('.pt') or f.endswith('.engine')]
            
        if not files:
            files = ["yolo11n.pt"] 
            
        self.models_list.blockSignals(True)
        self.models_list.clear()
        for f in files:
            item = QListWidgetItem(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if f == "yolo11n.pt" else Qt.Unchecked)
            self.models_list.addItem(item)
        self.models_list.blockSignals(False)
        
        self.on_models_changed()

    def on_models_changed(self):
        self.start_btn.setEnabled(False)
        self.start_btn.setText("Loading...")
        QApplication.processEvents()
        
        self.active_model_paths = []
        for i in range(self.models_list.count()):
            item = self.models_list.item(i)
            if item.checkState() == Qt.Checked:
                m_filename = item.text()
                m_path = os.path.join("models", m_filename)
                if not os.path.exists(m_path):
                    m_path = m_filename
                self.active_model_paths.append(m_path)
                
        self.global_names = {}
        self.model_to_global = {}
        global_id = 0
        
        for m_path in self.active_model_paths:
            try:
                model = YOLO(m_path, task='detect')
                self.model_to_global[m_path] = {}
                m_name = os.path.basename(m_path)
                
                for local_cls, cls_name in model.names.items():
                    global_cls_name = f"{m_name.split('.')[0]}: {cls_name}"
                    self.global_names[global_id] = global_cls_name
                    self.model_to_global[m_path][local_cls] = global_id
                    global_id += 1
            except Exception as e:
                print(f"Error inspecting {m_path}:", e)
                
        self.rebuild_dynamic_ui()
        
        self.start_btn.setEnabled(True)
        self.start_btn.setText("Start")
        
    def rebuild_dynamic_ui(self):
        # 1. Rebuild Checkboxes
        while self.classes_layout.count():
            item = self.classes_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        self.class_checkboxes = {}
        for cls_id, cls_name in self.global_names.items():
            chk = QCheckBox(cls_name.capitalize())
            # By default, check only the first 15 to avoid lag if many classes
            if cls_id < 15:
                chk.setChecked(True)
            chk.stateChanged.connect(self.update_detection_config)
            self.classes_layout.addWidget(chk)
            self.class_checkboxes[cls_id] = chk
            
        # 2. Rebuild Stats Labels
        while self.stats_layout.count():
            item = self.stats_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            
        self.class_stat_labels = {}
        for cls_id, cls_name in self.global_names.items():
            lbl = QLabel(f"{cls_name.capitalize()}: 0")
            lbl.setFont(QFont("Segoe UI", 12, QFont.Bold))
            lbl.setAlignment(Qt.AlignLeft)
            lbl.setStyleSheet("color: #a6e3a1; margin-bottom: 5px;")
            self.stats_layout.addWidget(lbl)
            self.class_stat_labels[cls_id] = lbl

    def refresh_region_cards(self):
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
            
            val_lbl = QLabel("0 Detected")
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
        for cls_id, chk in self.class_checkboxes.items():
            if chk.isChecked():
                active_classes.append(cls_id)
        
        if self.inference_thread is not None:
            self.inference_thread.set_detection_config(
                conf_val / 100.0, 
                active_classes, 
                float(dist_val),
                float(color_val),
                self.line_a_pct,
                self.line_b_pct,
                self.custom_regions,
                self.model_to_global
            )

    def open_dashboard(self):
        if self.dashboard_window is None:
            self.dashboard_window = DashboardWindow()
        self.dashboard_window.refresh_sessions()
        self.dashboard_window.show()

    def start_inference(self):
        source = self.source_input.text()
        if source.isdigit():
            source = int(source)

        if not self.active_model_paths:
            print("No models selected!")
            return
            
        # Generate new session ID
        db = self.SessionLocal()
        from sqlalchemy import desc
        last_session = db.query(InferenceSession).order_by(desc(InferenceSession.id)).first()
        if last_session:
            try:
                last_num = int(last_session.session_name.split('_')[1])
                new_num = last_num + 1
            except:
                new_num = 1
        else:
            new_num = 1
            
        session_name = f"inf_{new_num:03d}"
        
        new_session = InferenceSession(session_name=session_name, source=str(source))
        db.add(new_session)
        db.commit()
        db.close()

        self.inference_thread = InferenceThread(model_paths=self.active_model_paths, source=source, session_name=session_name)
        
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
        self.models_list.setEnabled(False)

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
        self.models_list.setEnabled(True)
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
        for cls_id, lbl in self.class_stat_labels.items():
            cnt = counts.get(cls_id, 0)
            cls_name = self.global_names.get(cls_id, "Unknown").capitalize()
            lbl.setText(f"{cls_name}: {cnt}")
        
        region_counts = stats.get("region_counts", {})
        for name, lbl in self.region_labels.items():
            cnt = region_counts.get(name, 0)
            lbl.setText(f"{cnt} Detected")

    def closeEvent(self, event):
        self.stop_inference()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SmartLensApp()
    window.show()
    sys.exit(app.exec())
