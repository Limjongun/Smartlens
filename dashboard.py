import os
import sys
import pandas as pd
import sqlite3
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QComboBox, QLabel, QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox, QFileDialog)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

MODERN_STYLE = """
QMainWindow {
    background-color: #11111b; 
}
QLabel {
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QComboBox, QPushButton {
    background-color: #181825;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 5px;
    padding: 8px;
    font-size: 14px;
}
QComboBox:focus {
    border: 1px solid #8caaee;
}
QPushButton#exportBtn {
    background-color: #a6d189;
    color: #232634;
    font-weight: bold;
}
QPushButton#exportBtn:hover {
    background-color: #81c8be;
}
QTableWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    border: 1px solid #313244;
    border-radius: 8px;
    gridline-color: #313244;
}
QHeaderView::section {
    background-color: #181825;
    color: #8caaee;
    padding: 4px;
    border: 1px solid #313244;
    font-weight: bold;
}
"""

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e2e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e2e')
        self.axes.tick_params(colors='#cdd6f4')
        self.axes.xaxis.label.set_color('#cdd6f4')
        self.axes.yaxis.label.set_color('#cdd6f4')
        self.axes.title.set_color('#8caaee')
        for spine in self.axes.spines.values():
            spine.set_edgecolor('#313244')
        super(MplCanvas, self).__init__(self.fig)

class DashboardWindow(QMainWindow):
    def __init__(self, db_path="traffic.db"):
        super().__init__()
        self.setWindowTitle("Smart Lens - Data Analytics Dashboard")
        self.resize(1200, 800)
        self.setStyleSheet(MODERN_STYLE)
        self.db_path = db_path
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Header Controls
        header_layout = QHBoxLayout()
        title_lbl = QLabel("Analytics Dashboard")
        title_lbl.setFont(QFont("Segoe UI", 24, QFont.Bold))
        header_layout.addWidget(title_lbl)
        
        header_layout.addStretch()
        
        header_layout.addWidget(QLabel("Select Session:"))
        self.session_combo = QComboBox()
        self.session_combo.setMinimumWidth(200)
        self.session_combo.currentIndexChanged.connect(self.load_session_data)
        header_layout.addWidget(self.session_combo)
        
        self.export_btn = QPushButton("Export to CSV")
        self.export_btn.setObjectName("exportBtn")
        self.export_btn.clicked.connect(self.export_to_csv)
        header_layout.addWidget(self.export_btn)
        
        main_layout.addLayout(header_layout)
        
        # Charts Layout
        charts_layout = QHBoxLayout()
        
        self.bar_chart = MplCanvas(self, width=5, height=4, dpi=100)
        charts_layout.addWidget(self.bar_chart)
        
        self.pie_chart = MplCanvas(self, width=5, height=4, dpi=100)
        charts_layout.addWidget(self.pie_chart)
        
        main_layout.addLayout(charts_layout, stretch=2)
        
        # Table Layout
        table_lbl = QLabel("Recent Violations (Stopped in Restricted Areas)")
        table_lbl.setFont(QFont("Segoe UI", 16, QFont.Bold))
        table_lbl.setStyleSheet("color: #e78284;")
        main_layout.addWidget(table_lbl)
        
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Track ID", "Class", "Region", "Image Path", "Session"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        main_layout.addWidget(self.table, stretch=1)
        
        self.refresh_sessions()
        
    def refresh_sessions(self):
        self.session_combo.blockSignals(True)
        self.session_combo.clear()
        
        if not os.path.exists(self.db_path):
            self.session_combo.addItem("No Data")
            self.session_combo.blockSignals(False)
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            sessions = pd.read_sql_query("SELECT session_name, start_time FROM inference_sessions ORDER BY id DESC", conn)
            conn.close()
            
            if sessions.empty:
                self.session_combo.addItem("No Data")
            else:
                for _, row in sessions.iterrows():
                    self.session_combo.addItem(f"{row['session_name']} ({row['start_time']})", row['session_name'])
        except Exception as e:
            print("Error loading sessions:", e)
            self.session_combo.addItem("Error Loading Data")
            
        self.session_combo.blockSignals(False)
        self.load_session_data()
        
    def load_session_data(self):
        if self.session_combo.currentText() in ["No Data", "Error Loading Data", ""]:
            return
            
        session_name = self.session_combo.currentData()
        if not session_name:
            return
            
        try:
            conn = sqlite3.connect(self.db_path)
            logs_df = pd.read_sql_query(f"SELECT * FROM vehicle_logs WHERE session_name='{session_name}'", conn)
            violations_df = pd.read_sql_query(f"SELECT * FROM violations WHERE session_name='{session_name}' ORDER BY timestamp DESC", conn)
            conn.close()
            
            self.update_bar_chart(logs_df)
            self.update_pie_chart(logs_df)
            self.update_table(violations_df)
            
        except Exception as e:
            print("Error loading session data:", e)
            
    def update_bar_chart(self, df):
        self.bar_chart.axes.clear()
        self.bar_chart.axes.set_facecolor('#1e1e2e')
        if not df.empty:
            counts = df['vehicle_class'].value_counts()
            counts.plot(kind='bar', ax=self.bar_chart.axes, color='#8caaee', edgecolor='#313244')
            self.bar_chart.axes.set_title("Vehicle Class Distribution")
            self.bar_chart.axes.set_xlabel("Class")
            self.bar_chart.axes.set_ylabel("Count")
            for tick in self.bar_chart.axes.get_xticklabels():
                tick.set_rotation(45)
        else:
            self.bar_chart.axes.text(0.5, 0.5, 'No Data', color='#cdd6f4', ha='center', va='center')
        self.bar_chart.draw()
        
    def update_pie_chart(self, df):
        self.pie_chart.axes.clear()
        self.pie_chart.axes.set_facecolor('#1e1e2e')
        if not df.empty and 'color' in df.columns:
            # Filter out calculating
            color_df = df[~df['color'].str.contains('Calculating|Too far', na=False)]
            counts = color_df['color'].value_counts()
            if not counts.empty:
                counts.plot(kind='pie', ax=self.pie_chart.axes, autopct='%1.1f%%', textprops={'color':"#cdd6f4"})
                self.pie_chart.axes.set_title("Vehicle Color Distribution")
                self.pie_chart.axes.set_ylabel("")
            else:
                self.pie_chart.axes.text(0.5, 0.5, 'No Color Data', color='#cdd6f4', ha='center', va='center')
        else:
            self.pie_chart.axes.text(0.5, 0.5, 'No Data', color='#cdd6f4', ha='center', va='center')
        self.pie_chart.draw()
        
    def update_table(self, df):
        self.table.setRowCount(0)
        for _, row in df.iterrows():
            r = self.table.rowCount()
            self.table.insertRow(r)
            self.table.setItem(r, 0, QTableWidgetItem(str(row['timestamp'])))
            self.table.setItem(r, 1, QTableWidgetItem(str(row['track_id'])))
            self.table.setItem(r, 2, QTableWidgetItem(str(row['vehicle_class'])))
            self.table.setItem(r, 3, QTableWidgetItem(str(row['region_name'])))
            self.table.setItem(r, 4, QTableWidgetItem(str(row['image_path'])))
            self.table.setItem(r, 5, QTableWidgetItem(str(row['session_name'])))
            
    def export_to_csv(self):
        if self.session_combo.currentText() in ["No Data", "Error Loading Data", ""]:
            QMessageBox.warning(self, "Export", "No session selected or no data available.")
            return
            
        session_name = self.session_combo.currentData()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Data", f"export_{session_name}.csv", "CSV Files (*.csv)")
        
        if file_path:
            try:
                conn = sqlite3.connect(self.db_path)
                logs_df = pd.read_sql_query(f"SELECT * FROM vehicle_logs WHERE session_name='{session_name}'", conn)
                conn.close()
                logs_df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Data exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export: {e}")

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = DashboardWindow()
    window.show()
    sys.exit(app.exec())
