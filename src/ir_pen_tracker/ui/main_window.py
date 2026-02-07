import cv2
import threading
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QStatusBar, QSplitter, QProgressBar
from PyQt5.QtCore import Qt, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap

from ir_pen_tracker.logic.app_controller import AppController

class ScalableImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self._pixmap = None
        
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.update_display()
        
    def resizeEvent(self, event):
        self.update_display()
        super().resizeEvent(event)
        
    def update_display(self):
        if self._pixmap and not self._pixmap.isNull():
            # Scale pixmap to fit label size while maintaining aspect ratio
            scaled = self._pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            super().setPixmap(scaled)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Pen Tracker (PyQt Modular)")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main Layout (Vertical: Views + Controls)
        main_layout = QVBoxLayout(central_widget)
        
        # Views Layout (Horizontal Splitter: Color | (IR + Ortho))
        views_splitter = QSplitter(Qt.Horizontal)
        
        # Left: Color View
        self.color_label = ScalableImageLabel()
        views_splitter.addWidget(self.color_label)
        
        # Right: Split Vertical (IR + Ortho)
        right_splitter = QSplitter(Qt.Vertical)
        self.ir_label = ScalableImageLabel()
        self.ortho_label = ScalableImageLabel()
        
        right_splitter.addWidget(self.ir_label)
        right_splitter.addWidget(self.ortho_label)
        
        views_splitter.addWidget(right_splitter)
        
        # Set initial stretch factors (e.g., Color gets more space)
        views_splitter.setStretchFactor(0, 2)
        views_splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(views_splitter, stretch=1)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.btn_record = QPushButton("Start Recording")
        self.btn_record.clicked.connect(self.on_record_clicked)
        controls_layout.addWidget(self.btn_record)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        controls_layout.addWidget(self.progress_bar)
        
        self.btn_calibrate = QPushButton("Calibrate")
        self.btn_calibrate.clicked.connect(self.on_calibrate_clicked)
        controls_layout.addWidget(self.btn_calibrate)
        
        self.btn_confirm = QPushButton("Confirm Calibration (Space)")
        self.btn_confirm.clicked.connect(self.on_confirm_clicked)
        controls_layout.addWidget(self.btn_confirm)
        
        main_layout.addLayout(controls_layout)
        
        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_perf = QLabel("Perf: -")
        self.status_bar.addPermanentWidget(self.lbl_fps)
        self.status_bar.addPermanentWidget(self.lbl_perf)
        
        # Controller
        self.controller = AppController()
        self.controller.frames_ready.connect(self.update_frames)
        self.controller.status_update.connect(self.update_status)
        self.controller.rec_manager.saving_started.connect(self.on_save_started)
        self.controller.rec_manager.saving_progress.connect(self.on_save_progress)
        self.controller.rec_manager.saving_finished.connect(self.on_save_finished)
        
        # Render timer and shared frame buffer
        self._frame_lock = threading.Lock()
        self._latest_frames = None
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(33)
        self.render_timer.timeout.connect(self.on_render_tick)
        self.render_timer.start()
        
        # Start
        self.controller.start()
        
    def closeEvent(self, event):
        self.controller.stop()
        super().closeEvent(event)
        
    @pyqtSlot(dict)
    def update_frames(self, frames):
        with self._frame_lock:
            self._latest_frames = frames
    
    def on_render_tick(self):
        with self._frame_lock:
            frames = self._latest_frames
        if not frames:
            return
        if "color" in frames:
            self.display_image(frames["color"], self.color_label)
        if "ir" in frames:
            self.display_image(frames["ir"], self.ir_label)
        if "ortho" in frames:
            self.display_image(frames["ortho"], self.ortho_label)

    def display_image(self, img_np, label_widget):
        if img_np is None:
            return
        
        # Convert to QImage
        # Assuming BGR for Color/IR/Ortho (OpenCV default)
        # Check channels
        if len(img_np.shape) == 2:
            h, w = img_np.shape
            ch = 1
            fmt = QImage.Format_Grayscale8
            bytes_per_line = w
        else:
            h, w, ch = img_np.shape
            fmt = QImage.Format_RGB888
            bytes_per_line = ch * w
            # Convert BGR to RGB in place or copy? 
            # Controller sends BGR, so we need to swap for Qt
            # Note: Doing color conversion here on UI thread might be costly for 3 streams.
            # Ideally Controller sends RGB. But OpenCV draws on BGR.
            # Let's convert here.
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            
        qt_image = QImage(img_np.data, w, h, bytes_per_line, fmt)
        # Copy to decouple from numpy buffer reuse
        pixmap = QPixmap.fromImage(qt_image.copy())
        label_widget.setPixmap(pixmap)
        
    @pyqtSlot(str)
    def update_status(self, text):
        if text.startswith("Input FPS:"):
            self.lbl_fps.setText(text)
            return
        if text.startswith("Perf avg(ms):"):
            self.lbl_perf.setText(text)
            return
        self.status_bar.showMessage(text)
        
    def on_record_clicked(self):
        self.controller.toggle_recording()
        # Note: We don't have direct access to is_recording state here easily unless we track it 
        # or controller emits state change.
        # But we can rely on status updates or just toggle text.
        # Better: Controller emits signal with recording state.
        # For now, simplistic toggle text update based on next click or assume sync.
        # Let's leave text static or update based on status msg if needed.
        # Ideally: self.controller.rec_manager.is_recording (thread safe?)
        if self.controller.rec_manager.is_recording:
            self.btn_record.setText("Stop Recording")
            self.btn_record.setStyleSheet("background-color: red; color: white;")
        else:
            self.btn_record.setText("Start Recording")
            self.btn_record.setStyleSheet("")
    
    @pyqtSlot(int)
    def on_save_started(self, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_record.setEnabled(False)
    
    @pyqtSlot(int)
    def on_save_progress(self, value):
        self.progress_bar.setValue(value)
    
    @pyqtSlot()
    def on_save_finished(self):
        self.progress_bar.setVisible(False)
        self.btn_record.setEnabled(True)
            
    def on_calibrate_clicked(self):
        self.controller.start_calibration()
        
    def on_confirm_clicked(self):
        self.controller.confirm_calibration()
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            if self.controller.state == "CALIBRATE":
                self.controller.confirm_calibration()
            elif self.controller.state == "MAIN":
                self.on_record_clicked()
        elif event.key() == Qt.Key_C:
            self.controller.start_calibration()
