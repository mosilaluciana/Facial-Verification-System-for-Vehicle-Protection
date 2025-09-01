from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2

class CameraWidget(QLabel):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.setPixmap(QPixmap.fromImage(qt_image).scaled(
                320, 240, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def close_camera(self):
        self.cap.release()

    # camera_widget.py

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None

