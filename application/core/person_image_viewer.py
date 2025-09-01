from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel, QFileDialog, QScrollArea, QHBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os
import shutil

from utils.mtcnn_crop import crop_and_save_face

class PersonImageViewer(QWidget):
    def __init__(self, person_name, mode, switch_back_callback):
        super().__init__()
        self.person_name = person_name
        self.mode = mode
        self.switch_back_callback = switch_back_callback

        self.original_folder = os.path.join("user_images", "original_images", self.person_name)
        self.cropped_folder  = os.path.join("user_images", "cropped_images", self.person_name)
        self.image_folder = self.original_folder if self.mode == "original" else self.cropped_folder

        self.layout = QVBoxLayout()

        if self.mode == "original":
            self.upload_button = QPushButton(f"Adauga imagini pentru {self.person_name}")
            self.upload_button.setStyleSheet("padding: 10px; font-size: 14px; border-radius: 8px; background-color: #4CAF50; color: white;")
            self.upload_button.clicked.connect(self.load_images)
            self.layout.addWidget(self.upload_button)

        btn_row = QHBoxLayout()
        self.btn_cropped = QPushButton("Cropped")
        self.btn_cropped.setStyleSheet("background-color: #388E3C; color: white; font-size: 14px; border-radius: 8px; padding: 8px;")
        self.btn_cropped.clicked.connect(self.show_cropped)
        btn_row.addWidget(self.btn_cropped)

        self.btn_original = QPushButton("Original")
        self.btn_original.setStyleSheet("background-color: #1976D2; color: white; font-size: 14px; border-radius: 8px; padding: 8px;")
        self.btn_original.clicked.connect(self.show_original)
        btn_row.addWidget(self.btn_original)

        self.layout.addLayout(btn_row)

        self.grid_layout = QGridLayout()
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_content.setLayout(self.grid_layout)
        self.scroll_area.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll_area)


        self.back_btn = QPushButton("inapoi la lista de persoane")
        self.back_btn.setStyleSheet("padding: 10px; font-size: 14px; border-radius: 8px; background-color: #606060; color: white;")
        self.back_btn.clicked.connect(self.switch_back_callback)
        self.layout.addWidget(self.back_btn)

        self.setLayout(self.layout)
        self.display_images()

    def load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Selecteaza imagini", "", "Images (*.png *.jpg *.jpeg)")
        os.makedirs(self.original_folder, exist_ok=True)
        os.makedirs(self.cropped_folder, exist_ok=True)

        flag_path = os.path.join(self.cropped_folder, "added.flag")
        if os.path.exists(flag_path):
            os.remove(flag_path)
        for file_path in files:
            filename = os.path.basename(file_path)
            destination = os.path.join(self.original_folder, filename)
            shutil.copy(file_path, destination)
            crop_and_save_face(destination, self.cropped_folder)
        self.display_images()

    def display_images(self):

        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()


        if not os.path.exists(self.image_folder):
            return

        row = col = 0
        for image_file in os.listdir(self.image_folder):
            image_path = os.path.join(self.image_folder, image_file)
            if image_path.lower().endswith((".png", ".jpg", ".jpeg")):
                pixmap = QPixmap(image_path).scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label = QLabel()
                label.setPixmap(pixmap)
                self.grid_layout.addWidget(label, row, col)
                col += 1
                if col >= 4:
                    col = 0
                    row += 1

    def show_original(self):
        self.mode = "original"
        self.image_folder = self.original_folder
        self.display_images()

    def show_cropped(self):
        self.mode = "cropped"
        self.image_folder = self.cropped_folder
        self.display_images()
