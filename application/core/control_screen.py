import os
from PyQt5.QtWidgets import QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from .camera_widget import CameraWidget
import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from keras.models import load_model
from .bluetooth_connection import BluetoothConnectionWidget

MODEL_RELATIVE_PATH = "../model/siamese_face_recognition_model.h5"
MODEL_ABS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_RELATIVE_PATH))

THRESHOLD = 0.55
MIN_IMAGES = 3
MIN_MEAN = 0.6

user_dir = "../APLICATIE/user_images/cropped_images"
person_scores = {}

class ControlScreen(QWidget):
    def __init__(self, switch_to_register_callback):
        super().__init__()

        self.setFocusPolicy(Qt.StrongFocus)
        self.pressed_keys = set()
        self.access_granted = False

        layout = QHBoxLayout()
        self.camera = CameraWidget()

        camera_layout = QVBoxLayout()
        camera_layout.setContentsMargins(20, 20, 20, 20)
        self.camera_frame = QFrame()
        self.camera_frame.setStyleSheet("background-color: white; border: 2px solid #ccc; border-radius: 10px;")
        camera_inner_layout = QVBoxLayout()
        camera_inner_layout.addWidget(self.camera)
        self.status_label = QLabel("Sistemul este in asteptare...")
        self.status_label.setFont(QFont("Arial", 12))
        camera_inner_layout.addWidget(self.status_label)
        self.camera_frame.setLayout(camera_inner_layout)
        camera_layout.addWidget(self.camera_frame)

        layout.addLayout(camera_layout)

        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setAlignment(Qt.AlignTop)

        self.start_btn = QPushButton("START")
        self.start_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 8px; padding: 10px;")
        self.start_btn.setFixedHeight(50)
        self.start_btn.clicked.connect(self.start_authentication)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setFont(QFont("Arial", 16, QFont.Bold))
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; border: none; border-radius: 8px; padding: 10px;")
        self.stop_btn.setFixedHeight(50)
        self.stop_btn.clicked.connect(self.reset_authentication)
        control_layout.addWidget(self.stop_btn)

        control_layout.addSpacing(30)

        self.controller_grid = QGridLayout()
        self.btn_up = self.create_btn("FATA - w ")
        self.btn_left = self.create_btn("STANGA - a ")
        self.btn_right = self.create_btn("DREAPTA - d ")
        self.btn_down = self.create_btn("SPATE - s ")
        self.btn_brake = self.create_btn("STOP - b ")

        self.btn_up.clicked.connect(lambda: self.bt_widget.send_serial_char('w'))
        self.btn_left.clicked.connect(lambda: self.bt_widget.send_serial_char('a'))
        self.btn_down.clicked.connect(lambda: self.bt_widget.send_serial_char('s'))
        self.btn_right.clicked.connect(lambda: self.bt_widget.send_serial_char('d'))
        self.btn_brake.clicked.connect(lambda: self.bt_widget.send_serial_char(' '))

        self.controller_grid.addWidget(self.btn_up, 0, 1)
        self.controller_grid.addWidget(self.btn_left, 1, 0)
        self.controller_grid.addWidget(self.btn_right, 1, 2)
        self.controller_grid.addWidget(self.btn_down, 2, 1)
        self.controller_grid.addWidget(self.btn_brake, 3, 1)
        control_layout.addLayout(self.controller_grid)
        control_layout.addSpacing(30)

        self.register_btn = QPushButton("Adauga utilizator nou")
        self.register_btn.setFont(QFont("Arial", 14))
        self.register_btn.setStyleSheet("background-color: #2196F3; color: white; border: none; border-radius: 8px; padding: 10px;")
        self.register_btn.setFixedHeight(50)
        self.register_btn.clicked.connect(switch_to_register_callback)
        control_layout.addWidget(self.register_btn)

        self.bt_widget = BluetoothConnectionWidget()
        control_layout.addWidget(self.bt_widget)

        layout.addLayout(control_layout)
        self.setLayout(layout)

    def create_btn(self, text):
        btn = QPushButton(text)
        btn.setFont(QFont("Arial", 14))
        btn.setFixedHeight(45)
        btn.setStyleSheet("border: 2px solid #888; border-radius: 10px; background-color: #eee;")
        btn.setEnabled(False)
        return btn

    def keyPressEvent(self, event):
        if not self.access_granted:
            return

        key = event.key()
        allowed_keys = {Qt.Key_W, Qt.Key_A, Qt.Key_S, Qt.Key_D, Qt.Key_B}
        if key not in allowed_keys:
            return

        if key == Qt.Key_W:
            self.activate_button(self.btn_up)
            self.bt_widget.send_serial_char('w')
        elif key == Qt.Key_A:
            self.activate_button(self.btn_left)
            self.bt_widget.send_serial_char('a')
        elif key == Qt.Key_S:
            self.activate_button(self.btn_down)
            self.bt_widget.send_serial_char('s')
        elif key == Qt.Key_D:
            self.activate_button(self.btn_right)
            self.bt_widget.send_serial_char('d')
        elif key == Qt.Key_B:
            self.activate_button(self.btn_brake)
            self.bt_widget.send_serial_char(' ')

    def keyReleaseEvent(self, event):
        if not self.access_granted:
            return

        key = event.key()
        if key == Qt.Key_W:
            self.deactivate_button(self.btn_up)
        elif key == Qt.Key_A:
            self.deactivate_button(self.btn_left)
        elif key == Qt.Key_S:
            self.deactivate_button(self.btn_down)
        elif key == Qt.Key_D:
            self.deactivate_button(self.btn_right)
        elif key == Qt.Key_B:
            self.deactivate_button(self.btn_brake)

    def activate_button(self, button):
        button.setStyleSheet("background-color: #888; border: 2px solid #555; border-radius: 10px;")

    def deactivate_button(self, button):
        button.setStyleSheet("border: 2px solid #888; border-radius: 10px; background-color: #eee;")

    def finish_authentication(self):
        if self.access_granted:
            self.status_label.setText("Acces permis.")
            for btn in [self.btn_up, self.btn_down, self.btn_left, self.btn_right, self.btn_brake]:
                btn.setEnabled(True)
        else:
            self.status_label.setText("Acces respins.")

    def reset_authentication(self):
        self.access_granted = False
        self.status_label.setText("Sistemul este in asteptare...")
        for btn in [self.btn_up, self.btn_down, self.btn_left, self.btn_right, self.btn_brake]:
            self.deactivate_button(btn)
            btn.setEnabled(False)

    def start_authentication(self):

        bt_conn = self.bt_widget.get_connection()
        if not bt_conn or not bt_conn.is_open:
            self.status_label.setText("Nu esti conectat la dispozitivul BT!")
            return

        self.status_label.setText("Se verifica identitatea...")
        frame = self.camera.capture_frame()
        if frame is not None:
            detector = MTCNN()
            rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.detect_faces(rgb_img)
            if len(results) == 0:
                self.status_label.setText("Nu s-a detectat nicio fata.")
                return
            x, y, w, h = results[0]['box']
            x, y = max(x, 0), max(y, 0)
            face = rgb_img[y:y + h, x:x + w]
            if face.size == 0:
                self.status_label.setText("Crop invalid!")
                return
            face = cv2.resize(face, (105, 105), interpolation=cv2.INTER_AREA)
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=0)

            if not hasattr(self, "face_model"):
                self.face_model = tf.keras.models.load_model(MODEL_ABS_PATH)

            person_scores.clear()
            best_mean = -1
            best_person = None

            for person in os.listdir(user_dir):
                person_dir = os.path.join(user_dir, person)
                if not os.path.isdir(person_dir):
                    continue
                scores = []
                for img_name in os.listdir(person_dir):
                    img_path = os.path.join(person_dir, img_name)
                    db_img = cv2.imread(img_path)
                    if db_img is None:
                        continue
                    db_img = cv2.cvtColor(db_img, cv2.COLOR_BGR2RGB)
                    db_img = cv2.resize(db_img, (105, 105), interpolation=cv2.INTER_AREA)
                    db_img = db_img.astype("float32") / 255.0
                    db_img = np.expand_dims(db_img, axis=0)
                    pred = self.face_model.predict([face, db_img])[0][0]
                    if pred > THRESHOLD:
                        scores.append(pred)
                if scores:
                    person_scores[person] = scores

            print("==== Rezultate scoruri peste prag ====")
            for person, scores in person_scores.items():
                print(f"{person}: {scores}")

            for person, scores in person_scores.items():
                if len(scores) >= MIN_IMAGES:
                    mean_score = np.mean(scores)
                    print(f"[CHECK] {person}: {len(scores)} imagini peste prag, medie scor: {mean_score:.4f}")
                    if mean_score > MIN_MEAN and mean_score > best_mean:
                        best_mean = mean_score
                        best_person = person

            if best_person is not None:
                self.access_granted = True
                self.status_label.setText(f"Acces permis: {best_person} (medie scor: {best_mean:.2f})")
                print(f"[ACCESS] Acces permis pentru {best_person} (medie scor: {best_mean:.4f})")
            else:
                self.access_granted = False
                self.status_label.setText("Acces respins!")
                print("[ACCESS] Nicio persoana cu minim 3 imagini peste prag si medie > 0.6!")

        self.finish_authentication()

