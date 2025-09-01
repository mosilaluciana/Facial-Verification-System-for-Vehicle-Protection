from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QLineEdit, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

class RegisterScreen(QWidget):
    def __init__(self, switch_back_callback, switch_to_upload_callback):
        super().__init__()
        self.switch_back_callback = switch_back_callback
        self.switch_to_upload_callback = switch_to_upload_callback

        layout = QVBoxLayout()
        layout.setSpacing(20)
        layout.setContentsMargins(60, 60, 60, 60)
        layout.setAlignment(Qt.AlignCenter)

        self.label = QLabel(" Introduceti parola pentru a adauga un nou utilizator:")
        self.label.setFont(QFont("Arial", 16))
        self.label.setAlignment(Qt.AlignCenter)

        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setPlaceholderText("Parola...")
        self.password_input.setFixedHeight(40)
        self.password_input.setStyleSheet(
            "font-size: 14px; padding: 8px; border-radius: 6px; border: 1px solid gray;"
        )

        self.submit_btn = QPushButton("Verifica parola")
        self.submit_btn.setFixedHeight(40)
        self.submit_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-size: 14px; border: none; border-radius: 6px;"
        )
        self.submit_btn.clicked.connect(self.check_password)

        self.back_btn = QPushButton("Inapoi")
        self.back_btn.setFixedHeight(40)
        self.back_btn.setStyleSheet(
            "background-color: #606060; color: white; font-size: 14px; border: none; border-radius: 6px;"
        )
        self.back_btn.clicked.connect(self.switch_back_callback)

        layout.addWidget(self.label)
        layout.addWidget(self.password_input)
        layout.addWidget(self.submit_btn)
        layout.addWidget(self.back_btn)

        self.setLayout(layout)

    def check_password(self):
        if self.password_input.text() == "1234":
            self.switch_to_upload_callback()

