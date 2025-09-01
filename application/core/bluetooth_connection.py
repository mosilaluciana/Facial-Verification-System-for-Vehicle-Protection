import serial
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QMessageBox
)

class BluetoothConnectionWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.bt_connection = None
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()

        self.label = QLabel("Port COM:")
        self.com_input = QLineEdit()
        self.com_input.setPlaceholderText("Ex: COM7")

        self.connect_button = QPushButton("ConectareBT")
        self.connect_button.clicked.connect(self.connect_bt)

        layout.addWidget(self.label)
        layout.addWidget(self.com_input)
        layout.addWidget(self.connect_button)

        self.setLayout(layout)

    def connect_bt(self):
        port = self.com_input.text().strip()
        if not port:
            QMessageBox.warning(self, "Eroare", "Introdu un port COM valid.")
            return

        try:
            self.bt_connection = serial.Serial(port, 9600, timeout=1)
            QMessageBox.information(self, "Succes", f"Conectat la {port}")
        except Exception as e:
            QMessageBox.critical(self, "Eroare", f"Conectare eșuata:\n{e}")

    def get_connection(self):
        return self.bt_connection

    def send_serial_char(self, char):
        try:
            if self.bt_connection and self.bt_connection.is_open:
                self.bt_connection.write((char + '\r').encode('utf-8'))
            else:
                QMessageBox.warning(self, "Conexiune", "Bluetooth nu este conectat.")
        except Exception as e:
            QMessageBox.warning(self, "Eroare", f"Eroare la trimitere: {e}")
