import os

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QHBoxLayout, QLabel

class FineTuneDialog(QDialog):
    def __init__(self, python_exe, script_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Fine-tuning în timp real")
        self.resize(800, 500)
        self.layout = QVBoxLayout(self)

        self.info_label = QLabel("Se ruleaza fine-tuning... Așteptati.")
        self.layout.addWidget(self.info_label)

        self.textedit = QTextEdit()
        self.textedit.setReadOnly(True)
        self.textedit.setStyleSheet("font-family: Consolas, monospace; font-size: 13px;")
        self.layout.addWidget(self.textedit)

        self.btn_close = QPushButton("Închide")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)
        self.layout.addWidget(self.btn_close)

        self.process = QProcess(self)
        self.process.setWorkingDirectory(os.path.abspath("../ModelSiamese"))

        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.handle_finished)

        self.process.start(python_exe, [script_path])

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.textedit.append(data)
        self.textedit.moveCursor(self.textedit.textCursor().End)

    def handle_stderr(self):
        data = self.process.readAllStandardError().data().decode()
        self.textedit.append(f"<span style='color:red'>{data}</span>")
        self.textedit.moveCursor(self.textedit.textCursor().End)

    def handle_finished(self, exitCode, exitStatus):
        if exitCode == 0:
            self.info_label.setText("Fine-tuning finalizat cu succes!")
        else:
            self.info_label.setText(" Eroare fine-tuning.")
        self.btn_close.setEnabled(True)
