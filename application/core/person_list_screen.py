import os
import shutil
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QScrollArea, QHBoxLayout, QMessageBox, QLabel
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton

from utils.result_fine_tune_dialog import FineTuneDialog


class PersonListScreen(QWidget):
    def __init__(self, open_person_callback, switch_back_callback):
        super().__init__()
        self.open_person_callback = open_person_callback
        self.switch_back_callback = switch_back_callback

        self.image_base_folder = "user_images"
        self.orig_base = os.path.join(self.image_base_folder, "original_images")
        self.crop_base = os.path.join(self.image_base_folder, "cropped_images")
        os.makedirs(self.orig_base, exist_ok=True)
        os.makedirs(self.crop_base, exist_ok=True)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll_area.setWidget(self.scroll_content)

        self.layout.addWidget(self.scroll_area)

        self.add_user_button = QPushButton("Adauga persoana noua")
        self.add_user_button.setStyleSheet(
            "padding: 10px; font-size: 14px; border-radius: 8px; background-color: #4CAF50; color: white;"
        )
        self.add_user_button.clicked.connect(self.create_new_person)
        self.layout.addWidget(self.add_user_button)


        self.back_btn = QPushButton("inapoi la ecranul principal")
        self.back_btn.setStyleSheet(
            "padding: 10px; font-size: 14px; border-radius: 8px; background-color: #606060; color: white;"
        )
        self.back_btn.clicked.connect(self.switch_back_callback)
        self.layout.addWidget(self.back_btn)

        self.finetune_btn = QPushButton("Adauga in sistem (fine-tune model cu toate persoanele)")
        self.finetune_btn.setStyleSheet(
            "padding: 10px; font-size: 14px; border-radius: 8px; background-color: #FFA726; color: black;"
        )
        self.finetune_btn.clicked.connect(self.run_fine_tuning)
        self.layout.addWidget(self.finetune_btn)

        self.refresh_list()

    def refresh_list(self):

        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        def make_del_callback(person):
            return lambda: self.confirm_delete_person(person)

        def make_callback(mode, person):
            return lambda: self.open_person_callback(person, mode)

        for person in sorted(os.listdir(self.orig_base)):
            orig_path = os.path.join(self.orig_base, person)
            crop_path = os.path.join(self.crop_base, person)
            if os.path.isdir(orig_path):
                row_widget = QWidget()
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(5, 5, 5, 5)
                row_layout.setSpacing(10)

                name_label = QLabel(f"{person}")
                row_layout.addWidget(name_label)


                btn_cropped = QPushButton("Cropped")
                btn_cropped.setStyleSheet(
                    "padding: 8px 12px; font-size: 14px; border-radius: 6px; background-color: #388E3C; color: white;"
                )
                btn_cropped.clicked.connect(make_callback("cropped", person))
                row_layout.addWidget(btn_cropped)

                btn_original = QPushButton("Original")
                btn_original.setStyleSheet(
                    "padding: 8px 12px; font-size: 14px; border-radius: 6px; background-color: #1976D2; color: white;"
                )
                btn_original.clicked.connect(make_callback("original", person))
                row_layout.addWidget(btn_original)

                del_btn = QPushButton("Elimina persoana")
                del_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #f44336;
                        color: white;
                        font-size: 14px;
                        border-radius: 8px;
                    }
                    QPushButton:hover {
                        background-color: #d32f2f;
                    }
                """)
                del_btn.clicked.connect(make_del_callback(person))
                row_layout.addWidget(del_btn)

                row_widget.setLayout(row_layout)
                self.scroll_layout.addWidget(row_widget)
                self.scroll_layout.addWidget(row_widget)

    def confirm_delete_person(self, person_name):
        reply = QMessageBox.question(
            self,
            "Confirmare stergere",
            f"Sigur doresti sa stergi persoana '{person_name}'?\nAceasta actiune este ireversibila.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            orig_folder = os.path.join(self.orig_base, person_name)
            crop_folder = os.path.join(self.crop_base, person_name)
            shutil.rmtree(orig_folder, ignore_errors=True)
            shutil.rmtree(crop_folder, ignore_errors=True)
            self.refresh_list()

    def create_new_person(self):
        from PyQt5.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Nume persoana noua", "Introdu numele:")
        if ok and name.strip():
            orig_folder = os.path.join(self.orig_base, name.strip())
            crop_folder = os.path.join(self.crop_base, name.strip())
            if not os.path.exists(orig_folder) and not os.path.exists(crop_folder):
                os.makedirs(orig_folder)
                os.makedirs(crop_folder)
                self.refresh_list()
            else:
                QMessageBox.warning(self, "Eroare", "Exista deja o persoana cu acest nume.")

    def run_fine_tuning(self):
        python_path = "../ModelSiamese/venv/Scripts/python.exe"
        path_script = "../ModelSiamese/fine_tune.py"
        dlg = FineTuneDialog(python_path, path_script, parent=self)
        dlg.exec_()
