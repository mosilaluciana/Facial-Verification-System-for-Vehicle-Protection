from PyQt5.QtWidgets import QMainWindow, QStackedWidget
from .control_screen import ControlScreen
from .register_screen import RegisterScreen
from .person_list_screen import PersonListScreen
from .person_image_viewer import PersonImageViewer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Masina Inteligenta")
        self.setGeometry(300, 200, 800, 500)

        self.stack = QStackedWidget()

        self.control_screen = ControlScreen(self.show_register_screen)
        self.stack.addWidget(self.control_screen)


        self.person_list_screen = PersonListScreen(
            open_person_callback=self.open_person_viewer,
            switch_back_callback=self.show_control_screen
        )

        self.stack.addWidget(self.person_list_screen)

        self.register_screen = RegisterScreen(self.show_control_screen, self.show_person_list)
        self.stack.addWidget(self.register_screen)

        self.stack.setCurrentWidget(self.control_screen)

        self.setCentralWidget(self.stack)



    def show_register_screen(self):
        self.stack.setCurrentWidget(self.register_screen)

    def show_control_screen(self):
        self.stack.setCurrentWidget(self.control_screen)

    def show_person_list(self):
        self.person_list_screen.refresh_list()
        self.stack.setCurrentWidget(self.person_list_screen)

    def open_person_viewer(self, person_name, mode):
        self.person_image_viewer = PersonImageViewer(person_name, mode, self.show_person_list)
        self.stack.addWidget(self.person_image_viewer)
        self.stack.setCurrentWidget(self.person_image_viewer)

    def closeEvent(self, event):
        self.control_screen.camera.close_camera()
        event.accept()
