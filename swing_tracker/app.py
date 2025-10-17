import sys
from pathlib import Path

from PyQt5 import QtWidgets

from .controller import SwingTrackerController
from .view import SwingTrackerWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    root_path = Path(__file__).resolve().parent.parent
    controller = SwingTrackerController(root_path)
    window = SwingTrackerWindow(controller)
    window.show()
    sys.exit(app.exec_())
