import sys
from pathlib import Path
import logging

from PyQt5 import QtWidgets

from .controller import SwingTrackerController
from .view import SwingTrackerWindow

# Runs the GUI
def main() -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    app = QtWidgets.QApplication(sys.argv)
    root_path = Path(__file__).resolve().parent.parent
    controller = SwingTrackerController(root_path)
    window = SwingTrackerWindow(controller)
    window.show()
    sys.exit(app.exec_())
