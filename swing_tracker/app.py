import sys

from PyQt5 import QtWidgets

from .gui import SwingTrackerWindow


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = SwingTrackerWindow()
    window.show()
    sys.exit(app.exec_())
