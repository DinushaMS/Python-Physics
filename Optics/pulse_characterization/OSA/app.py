import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QComboBox, QPushButton
)
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyvisa


# -------------------- Matplotlib Canvas --------------------
class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.figure = Figure(facecolor="#2b2b2b")
        self.ax = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)

        self.ax.set_facecolor("#2b2b2b")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

    def plot_data(self, mode):
        self.ax.clear()

        x = np.linspace(0, 10, 100)

        if mode == "Sine":
            y = np.sin(x)
        elif mode == "Cosine":
            y = np.cos(x)
        else:
            y = x

        self.ax.plot(x, y, color="#00c8ff", linewidth=2)
        self.ax.set_title(f"{mode} Plot", color="white")
        self.ax.tick_params(colors="white")
        self.ax.set_facecolor("#2b2b2b")

        for spine in self.ax.spines.values():
            spine.set_color("white")

        self.draw()


# -------------------- Main Window --------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dark PyQt + PyPlot App")
        self.resize(600, 500)

        layout = QVBoxLayout()

        # Drop-down (replaces list box)
        self.combo = QComboBox()
        #self.combo.addItems(["Sine", "Cosine", "Linear"])

        # Connect button
        self.button = QPushButton("Connect")
        self.button.clicked.connect(self.on_connect)

        # Plot area
        self.plot_canvas = PlotCanvas(self)

        layout.addWidget(self.combo)
        layout.addWidget(self.button)
        layout.addWidget(self.plot_canvas)

        self.setLayout(layout)
        self.instrument = None
        self.rm = None

    def on_connect(self):
        self.rm = pyvisa.ResourceManager()
        resource = self.combo.currentText()
        self.instrument = self.rm.open_resource(resource)
        self.instrument
        self.instrument.write("*IDN?")
        response = self.instrument.read()
        print(f"Instrument Response: {response}")
        #self.plot_canvas.plot_data(mode)

    def closeEvent(self, event):
        self.cleanup()
        event.accept()

    def cleanup(self):
        print("Stopping threads...")
        if self.instrument != None:
            self.instrument.close()
        if self.rm != None:
            self.rm.close()        

# -------------------- Dark Theme --------------------
def set_dark_theme(app):
    palette = QPalette()

    palette.setColor(QPalette.Window, QColor(43, 43, 43))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(30, 30, 30))
    palette.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.HighlightedText, Qt.black)

    app.setPalette(palette)

# -------------------- Connect to pyvis and get list of instrumnents --------------------
def get_instruments():
    rm = pyvisa.ResourceManager()
    instruments = []
    resources = []
    for resource in rm.list_resources():
        instrument = rm.open_resource(resource)
        instrument.write("*IDN?")
        response = instrument.read()
        #print(f"Instrument Response: {response}")
        if response != "":
            instruments.append(response.replace("\n", ""))
            resources.append(resource)
        instrument.close()
        rm.close()
    return instruments, resources


# -------------------- Run App --------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    set_dark_theme(app)

    window = MainWindow()
    app.aboutToQuit.connect(window.cleanup)
    window.show()
    instruments, resources = get_instruments()
    for item in resources:
        window.combo.addItem(item)

    sys.exit(app.exec_())

