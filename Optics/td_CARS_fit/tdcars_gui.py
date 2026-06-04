
import sys
import numpy as np

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton,
    QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTextEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from TdCARS2 import TdCARS


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.cars = None

        self.setWindowTitle("td-CARS Analysis")
        self.resize(1400, 900)

        self.canvas = MplCanvas()

        self.load_btn = QPushButton("Load Data")
        self.transient_btn = QPushButton("Plot Transient")
        self.contour_btn = QPushButton("Contour Plot")
        self.sim_btn = QPushButton("Run Simulation")
        self.t2_btn = QPushButton("Calculate T2")

        self.nuR_edit = QLineEdit("730,800")
        self.T2_edit = QLineEdit("377,300")
        self.A_edit = QLineEdit("1.9e25,0")

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)

        self._build_layout()
        self._connect()

    def _build_layout(self):

        left = QVBoxLayout()

        left.addWidget(self.load_btn)

        left.addWidget(QLabel("νR (cm⁻¹)"))
        left.addWidget(self.nuR_edit)

        left.addWidget(QLabel("T₂ (fs)"))
        left.addWidget(self.T2_edit)

        left.addWidget(QLabel("Amplitude"))
        left.addWidget(self.A_edit)

        left.addWidget(self.sim_btn)
        left.addWidget(self.transient_btn)
        left.addWidget(self.contour_btn)
        left.addWidget(self.t2_btn)

        left.addStretch()
        left.addWidget(self.result_box)

        left_widget = QWidget()
        left_widget.setLayout(left)

        layout = QHBoxLayout()
        layout.addWidget(left_widget, 1)
        layout.addWidget(self.canvas, 3)

        central = QWidget()
        central.setLayout(layout)

        self.setCentralWidget(central)

    def _connect(self):
        self.load_btn.clicked.connect(self.load_data)
        self.transient_btn.clicked.connect(self.plot_transient)
        self.contour_btn.clicked.connect(self.plot_contour)
        self.sim_btn.clicked.connect(self.run_simulation)
        self.t2_btn.clicked.connect(self.calculate_t2)

    def load_data(self):

        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open td-CARS File",
            "",
            "*.dat"
        )

        if not fname:
            return

        self.cars = TdCARS.from_file(fname)
        self.result_box.append(f"Loaded sample: {self.cars.sample}")

    def plot_transient(self):

        if self.cars is None:
            return

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        ax.semilogy(
            self.cars.td_arr,
            self.cars.signal_exp_corrected,
            "ko"
        )

        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Signal")
        ax.set_title("Experimental Transient")

        self.canvas.draw()

    def run_simulation(self):

        if self.cars is None:
            return

        self.cars.nuR = np.array(
            [float(x) for x in self.nuR_edit.text().split(",")]
        )

        self.cars.T2 = np.array(
            [float(x) for x in self.T2_edit.text().split(",")]
        )

        self.cars.A = np.array(
            [float(x) for x in self.A_edit.text().split(",")]
        )

        td, signal = self.cars.simulate_cars()

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        ax.semilogy(
            self.cars.td_arr,
            self.cars.signal_exp_corrected,
            "ko",
            label="Experiment"
        )

        ax.semilogy(
            td,
            signal,
            "r-",
            label="Simulation"
        )

        ax.legend()
        self.canvas.draw()

    def plot_contour(self):

        if self.cars is None:
            return

        Z, _ = self.cars.get_spectra_contour()

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)

        im = ax.imshow(
            Z,
            aspect="auto",
            origin="lower"
        )

        self.canvas.fig.colorbar(im, ax=ax)
        ax.set_title("Spectral Contour")

        self.canvas.draw()

    def calculate_t2(self):

        if self.cars is None:
            return

        T2, dT2 = self.cars.get_T2(
            500,
            2000,
            show_plot=False
        )

        self.result_box.append(
            f"T2 = {T2:.1f} ± {dT2:.1f} fs"
        )


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
