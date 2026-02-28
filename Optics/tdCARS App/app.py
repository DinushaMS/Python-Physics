import sys
import os

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QSplitter, 
    QTreeView, QToolBar, QStatusBar, QFileDialog
)
from PyQt6.QtGui import QAction, QFileSystemModel
from PyQt6.QtCore import Qt, QDir, QSortFilterProxyModel

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ==============================
# Custom Proxy Filter
# ==============================
class FileFilterProxy(QSortFilterProxyModel):
    def filterAcceptsRow(self, source_row, source_parent):
        index = self.sourceModel().index(source_row, 0, source_parent)
        file_name = self.sourceModel().fileName(index)

        # Always allow directories (so navigation works)
        if self.sourceModel().isDir(index):
            return True

        # Hide files containing "_spectra" (case insensitive)
        if "_spectra" in file_name.lower() or "_floor" in file_name.lower() or "_notes" in file_name.lower():
            return False

        return True


# ==============================
# Matplotlib Canvas
# ==============================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.figure = Figure()
        self.ax = self.figure.add_subplot(111)
        self.figure.set_tight_layout(True)
        super().__init__(self.figure)

        self.ax.set_title("Data Viewer")
        self.ax.set_xlabel("delay [fs]")
        self.ax.set_ylabel("CARS signal [counts]")


# ==============================
# Main Window
# ==============================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("CARS Data Viewer")
        self.resize(1200, 700)

        # ======================
        # Central Layout
        # ======================
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        # ======================
        # File System Model
        # ======================
        custom_path = r"D:\Academic\URI\Research\Data_and_Results\experimental_data\CARS"

        self.model = QFileSystemModel()
        self.model.setRootPath(custom_path)

        # Show only *.dat files
        self.model.setNameFilters(["*.dat"])
        self.model.setNameFilterDisables(False)

        # Allow folders + files
        self.model.setFilter(
            QDir.Filter.AllDirs |
            QDir.Filter.Files |
            QDir.Filter.NoDotAndDotDot
        )

        # ======================
        # Proxy Filter Model
        # ======================
        self.proxy_model = FileFilterProxy()
        self.proxy_model.setSourceModel(self.model)

        # ======================
        # Tree View
        # ======================
        self.tree = QTreeView()
        self.tree.setModel(self.proxy_model)

        source_index = self.model.index(custom_path)
        proxy_index = self.proxy_model.mapFromSource(source_index)
        self.tree.setRootIndex(proxy_index)

        self.tree.setColumnWidth(0, 300)
        splitter.addWidget(self.tree)

        # ======================
        # Matplotlib Canvas
        # ======================
        self.canvas = MplCanvas()
        splitter.addWidget(self.canvas)
        splitter.setStretchFactor(1, 1)

        # ======================
        # Menu Bar
        # ======================
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # ======================
        # Tool Bar
        # ======================
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)
        toolbar.addAction(open_action)
        toolbar.addAction(exit_action)

        # ======================
        # Status Bar
        # ======================
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Ready")

        # ======================
        # Tree Click Event
        # ======================
        self.tree.doubleClicked.connect(self.on_file_double_clicked)

    # ==============================
    # File Open Handler
    # ==============================
    def on_file_double_clicked(self, proxy_index):
        source_index = self.proxy_model.mapToSource(proxy_index)
        file_path = self.model.filePath(source_index)

        if os.path.isfile(file_path):
            self.status.showMessage(f"Selected: {file_path}")
            self.load_and_plot(file_path)

    # ==============================
    # Plot Data from .dat File
    # ==============================
    def load_and_plot(self, file_path):
        try:
            x = []
            y = []

            with open(file_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))

            self.canvas.ax.clear()
            self.canvas.ax.plot(x, y, 'ko', mfc='none', ms=4)
            self.canvas.ax.set_title(os.path.basename(file_path))
            self.canvas.ax.set_yscale('log')
            self.canvas.ax.set_xlabel('delay [fs]')
            self.canvas.ax.set_ylabel('CARS signal [counts]')
            self.canvas.draw()

        except Exception as e:
            self.status.showMessage(f"Error loading file: {e}")

    def open_file(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            "",  # start directory ("" = default)
            QFileDialog.Option.ShowDirsOnly
        )

        if folder:
            # Update model root
            self.model.setRootPath(folder)

            source_index = self.model.index(folder)
            proxy_index = self.proxy_model.mapFromSource(source_index)
            self.tree.setRootIndex(proxy_index)

            self.status.showMessage(f"Folder loaded: {folder}")

# ==============================
# Application Entry
# ==============================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())