import sys
import os
import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QSplitter, QTableView,
    QTreeView, QToolBar, QStatusBar, QFileDialog, QTabWidget, QLabel
)
from PyQt6.QtGui import QAction, QFileSystemModel, QGuiApplication
from PyQt6.QtCore import Qt, QDir, QSortFilterProxyModel, QAbstractTableModel

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from TdCARS import TdCARS

# ---------- Pandas Model ----------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def rowCount(self, parent=None):
        return self._df.shape[0]

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                value = self._df.iloc[index.row(), index.column()]
                return str(value)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            if orientation == Qt.Orientation.Vertical:
                return str(self._df.index[section])
        return None

    def update_dataframe(self, new_df):
        self.beginResetModel()
        self._df = new_df
        self.endResetModel()
    
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
    
        # self.dragging = False
        # self.start_point = None
        # self.end_point = None

        # self.figure.canvas.mpl_connect('button_press_event', self.on_press)
        # self.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
        # self.figure.canvas.mpl_connect('button_release_event', self.on_release)
    
    # def on_press(self, event):        
    #     if event.inaxes != self.ax:
    #         return
        
    #     self.dragging = True
    #     self.start_point = (event.xdata, event.ydata)
    #     print("Pressed at:", self.start_point)

    # def on_motion(self, event):        
    #     if self.dragging and event.inaxes == self.ax:
    #         current_point = (event.xdata, event.ydata)
    #         print("Dragging at:", current_point)

    # def on_release(self, event):        
    #     if event.inaxes != self.ax:
    #         return
        
    #     self.dragging = False
    #     self.end_point = (event.xdata, event.ydata)
    #     print("Released at:", self.end_point)

    #     print("Selected region:")
    #     print("x range:", self.start_point[0], "to", self.end_point[0])
    #     print("y range:", self.start_point[1], "to", self.end_point[1])


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
        splitter.setChildrenCollapsible(True)   # allow collapse
        #splitter.setCollapsible(0, True)        # first widget collapsible
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
        # Tab
        # ======================
        self.tabs = QTabWidget()
        # setup tabs
        self.tab1 = QWidget()
        layout = QVBoxLayout()
        #layout.addWidget(QLabel("This is Tab 1"))
        self.tab1.setLayout(layout)

        self.tab2 = QWidget()
        self.tab3 = QWidget()
        layout = QVBoxLayout(self.tab3)

        df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [25, 30, 35],
            "Score": [88.5, 92.0, 79.5]
        })

        self.table = QTableView()
        self.PDmodel = PandasModel(df)
        self.table.setModel(self.PDmodel)

        self.table.setSortingEnabled(True)
        self.table.resizeColumnsToContents()

        layout.addWidget(self.table)
        # add tab to tabs
        self.tabs.addTab(self.tab1, "CARS vs delay")
        self.tabs.addTab(self.tab2, "Spectra")
        self.tabs.addTab(self.tab3, "Notes")
        main_layout = QVBoxLayout(self.tab2)

        # ---- Main horizontal splitter ----
        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(h_splitter)
        self.left_canvas = MplCanvas()
        h_splitter.addWidget(self.left_canvas)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        h_splitter.addWidget(v_splitter)
        self.top_canvas = MplCanvas()
        v_splitter.addWidget(self.top_canvas)
        self.bottom_canvas = MplCanvas()
        v_splitter.addWidget(self.bottom_canvas)
        h_splitter.setSizes([600, 600])   # Left / Right equal
        v_splitter.setSizes([400, 400])   # Top / Bottom equal
        # add tabs to splitter
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)

        # ======================
        # Matplotlib Canvas
        # ======================
        self.canvas = MplCanvas()
        #splitter.addWidget(self.canvas)
        #splitter.setStretchFactor(1, 1)
        self.tab1.layout().addWidget(self.canvas)

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

        # ======================
        # TdCARS instance
        # ======================
        self.tdcars = None

        self.selector = RectangleSelector(
            self.left_canvas.ax,
            self.on_select,
            useblit=True,
            button=[1],
            interactive=True
        )

        self.selector.set_active(False)

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
            print("1")
            self.tdcars = TdCARS.from_file(file_path)
            print("2")
            self.plot_2d_spectra()
            print("3")
            self.plot_td_cars()
            self.selector.set_active(True)
        except Exception as e:
            self.status.showMessage(f"Error loading file: {e}")

    def plot_td_cars(self):
        x,y = self.tdcars.td_arr, self.tdcars.signal_exp
        self.PDmodel.update_dataframe(self.tdcars.notes_df)
        self.table.resizeColumnsToContents()
        self.canvas.ax.clear()
        self.canvas.ax.plot(x, y, 'ko', mfc='none', ms=4)
        #self.canvas.ax.set_title(os.path.basename(file_path))
        self.canvas.ax.set_yscale('log')
        self.canvas.ax.set_xlabel('delay [fs]')
        self.canvas.ax.set_ylabel('CARS signal [counts]')
        self.canvas.ax.grid(which='both')
        self.canvas.draw()

    def plot_2d_spectra(self):
        try:
            wavelengths = self.tdcars.wl_as  # nm
            delays = self.tdcars.td_arr     # fs

            # Fake signal
            intensity = self.tdcars.get_spectra_contour()
            self.left_canvas.ax.clear()
            #self.left_canvas.ax.contourf(wavelengths, delays, intensity)
            #self.left_canvas.ax.colorbar()

            #self.left_canvas.ax.figure()
            self.left_canvas.ax.imshow(
                intensity,
                aspect='auto',
                origin='lower',   # so small wavelength at bottom
                extent=[
                    wavelengths.min(), wavelengths.max(),
                    delays.min(), delays.max()                    
                ],
                cmap='plasma',#'viridis', 'plasma', 'inferno', 'magma', 'cividis'
                vmin=np.min(intensity),
                vmax=np.max(intensity)
            )
            self.left_canvas.ax.set_ylabel("Delay [fs]")
            self.left_canvas.ax.set_xlabel("Wavelength [nm]")
            #self.left_canvas.ax.colorbar(label="Intensity (counts)")
            #self.left_canvas.ax.tight_layout()
            self.left_canvas.ax.grid(which='both')
            self.left_canvas.draw()
        except Exception as e:
            print(e)
        
    def on_select(self, eclick, erelease):
        # x -> delay[fs], y -> wl[nm]
        print("Start:", eclick.xdata, eclick.ydata)
        print("End:", erelease.xdata, erelease.ydata)
        in_range_td_idx_arr = (self.tdcars.td_arr>eclick.ydata) & (self.tdcars.td_arr<erelease.ydata)
        in_range_wl_idx_arr = (self.tdcars.wl_as>eclick.xdata) & (self.tdcars.wl_as<erelease.xdata)
        #print(f"td={td}")
        #print(f"wl={wl}")
        self.top_canvas.ax.clear()
        self.bottom_canvas.ax.clear()
        for td in self.tdcars.td_arr[in_range_td_idx_arr]:
            real_td, spectra = self.tdcars.plot_spectra_at_td(td)
            cropped_spectra = spectra[in_range_wl_idx_arr]
            cropped_wl = self.tdcars.wl_as[in_range_wl_idx_arr]
            self.top_canvas.ax.plot(cropped_wl, cropped_spectra, label=f"td = {real_td:.0f} fs")
        self.top_canvas.ax.set_xlabel('wavelength [nm]')
        self.top_canvas.ax.set_ylabel('CARS signal [counts]')
        self.top_canvas.ax.grid(which='both')
        self.top_canvas.ax.legend()
        self.top_canvas.draw()

        transient = [np.max(spectra) for spectra in self.tdcars.spectra[:,in_range_wl_idx_arr]]
        self.bottom_canvas.ax.plot(self.tdcars.td_arr, transient)
        self.bottom_canvas.ax.set_xlabel('delay [fs]')
        self.bottom_canvas.ax.set_ylabel('CARS signal [counts]')
        self.bottom_canvas.ax.grid(which='both')
        self.bottom_canvas.ax.set_yscale('log')
        self.bottom_canvas.ax.set_xlim([eclick.ydata, erelease.ydata])
        self.bottom_canvas.draw()

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
    screens = QGuiApplication.screens()
    if len(screens) > 1:
        second_screen = screens[1]
        
        # Move window to second screen
        geometry = second_screen.availableGeometry()
        window.setGeometry(geometry)
        
        # Maximize on that screen
        window.showMaximized()
    else:
        window.showMaximized()  # fallback
    window.show()
    sys.exit(app.exec())