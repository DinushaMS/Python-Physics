from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import datetime

# Use raw string for Windows path
dir_path = Path(r"C:\Users\feruz\Documents\experimental_data\Anritsu_OSA_spectra\2021")

# Find all .dat files recursively, excluding those with "summary" in the name
dat_files = [
    f for f in dir_path.rglob("*.dat")
    if "summary" not in f.name.lower()
]
plt.figure(figsize=(10,6))
for file in dat_files:
    # get file creatred date
    created_time = file.stat().st_mtime
    created_date = datetime.datetime.fromtimestamp(created_time)

    # Load tab-separated x, y data
    data = np.loadtxt(file, delimiter="\t")
    x = data[:, 0]
    y = data[:, 1]

    y = (y - np.min(y))/(np.max(y)-np.min(y))

    # Find peaks (tweak parameters as needed)
    peaks, _ = find_peaks(y, prominence=0.05)

    # Get x-values of peaks
    peak_x_values = x[peaks]

    for px in peak_x_values:
        if px > 1140 and px < 1180:
            print(f"{created_date.strftime("%m-%d-%Y")}: {px:.1f} nm")
            plt.plot(x,y,label=f"{created_date.strftime("%m-%d-%Y")}")
plt.legend()
plt.show()