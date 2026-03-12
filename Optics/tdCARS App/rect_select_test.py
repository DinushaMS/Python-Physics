import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# Example image
data = np.random.rand(200, 300)

fig, ax = plt.subplots()
ax.imshow(data, origin='lower', aspect='auto')

def on_select(eclick, erelease):
    print("Start:", eclick.xdata, eclick.ydata)
    print("End:", erelease.xdata, erelease.ydata)

rect = RectangleSelector(
    ax,
    on_select,
    useblit=True,
    button=[1],              # Left mouse button
    interactive=True,        # Allow resizing after selection
    drag_from_anywhere=True
)

plt.show()