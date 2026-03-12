from MS9710C import AnritsuOSA
from TBS1102B import TektronixDO
import matplotlib.pyplot as plt
import numpy as np

osa = AnritsuOSA()
osa.get_trace()
osa.cleanup()

dso = TektronixDO()
dso.get_trace()
dso.cleanup()

fig, ax = plt.subplots(1, 3, figsize=(12,4))

# Spectral data
wl = osa.wlArray
osa_trace = osa.trace-np.min(osa.trace)

ax[0].plot(wl, osa_trace)

# Temporal data
time = dso.time
dso_trace = dso.trace

ax[1].plot(time, dso_trace)

plt.show()