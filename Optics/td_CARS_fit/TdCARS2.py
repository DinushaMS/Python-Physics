"""
tdcars.py
---------
Time-domain Coherent Anti-Stokes Raman Scattering (td-CARS) analysis module.

Provides the TdCARS class for simulating, fitting, and analyzing time-resolved
CARS transient signals and spectra acquired with a pulsed laser system.
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
C_CM_S = 2.99792e10   # Speed of light [cm/s]
NM_TO_CM = 1e7        # Conversion factor: nm⁻¹ → cm⁻¹
FS_TO_S = 1e-15       # Femtosecond → second conversion factor

# ---------------------------------------------------------------------------
# Spectrometer geometry (fixed hardware parameters)
# ---------------------------------------------------------------------------
_GRATING_DENSITY = 1200   # Grooves/mm
_CHIP_SIZE       = 2048   # CCD pixel count
_PIXEL_SPACING   = 14e-3  # mm
_FOCAL_LENGTH    = 318.719
_DETECTOR_ANGLE  = np.radians(-5.5)
_BLAZE_ANGLE_2X  = np.radians(10.63 * 2)


class TdCARS:
    """
    Time-domain CARS (Coherent Anti-Stokes Raman Scattering) analysis.

    Encapsulates experimental data loading, signal simulation, spectral
    calibration, and dephasing-time extraction for a single td-CARS
    measurement.

    Parameters
    ----------
    notes_df : pd.DataFrame
        Metadata table read from the _Notes.dat file.
    sample : str
        Sample identifier string.
    wl1 : float
        Pump wavelength [nm].
    wl2 : float
        Stokes wavelength [nm].
    wl3 : float
        Probe (Ti:Sa) wavelength [nm].
    mono : float
        Monochromator centre wavelength [nm].
    td_exp : array_like, shape (N,)
        Experimental time-delay axis [fs].
    signal_exp : array_like, shape (N,)
        Integrated CARS transient signal [a.u.].
    spectra : array_like, shape (N, 2048)
        Background-corrected, attenuation-scaled spectral stack.
    attenuation : array_like, shape (N,)
        Attenuation (ND-filter) values recorded at each delay step.
    tp1, tp2, tp3 : float
        FWHM pulse durations for pump, Stokes, and probe pulses [fs].
    tmin, tmax : float
        Simulation time-delay window [fs].
    floor : float
        DC offset / detector dark-count baseline added to the simulated signal.
    nuR : array_like
        Raman resonance frequencies [cm⁻¹].
    T2 : array_like
        Vibrational dephasing times T₂ (one per mode) [fs].
    A : array_like
        Raman mode amplitudes [a.u.].
    phi : float, optional
        Global phase offset applied to all Raman oscillators [rad]. Default 0.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        notes_df, sample,
        wl1, wl2, wl3, mono,
        td_exp, signal_exp, spectra, attenuation,
        tp1, tp2, tp3,
        tmin, tmax,
        floor,
        nuR, T2, A,
        phi=0,
    ):
        self.notes_df = notes_df
        self.sample   = sample

        # Wavelengths / monochromator
        self.wl1   = wl1
        self.wl2   = wl2
        self._wl3  = wl3
        self._mono = mono

        # Pulse parameters
        self.tp1, self.tp2, self.tp3 = tp1, tp2, tp3
        self.tmin, self.tmax = tmin, tmax
        self.floor = floor

        # Raman mode parameters
        self._nuR  = np.atleast_1d(nuR)   # [cm⁻¹]
        self.T2   = np.atleast_1d(T2)    # [fs]
        self.A    = np.atleast_1d(A)     # [a.u.]
        self.phi  = phi                  # [rad]

        # Experimental data
        self.td_arr     = np.asarray(td_exp)
        self.signal_exp = np.asarray(signal_exp)
        self.attenuation = np.asarray(attenuation)
        self.spectra     = np.asarray(spectra, dtype=float)

        # Trim mismatched arrays (legacy tolerance)
        n = len(self.td_arr)
        self.signal_exp  = self.signal_exp[:n]
        self.attenuation = self.attenuation[:n]

        # Working copies of correctable arrays
        self.signal_exp_corrected = self.signal_exp.copy()
        self.att_exp_corrected    = self.attenuation.copy()

        # Spectral calibration axes
        px = np.arange(_CHIP_SIZE)
        self.wl_as = self._px2wl(px)
        self.wn_as = self._px2wn(px)

        # Baseline-corrected spectral stack (always non-negative)
        self.spectra_sc_full = self.spectra #- self.spectra.min()

        # Derived frequency quantities
        self._update_frequencies()

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, cars_file_path):
        """
        Construct a TdCARS instance from a set of measurement files.

        Expects the following files alongside *cars_file_path*:
        - ``<base>_Notes.dat``   – tab-separated metadata (key/value pairs)
        - ``<base>_Spectra.dat`` – spectral stack with time delays and attenuations
        - ``<base>_Floor.dat``   – background (dark-count) spectra per attenuation step
        - ``<base>.dat``         – integrated CARS transient (td, signal, -, attenuation)

        Parameters
        ----------
        cars_file_path : str
            Path to the primary ``.dat`` transient file.

        Returns
        -------
        TdCARS
        """
        cars_file_path = str(cars_file_path)
        base = cars_file_path[:-4]

        # --- Notes ---
        notes_df = pd.read_csv(base + "_Notes.dat", sep="\t", header=None)
        _get = lambda key: notes_df.loc[notes_df[0] == key, 1].iat[0]

        sample = _get("SMP")
        wl3    = float(_get("Ti-Sa"))
        wl1    = float(_get("OPO1"))
        wl2    = float(_get("OPO2"))
        mono   = float(_get("MONO"))

        # --- Spectra + floor ---
        spectra_data = np.loadtxt(base + "_Spectra.dat", delimiter="\t")
        floor_raw    = np.loadtxt(base + "_Floor.dat",   delimiter="\t")

        td_exp      = spectra_data[:, 0]
        att_column  = spectra_data[:, 1]
        raw_spectra = spectra_data[:, 2:]

        n_rows, n_cols = raw_spectra.shape

        # Expand 1-D floor array to 2-D if only one background was recorded
        if floor_raw.ndim == 1:
            floor_raw = floor_raw[np.newaxis, :]

        # Build per-row floor and attenuation 2-D arrays in one pass
        att2d   = np.empty((n_rows, n_cols))
        floor2d = np.empty((n_rows, n_cols))
        floor_idx  = 0
        prev_att   = att_column[0]

        for i in range(n_rows):
            if att_column[i] != prev_att:
                prev_att = att_column[i]
                floor_idx += 1
            att2d[i]   = att_column[i]
            floor2d[i] = floor_raw[floor_idx]

        spectra = (raw_spectra - floor2d) * att2d

        # --- Transient ---
        transient   = np.loadtxt(cars_file_path, delimiter="\t")
        signal_exp  = transient[:, 1]
        attenuation = transient[:, 3]

        # Default model parameters (can be updated after construction)
        tp1, tp2, tp3 = 260, 260, 220          # [fs]
        tmin, tmax    = -3000, 6000            # [fs]
        floor         = 100
        nuR = np.array([730, 800])             # [cm⁻¹]
        T2  = np.array([377, 300])             # [fs]
        A   = np.array([1.9e25, 0.0])          # [a.u.]

        return cls(
            notes_df, sample,
            wl1, wl2, wl3, mono,
            td_exp, signal_exp, spectra, attenuation,
            tp1, tp2, tp3, tmin, tmax, floor,
            nuR, T2, A,
        )

    @classmethod
    def from_params(cls, wl1, wl2, tp1, tp2, tp3, tmin, tmax, floor, nuR, T2, A, phi=0):
        """
        Construct a TdCARS instance from model parameters only (no data files).

        Useful for forward simulations and parameter sweeps.

        Parameters
        ----------
        wl1, wl2 : float
            Pump and Stokes wavelengths [nm].
        tp1, tp2, tp3 : float
            FWHM pulse durations [fs].
        tmin, tmax : float
            Simulation window [fs].
        floor : float
            Baseline offset.
        nuR : array_like
            Raman frequencies [cm⁻¹].
        T2 : array_like
            Dephasing times [fs].
        A : array_like
            Raman amplitudes [a.u.].
        phi : float, optional
            Global phase [rad]. Default 0.

        Returns
        -------
        TdCARS
        """
        wl3  = 800.0
        mono = 750.0
        td_exp    = np.arange(tmin, tmax, 20, dtype=float)
        n         = len(td_exp)
        return cls(
            pd.DataFrame(), "Mock_Sample",
            wl1, wl2, wl3, mono,
            td_exp,
            np.zeros(n),
            np.zeros((n, _CHIP_SIZE)),
            np.ones(n),
            tp1, tp2, tp3, tmin, tmax, floor,
            nuR, T2, A, phi,
        )

    # ------------------------------------------------------------------
    # Properties (wavelength / monochromator setters recalibrate axes)
    # ------------------------------------------------------------------

    @property
    def mono(self):
        """Monochromator centre wavelength [nm]."""
        return self._mono

    @mono.setter
    def mono(self, value):
        self._mono = value
        px = np.arange(_CHIP_SIZE)
        self.wl_as = self._px2wl(px)
        self.wn_as = self._px2wn(px)

    @property
    def wl3(self):
        """Probe (Ti:Sa) wavelength [nm]."""
        return self._wl3

    @wl3.setter
    def wl3(self, value):
        self._wl3 = value
        self.wn_as = self._px2wn(np.arange(_CHIP_SIZE))
        self._update_frequencies()
    @property
    def nuR(self):
        """Raman resonance frequencies [cm⁻¹]."""
        return self._nuR
    
    @nuR.setter
    def nuR(self, value):
        self._nuR = value
        self._update_frequencies()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_frequencies(self):
        """Recompute derived angular-frequency quantities from wl1, wl2, wl3."""
        w1 = NM_TO_CM * 2 * np.pi * C_CM_S / self.wl1
        w2 = NM_TO_CM * 2 * np.pi * C_CM_S / self.wl2
        self.wt  = w1 - w2                                      # Raman drive frequency [rad/s]
        wR       = 2 * np.pi * self.nuR * C_CM_S               # [rad/s]
        self.wr  = (self.wt - wR) * FS_TO_S                    # detuning [rad/fs]
        self.nut = self.wt / (2 * np.pi * C_CM_S)              # [cm⁻¹]
        self.target_as_wl = 1.0 / (1/self._wl3 + 1/self.wl1 - 1/self.wl2)

    def _px2wl(self, px):
        """
        Convert CCD pixel index to wavelength.

        Uses the fixed grating spectrometer geometry constants.

        Parameters
        ----------
        px : int or ndarray
            Pixel index (0-based).

        Returns
        -------
        float or ndarray
            Wavelength [nm].
        """
        c1 = _BLAZE_ANGLE_2X / 2
        a  = np.arcsin((self._mono * _GRATING_DENSITY * 1e-6) / (np.cos(c1) * 2)) - c1
        b  = a + _BLAZE_ANGLE_2X + _DETECTOR_ANGLE

        h  = np.sin(_DETECTOR_ANGLE) * _FOCAL_LENGTH
        l  = np.cos(_DETECTOR_ANGLE) * _FOCAL_LENGTH
        m  = b - np.arctan(((_CHIP_SIZE / 2 - px + 1) * _PIXEL_SPACING + h) / l)

        return (np.sin(m) + np.sin(a)) * (1e6 / _GRATING_DENSITY)

    def _px2wn(self, px):
        """
        Convert CCD pixel index to Raman shift wavenumber.

        Parameters
        ----------
        px : int or ndarray
            Pixel index (0-based).

        Returns
        -------
        float or ndarray
            Raman shift [cm⁻¹] relative to the probe wavelength wl3.
        """
        return 1e7 * (1.0 / self._px2wl(px) - 1.0 / self._wl3)

    # ------------------------------------------------------------------
    # Core simulation
    # ------------------------------------------------------------------

    def simulate_cars(self, show_plot=False, step0=5, step1=5, step2=5):
        """
        Simulate the time-domain CARS transient signal.

        The simulation follows the Fourier-Green's function (FG) approach:

        1. For each pump–Stokes delay *t*, the nuclear response Q(t) is
           computed by integrating the driven oscillator response over the
           pump/Stokes pulse envelope.
        2. |Q(t)|² is then convolved with the probe intensity profile to
           yield the detected CARS signal.

        Parameters
        ----------
        show_plot : bool, optional
            If True, overlays the simulation on the experimental transient.
        step0 : float, optional
            Integration grid spacing for the Q(t) inner integral [fs]. Default 5.
        step1 : float, optional
            Time-delay grid spacing for Q(t) [fs]. Default 5.
        step2 : float, optional
            Time-delay grid spacing for the probe convolution [fs]. Default 5.

        Returns
        -------
        td : ndarray
            Simulated time-delay axis [fs].
        signal : ndarray
            Simulated CARS signal [a.u.].
        """
        # Gaussian pump × Stokes joint spectral amplitude exponent
        a12 = -2 * np.log(2) * (1/self.tp1**2 + 1/self.tp2**2)

        # Oscillator decay constants (complex): −1/T₂ − iΔω
        b = -1.0 / self.T2 - 1j * self.wr    # shape (n_modes,)

        # --- Inner integral: nuclear response Q(t) ---
        lim = 5 * self.tp1
        ts  = np.arange(-lim, lim, step0)       # integration variable
        t1  = np.arange(self.tmin, self.tmax, step1)

        pump_stokes_env = np.exp(a12 * ts**2)    # pump × Stokes envelope (precomputed)

        Q = np.zeros(len(t1), dtype=complex)

        for j, t in enumerate(t1):
            # Heaviside step function (vectorised via clipping)
            if t <= -lim:
                hs = np.zeros(len(ts))
            elif t >= lim:
                hs = np.ones(len(ts))
            else:
                p = int(round((t + lim) / step0))
                hs = np.zeros(len(ts))
                hs[:p] = 1.0

            # Sum oscillator contributions: shape (n_modes, n_ts) → scalar
            osc = np.sum(
                self.A[:, None] * np.exp(b[:, None] * (t - ts)),
                axis=0,
            ) * np.exp(1j * self.phi)

            F1   = hs * osc * pump_stokes_env
            Q[j] = step0 * np.trapezoid(F1)

        Q_sq = (FS_TO_S * np.abs(Q)) ** 2

        # --- Outer convolution with probe pulse ---
        a3 = -4 * np.log(2) / self.tp3**2

        td     = np.arange(self.tmin + 5*self.tp3, self.tmax - 5*self.tp3 + step2, step2)
        signal = np.empty(len(td))

        for j, t_d in enumerate(td):
            I3       = np.sqrt(-a3 / np.pi) * np.exp(a3 * (t1 - t_d)**2)
            signal[j] = FS_TO_S * step2 * np.trapezoid(Q_sq * I3) + self.floor

        if show_plot:
            self._plot_transient(td, signal)

        return td, signal

    # Backwards-compatible alias
    def CARS_simulation_FG(self, showPlot=False):
        """Alias for :meth:`simulate_cars` (backwards compatibility)."""
        return self.simulate_cars(show_plot=showPlot)

    # ------------------------------------------------------------------
    # Spectral analysis
    # ------------------------------------------------------------------

    def get_spectra_contour(self, show_plot=False, wn_lim=None, td_lim=None, no_of_peaks=1):
        """
        Compute (and optionally display) the log-intensity CARS spectral contour.

        Parameters
        ----------
        show_plot : bool, optional
            Plot a filled contour map with peak-position markers. Default False.
        wn_lim : tuple of float, optional
            ``(wn_min, wn_max)`` wavenumber crop range [cm⁻¹].
        td_lim : tuple of float, optional
            ``(td_min, td_max)`` time-delay crop range [fs].
        no_of_peaks : int, optional
            Number of peaks to mark at each delay. Default 1.

        Returns
        -------
        Z : ndarray
            Log-intensity spectral map, shape ``(n_td, n_wn)``.
        """
        # Clip negative counts to zero (dark-count over-subtraction artefacts),
        # then shift so the minimum positive value becomes the new zero.
        # This preserves the true signal dynamic range instead of compressing it
        # by shifting up a potentially large negative baseline.
        spectra_pos = np.clip(self.spectra, 0, None)
        self.spectra_sc_full = spectra_pos

        wn_mask = np.ones(len(self.wn_as), dtype=bool)
        td_mask = np.ones(len(self.td_arr), dtype=bool)

        if wn_lim is None:
            wn_lim = [self._px2wn(_CHIP_SIZE - 1), self._px2wn(0)]
        if td_lim is None:
            td_lim = [self.td_arr.min(), self.td_arr.max()]
        print(wn_lim, td_lim)
        wn_mask = (self.wn_as >= wn_lim[0]) & (self.wn_as <= wn_lim[1])
        td_mask = (self.td_arr >= td_lim[0]) & (self.td_arr <= td_lim[1])
        X = self.wn_as[wn_mask]
        Y = self.td_arr[td_mask]
        cropped = self.spectra_sc_full[np.ix_(td_mask, wn_mask)]

        # Replace zeros/negatives with the smallest positive value in the
        # cropped region so log is well-defined without distorting signal pixels.
        pos_vals = cropped[cropped > 0]
        fill = pos_vals.min() if pos_vals.size > 0 else 1.0
        Z = np.log(np.where(cropped > 0, cropped, fill))
        P = np.zeros_like(Z)  # Placeholder for peak positions (not used in this implementation)

        if show_plot:
            # Use percentile-based colour limits so hot/cold outlier pixels
            # don't wash out the features of interest.
            vmin = np.percentile(Z, 2)
            vmax = np.percentile(Z, 98)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
            cf = axes[0].contourf(X, Y, Z, levels=20, vmin=vmin, vmax=vmax, cmap="viridis")
            fig.colorbar(cf, ax=axes[0], label="ln(counts)")

            # Mark spectral peak at each delay
            #peak_idx = np.argmax(Z, axis=1)
            #ax.plot(X[peak_idx], Y, "rx", markersize=3)
            for i, td in enumerate(Y):
                spec_sm = self.plot_spectra_at_td([td], show_plot=False, boxcar_window=20)[0]
                spec_norm = (spec_sm - np.min(spec_sm)) / (np.max(spec_sm) - np.min(spec_sm))
                peaks, properties = find_peaks(spec_norm, prominence=0.01)  # Adjust parameters as needed
                top_idxes = peaks[np.argsort(properties["prominences"])[-no_of_peaks:][::-1]]
                axes[0].plot(X[top_idxes], np.ones_like(X[top_idxes]) * td, "rx", markersize=3)
                P[i, top_idxes] = 1  # Mark peaks in the P array
            
            axes[0].set_title(f"CARS Spectral Contour: {self.sample}", fontsize=10)
            axes[0].set_xlabel("Wavenumber [cm⁻¹]")
            axes[0].set_ylabel("Time delay [fs]")
            axes[0].grid(True)
            
            peak_wns = []
            for i, td in enumerate(Y):
                peak_wns.append(X[P[i,:]==1])

            hist = np.histogram(np.concatenate(peak_wns), bins=30)

            x, y = hist[1][:-1], hist[0]
            y = self.boxcar_avg(y, 5)
            y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize to [0, 1]
            # add dummy points at the start and end to ensure peaks are detected at the edges
            x = np.concatenate(([x[0] - 1], x, [x[-1] + 1]))
            y = np.concatenate(([0], y, [0]))
            peaks, _ = find_peaks(y, height=0.2, distance=10)  # Adjust parameters as needed
            axes[1].plot(x, y, color="blue")
            axes[1].scatter(x[peaks], y[peaks], color="red", s=50)
            for idx in peaks:
                axes[1].text(x[idx], y[idx], f"{int(x[idx])} cm⁻¹", fontsize=8, ha='center', va='bottom')
            
            axes[1].set_title("Peak wavenumber distribution: td={}–{} fs, wn={}–{} cm⁻¹".format(int(td_lim[0]), int(td_lim[1]), int(wn_lim[0]), int(wn_lim[1])), fontsize=10)
            axes[1].set_xlabel("Wavenumber [cm⁻¹]")
            axes[1].set_ylabel("Normalized count")
            plt.tight_layout()
            plt.show()

        return Z, P

    def plot_spectra_at_td(self, td, show_plot=False, boxcar_window=1):
        """
        Extract CARS spectra at one or more specified time delays.

        Parameters
        ----------
        td : float or list of float
            Target time delay(s) [fs]. The nearest recorded delay is used.
        show_plot : bool, optional
            Plot spectra vs. wavelength and wavenumber side by side. Default False.
        boxcar_window : int, optional
            Smoothing window width (pixels). Default 1 (no smoothing).

        Returns
        -------
        spectra_at_td : ndarray, shape (n_td, 2048)
            Extracted spectral rows.
        """
        tds = np.atleast_1d(td)
        indices = [np.searchsorted(self.td_arr, t) for t in tds]
        spectra_at_td = self.spectra[indices, :]
        smoothed = np.array([self.boxcar_avg(spectra_at_td[i], boxcar_window) for i in range(len(tds))])

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for i, t in enumerate(tds):
                smoothed[i] = self.boxcar_avg(spectra_at_td[i], boxcar_window)
                axes[0].plot(self.wl_as, smoothed[i], label=f"{t} fs")
                axes[1].plot(self.wn_as, smoothed[i], label=f"{t} fs")

            for ax, xlabel in zip(axes, ["Wavelength (nm)", "Wavenumber (cm⁻¹)"]):
                ax.set_xlabel(xlabel)
                ax.set_ylabel("CARS Signal (a.u.)")
                ax.set_title(f"CARS Spectrum at {list(tds)} fs")
                ax.legend()
            plt.tight_layout()
            plt.show()

        return smoothed

    def get_transient_at_wn(self, wn_target, show_plot=False):
        """
        Extract the time-resolved CARS transient integrated over a wavenumber range.

        Parameters
        ----------
        wn_target : float or tuple of float
            Single wavenumber [cm⁻¹] or ``(wn_min, wn_max)`` integration window.
        show_plot : bool, optional
            Plot the transient on a logarithmic scale. Default False.

        Returns
        -------
        td : ndarray
            Time-delay axis [fs].
        signal : ndarray
            Integrated CARS signal [a.u.].
        """
        sc = self.spectra_sc_full
        if isinstance(wn_target, (list, tuple)):
            mask = (self.wn_as >= wn_target[0]) & (self.wn_as <= wn_target[1])
            signal  = sc[:, mask].max(axis=1)
            label = f"{wn_target[0]:.0f}–{wn_target[1]:.0f} cm⁻¹"
        else:
            idx  = np.searchsorted(self.wn_as, wn_target)
            signal  = sc[:, idx]
            label = f"{self.wn_as[idx]:.0f} cm⁻¹"

        if show_plot:
            fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
            ax.semilogy(self.td_arr, signal, "-o", color="k",
                        mfc="none", mec="k", mew=0.5, ms=2, lw=0.5)
            ax.set_title(f"CARS transient @ {label} – {self.sample}")
            ax.set_xlabel("Delay [fs]")
            ax.set_ylabel("Signal [counts]")
            ax.grid(True)
            plt.show()

        return self.td_arr, signal

    # ------------------------------------------------------------------
    # Data correction
    # ------------------------------------------------------------------

    def correct_experimental_data(self, corrections):
        """
        Correct the integrated CARS transient for mis-logged attenuation values.

        Applies a remapping of individual and combined (product) attenuations,
        then rescales ``signal_exp`` accordingly.

        Parameters
        ----------
        corrections : ndarray, shape (K, 2)
            Each row ``[old_att, new_att]`` specifies an attenuation that was
            recorded as *old_att* but should be treated as *new_att*.  Pairwise
            products of the supplied values are also remapped automatically to
            cover ND-filter combinations.

        Returns
        -------
        None
            Updates ``att_exp_corrected`` and ``signal_exp_corrected`` in place.
        """
        corrections = np.asarray(corrections)
        old_vals = corrections[:, 0]
        new_vals = corrections[:, 1]

        # Extend to all pairwise products (ND filter combinations)
        pair_old = [x * y for x, y in itertools.combinations(old_vals, 2)]
        pair_new = [x * y for x, y in itertools.combinations(new_vals, 2)]

        all_old = np.concatenate([old_vals, pair_old])
        all_new = np.concatenate([new_vals, pair_new])

        att_corrected = self.attenuation.copy()
        for old, new in zip(all_old, all_new):
            att_corrected[att_corrected == old] = new

        self.att_exp_corrected    = att_corrected
        self.signal_exp_corrected = self.signal_exp / self.attenuation * att_corrected

    # ------------------------------------------------------------------
    # Dephasing time estimation
    # ------------------------------------------------------------------

    def get_T2(self, td1, td2, show_plot=False):
        """
        Estimate the vibrational dephasing time T₂ by linear regression on
        the log-decay of the CARS transient.

        The slope *m* of ``ln(signal)`` vs. *t_d* gives T₂ = −2 / m
        (the factor of 2 arises because the CARS signal ∝ |Q|²).
        Uncertainties are reported at the 98 % confidence interval
        (2.326σ, two-tailed).

        Parameters
        ----------
        td1, td2 : float
            Start and end of the exponential decay window [fs].
        show_plot : bool, optional
            Display the transient and the log-linear fit. Default False.

        Returns
        -------
        T2 : float
            Estimated dephasing time [fs].
        dT2 : float
            Uncertainty in T₂ at 98 % CI [fs].
        """
        mask = (self.td_arr >= td1) & (self.td_arr <= td2)
        x    = self.td_arr[mask]
        y    = np.log(self.signal_exp_corrected[mask])

        (m, b), cov = np.polyfit(x, y, 1, cov=True)
        dm = 2.326 * np.sqrt(cov[0, 0])   # slope uncertainty (98 % CI)
        db = 2.326 * np.sqrt(cov[1, 1])   # intercept uncertainty

        T2  =  -2.0 / m
        dT2 = 2.0 * dm / m**2

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            axes[0].semilogy(self.td_arr, self.signal_exp_corrected,
                             "ko", mfc="none", label="All data")
            axes[0].semilogy(x, np.exp(y), "bo", mfc="none", label="Decay window")
            axes[0].set_xlabel("Time Delay (fs)")
            axes[0].set_ylabel("CARS Signal (a.u.)")
            axes[0].set_title("Experimental Transient")
            axes[0].legend()

            axes[1].plot(x, y, "bo", mfc="none", label="Log data")
            axes[1].plot(x, m*x + b, "r-",
                         label=rf"Fit: T₂ = {T2:.0f}±{dT2:.0f} fs")
            axes[1].fill_between(x,
                                 (m - dm)*x + (b - db),
                                 (m + dm)*x + (b + db),
                                 color="r", alpha=0.15, label="98 % CI")
            axes[1].set_xlabel("Time Delay (fs)")
            axes[1].set_ylabel("ln(Signal)")
            axes[1].set_title("Log-linear Decay Fit")
            axes[1].legend()

            plt.tight_layout()
            plt.show()

        return T2, dT2

    # ------------------------------------------------------------------
    # Signal processing utilities
    # ------------------------------------------------------------------

    @staticmethod
    def boxcar_avg(x, window):
        """
        Apply a uniform (boxcar) moving-average filter.

        Uses ``scipy.signal.convolve`` in 'same' mode so the output has the
        same length as the input.

        Parameters
        ----------
        x : array_like
            1-D input signal.
        window : int
            Number of samples in the averaging window.

        Returns
        -------
        ndarray
            Smoothed signal (same length as *x*).
        """
        if window <= 1:
            return np.asarray(x, dtype=float)
        kernel = np.ones(window) / window
        return convolve(x, kernel, mode="same")

    # ------------------------------------------------------------------
    # Private plotting helper
    # ------------------------------------------------------------------

    def _plot_transient(self, td_sim, signal_sim):
        """Overlay simulated signal on experimental transient."""
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(self.td_arr, self.signal_exp_corrected,
                    "ko", mfc="none", label="Experimental")
        ax.semilogy(td_sim, signal_sim, "r-",
                    label=rf"Simulation  $T_2$={self.T2[0]:.0f} fs")
        ax.set_xlabel("Time Delay (fs)")
        ax.set_ylabel("CARS Signal (a.u.)")
        ax.set_title("CARS Transient: Experiment vs. Simulation")
        ax.legend()
        plt.show()