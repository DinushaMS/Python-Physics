import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

class TdCARS:
    def __init__(self, notes_df, sample, wl1, wl2, _wl3, _mono, td_exp, signal_exp, spectra, attenuation,tp1, tp2, tp3, tmin, tmax, floor, nuR1, T21, A1, phi=0):
        self.notes_df = notes_df
        self.tp1,self.tp2,self.tp3 = tp1, tp2, tp3  # Pulse durations [fs]
        self.tmin,self.tmax = tmin, tmax  # Time delay range [fs]
        self.floor = floor  # Baseline signal level
        self.nuR1 = nuR1  # Raman shift frequencies [cm^-1]
        self.T21 = T21  # Dephasing times [fs]
        self.A1 = A1  # Amplitudes of Raman modes [au]
        self.phi1 = phi  # Phase of first Raman mode [rad]

        self.sample = sample
        self.wl1 = wl1  # Pump wavelength [nm]
        self.wl2 = wl2  # Stokes wavelength [nm]
        self._wl3 = _wl3  # Probe wavelength [nm]
        self._mono = _mono  # Monochromator setting [nm]
        self.td_arr = td_exp  # Experimental time delays [fs]
        self.signal_exp = signal_exp  # Experimental CARS signal [a.u.]
        self.spectra = spectra  # CARS spectra [a.u.]
        self.attenuation = attenuation  # Attenuation values [a.u.]

        if len(td_exp) != len(signal_exp):
            self.signal_exp = self.signal_exp[0:len(td_exp)]
            self.attenuation = self.attenuation[0:len(td_exp)]

        self.c = 2.99792e10  # speed of light [cm/s]
        w1 = 1e7 * 2 * np.pi * self.c / self.wl1
        w2 = 1e7 * 2 * np.pi * self.c / self.wl2
        self.wt = w1 - w2
        wR1 = 2 * np.pi * self.nuR1 * self.c
        self.wr1 = (self.wt - wR1) * 1e-15
        self.nut = self.wt / (2 * np.pi * self.c)
        self.target_as_wl = 1/(1/self._wl3 + 1/self.wl1 - 1/self.wl2)

        self.signal_exp_corrected = self.signal_exp.copy()
        self.att_exp_corrected = self.attenuation.copy()
        self.wl_as = self._px2wl(np.arange(2048))
        self.wn_as = self._px2wn(np.arange(2048))
        self.spectra_sc_full = self.spectra+np.abs(np.min(self.spectra))

    @classmethod
    def from_file(cls, cars_file_path):
        #cars_file_path = r"D:\Academic\URI\Research\Data_and_Results\experimental_data\2024\CARS\Aug_07\LNB_1.dat"
        #Read notes file
        cars_notes_file_path = cars_file_path[:-4]+"_Notes.dat"
        notes_df = pd.read_csv(cars_notes_file_path, sep="\t",header=None)

        # Extract parameters from notes
        sample = notes_df.loc[notes_df[0] == 'SMP'][1].tolist()[0]
        _wl3 = float(notes_df.loc[notes_df[0] == 'Ti-Sa'][1].tolist()[0])
        wl1 = float(notes_df.loc[notes_df[0] == 'OPO1'][1].tolist()[0])
        wl2 = float(notes_df.loc[notes_df[0] == 'OPO2'][1].tolist()[0])
        _mono = float(notes_df.loc[notes_df[0] == 'MONO'][1].tolist()[0])
        #self.spectral_window = [float(x) for x in self.notes_df.loc[self.notes_df[0] == 'Spectral window (px)'][1].tolist()[0].split(' to ')]
        #self.notes_version = float(self.notes_df.loc[self.notes_df[0] == 'Version'][1].tolist()[0])

        # Read spectra and floor data files
        cars_spectra_file_path = cars_file_path[:-4]+"_Spectra.dat"
        cars_floor_file_path = cars_file_path[:-4]+"_Floor.dat"
        spectra_data = np.loadtxt(cars_spectra_file_path, delimiter='\t')
        cars_floor = np.loadtxt(cars_floor_file_path, delimiter='\t')
        td_exp = spectra_data[:,0]
        attSpectra = spectra_data[:,1]
        rawSpectra = spectra_data[:,2:]

        rCnt,cCnt = len(rawSpectra[:,0]),len(rawSpectra[0,:])
        att2d = np.zeros([rCnt,cCnt])
        floor2d = np.zeros([rCnt,cCnt])
        curr_att = 1
        curr_floor_idx = 0
        for i in range(rCnt):
            if not curr_att == attSpectra[i]:
                curr_att = attSpectra[i]
                curr_floor_idx += 1
            else:
                pass
            att2d[i,:] = np.ones(cCnt)*curr_att
            floor2d[i,:] = cars_floor[curr_floor_idx,:]

        spectra = np.multiply((rawSpectra-floor2d),att2d)

        #imprrot experimental CARS transient data
        data = np.loadtxt(cars_file_path, delimiter='\t')
        #td_exp = data[:,0]
        signal_exp = data[:,1]
        #self.int_signal = data[:,2]
        attenuation = data[:,3]

        tp1,tp2,tp3 = 260,260,220  # Pulse durations [fs]
        tmin,tmax = -3000,6000  # Time delay range [fs]
        floor = 100  # Baseline signal level
        nuR1 = np.array([730,800])  # Raman shift frequencies [cm^-1]
        T21 = np.array([377,300])  # Dephasing times [fs]
        A1 = np.array([1.9e25,0])  # Amplitudes of Raman modes [au]
        return cls(notes_df, sample, wl1, wl2, _wl3, _mono, td_exp, signal_exp, spectra, attenuation,tp1, tp2, tp3, tmin, tmax, floor, nuR1, T21, A1)
    
    @classmethod
    def from_params(cls, wl1, wl2, tp1, tp2, tp3, tmin, tmax, floor, nuR, T2, A, phi):
        sample = "Mock_Sample"
        _wl3 = 800  # Example probe wavelength [nm]
        _mono = 750  # Example monochromator setting [nm]
        td_exp = np.arange(tmin, tmax, 20)  # Example experimental time delays [fs]
        signal_exp = np.zeros_like(td_exp)  # Empty experimental signal array
        spectra = np.zeros((len(td_exp), 2048))  # Empty spectra array
        attenuation = np.ones_like(td_exp)  # Example attenuation array
        T21 = T2
        nuR1 = nuR
        A1 = A
        notes_df = pd.DataFrame()
        return cls(notes_df, sample, wl1, wl2, _wl3, _mono, td_exp, signal_exp, spectra, attenuation,tp1, tp2, tp3, tmin, tmax, floor, nuR1, T21, A1, phi)

    @property
    def mono(self):
        return self._mono
    
    @property
    def wl3(self):
        return self._wl3
    
    @mono.setter
    def mono(self, value):
        self._mono = value        
        self.wl_as = self._px2wl(np.arange(2048))
        self.wn_as = self._px2wn(np.arange(2048))

    @wl3.setter
    def wl3(self, value):
        self._wl3 = value
        self.wn_as = self._px2wn(np.arange(2048))
        self.target_as_wl = 1/(1/self._wl3 + 1/self.wl1 - 1/self.wl2)
  
    def CARS_simulation_FG(self, showPlot=False):
        """
        Simulates Coherent Anti-Stokes Raman Scattering (CARS) signal.

        Returns:
        td : Time delay array [fs]
        signal : Simulated CARS signal array [a.u.]
        """
        # ---------------------------------------------------------
        # Constants & Input Parameters
        # ---------------------------------------------------------
        w1 = 1e7 * 2 * np.pi * self.c / self.wl1
        w2 = 1e7 * 2 * np.pi * self.c / self.wl2
        self.wt = w1 - w2
        wR1 = 2 * np.pi * self.nuR1 * self.c
        self.wr1 = (self.wt - wR1) * 1e-15
        self.nut = self.wt / (2 * np.pi * self.c)
        self.target_as_wl = 1/(1/self._wl3 + 1/self.wl1 - 1/self.wl2)

        norm = 1e-15  # [fs] to [s] conversion factor

        a12 = -2 * np.log(2) * (1/self.tp1**2 + 1/self.tp2**2)

        b21 = -1/self.T21 - 1j*self.wr1

        # ---------------------------------------------------------
        # Time grid
        # ---------------------------------------------------------
        step0 = 5
        step1 = 5
        t1 = np.arange(self.tmin, self.tmax, step1)
        m1 = len(t1)

        lim = 5 * self.tp1
        ts = np.arange(-lim, lim, step0)

        Q = np.zeros(m1, dtype=complex)

        # ---------------------------------------------------------
        # Main integration loop
        # ---------------------------------------------------------
        for j1, t in enumerate(t1):

            if t < -lim:
                hs = np.zeros_like(ts)
            elif t > lim:
                hs = np.ones_like(ts)
            else:
                p1 = int(round((t + lim) / step0))
                hs = np.concatenate((np.ones(p1), np.zeros(len(ts) - p1)))
            #hs = np.heaviside(t-ts,1)
            F1 = hs * (
                self.A1[0] * np.exp(b21[0] * (t - ts) + self.phi1 * 1j) +
                self.A1[1] * np.exp(b21[1] * (t - ts))
            ) * np.exp(a12 * ts**2)

            Q[j1] = step0 * np.trapezoid(F1)

        Q11 = (norm * np.abs(Q))**2

        # ---------------------------------------------------------
        # Convolution with probe pulse
        # ---------------------------------------------------------
        a3 = -4 * np.log(2) / self.tp3**2
        step2 = 5

        td = np.arange(self.tmin + 5*self.tp3, self.tmax - 5*self.tp3 + step2, step2)
        m2 = len(td)

        signal = np.zeros(m2)

        for j2 in range(m2):
            I3 = np.sqrt(-a3/np.pi) * np.exp(a3 * (t1 - td[j2])**2)
            F2 = Q11 * I3
            signal[j2] = norm * step2 * np.trapezoid(F2) + self.floor
        if showPlot:
            fig = plt.figure(figsize=(7,5))
            plt.semilogy(self.td_arr, self.signal_exp_corrected, 'ko', mfc='none', label='Experimental Data')
            plt.semilogy(td, signal, 'r-', label='Fitted Data, '+r"$T_2$={:.0f}".format(self.T21[0])+" fs")
            plt.xlabel('Time Delay (fs)')
            plt.ylabel('CARS Signal (a.u.)')
            plt.title('CARS Signal: Experimental vs Fitted')
            plt.legend()
            plt.show()
        return td, signal
    
    def get_spectra_contour(self, show_plot=False, **kwargs):
        #self.X_full,self.Y_full = np.arange(1,2049),self.td[:-1]
        self.spectra_sc_full = self.spectra+np.abs(np.min(self.spectra))
        spectra_sc_crop = self.spectra_sc_full
        X, Y = self.wn_as.copy(), self.td_arr.copy()

        if 'wn_lim' in kwargs:
            X= self.wn_as[(self.wn_as<kwargs['wn_lim'][1])&(self.wn_as>kwargs['wn_lim'][0])]
            spectra_sc_crop = spectra_sc_crop[:,(self.wn_as<kwargs['wn_lim'][1])&(self.wn_as>kwargs['wn_lim'][0])]

        if 'td_lim' in kwargs:
            Y = self.td_arr[(self.td_arr<kwargs['td_lim'][1])&(self.td_arr>kwargs['td_lim'][0])]
            spectra_sc_crop = spectra_sc_crop[(self.td_arr<kwargs['td_lim'][1])&(self.td_arr>kwargs['td_lim'][0]),:]
        
        Z = np.log(spectra_sc_crop)
        
        if show_plot:
            plt.figure(figsize=(4,3),dpi=150)
            plt.contourf(X, Y, Z)
            plt.colorbar()
            for i,y in enumerate(Y):
                #Z[Y,X]
                zx = Z[i,:]
                max_zx = np.max(zx)
                max_idx = np.where(zx==max_zx)[0][0]
                plt.plot(X[max_idx], y, 'rx', markersize=3)
            plt.title(f"CARS signal for delay vs pixel for {self.sample}")
            plt.xlabel("$wavenumber$ [1/cm]"); plt.ylabel("$t_d\, [fs]$"); plt.grid(); plt.show()
        return Z
    
    def correct_experimental_data(self, corrections):
        """
        Corrects the experimental CARS signal for incorrect attenuations.
        signal_exp : Experimental CARS signal [a.u.]
        attenuation : list of attenuation values [a.u.]
        corrections : 2D array of attenuation corrections [[old1,new1],[old2,new2],...]
        Returns:
        signal_exp_corrected : Corrected experimental CARS signal [a.u.]
        """
        self.att_exp_corrected = self.attenuation.copy()
        att_bofr = np.append(corrections[:,0],[ x * y for x, y in itertools.combinations(corrections[:,0], 2)])
        att_aftr = np.append(corrections[:,1],[ x * y for x, y in itertools.combinations(corrections[:,1], 2)])
        for i in range(len(att_bofr)):
            self.att_exp_corrected[self.att_exp_corrected == att_bofr[i]] = att_aftr[i]
        self.signal_exp_corrected = self.signal_exp/self.attenuation*self.att_exp_corrected

    def _px2wl(self, px):
        """
        Converts pixel to wavelength[nm].

        Parameters
        ----------
        px : int or numpy.ndarray
            The pixel value.

        Returns
        -------
        float or numpy.ndarray
            The wavelength value in nm.
        """
        gd=1200#grating density[groves/mm]
        cs=2048#chip size
        ps=14e-3#pixel spacing [mm]
        c1 = np.radians(10.63*2)/2
        a = np.arcsin((self._mono*gd*1e-6)/(np.cos(c1)*2)) - c1
        b = a + np.radians(10.63*2) + np.radians(-5.5)
        h = np.sin(np.radians(-5.5))*318.719
        l = np.cos(np.radians(-5.5))*318.719
        m = b-np.arctan(((((cs/2)-px+1)*ps)+h)/l)
        (np.sin(m)+np.sin(a))*(1000000/gd)
        return (np.sin(m)+np.sin(a))*(1000000/gd)
    
    def _px2wn(self, px):
        """
        Converts pixel to wavenumber[1/cm].

        Parameters
        ----------
        px : int or numpy.ndarray
            The pixel value.

        Returns
        -------
        arr : float or numpy.ndarray
            The wavenumber values in 1/cm.
        """
        wl = self._px2wl(px)
        return 1E7*(1/wl-1/self._wl3)

    def plot_spectra_at_td(self, td, show_plot=False):
        spectra_at_td = self.spectra[np.where(self.td_arr >= td)[0][0],:]
        #spectra_at_td = spectra_at_td/np.max(spectra_at_td)
        if show_plot:
            fig = plt.figure(figsize=(10,5))
            plt.subplot(121)
            plt.plot(self.wl_as, spectra_at_td)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('CARS Signal (a.u.)')
            plt.title(f'CARS Spectrum at {td} fs Delay')
            plt.subplot(122)
            plt.plot(self.wn_as, self.plot_spectra_at_td(td))
            plt.xlabel('Wavenumber (1/cm)')
            plt.ylabel('CARS Signal (a.u.)')
            plt.title(f'CARS Spectrum at {td} fs Delay')
            plt.show()
        return spectra_at_td
    
    def get_transient_at_wn(self,wn_target,showPlot=False):
        res = self.spectra_sc_full[:,np.where(self.wn_as>=wn_target)[0][-1]]
        if showPlot:
            plt.figure(figsize=(5,3),dpi=150)
            plt.yscale("log")
            plt.plot(self.td_arr, res, '-o',color = 'k', mfc='none', mec='k', mew=0.5, ms=2, lw=0.5)
            plt.title(f"$CARS\,\, signal\,\,vs\,\, dellay\,\, @\,\, wavenumber\,\, =\,\, {self.wn_as[self.wn_as>=wn_target][-1]:.0f}\,$"+r"$cm^{-1}$"+"$\, for\,\, {self.sample}$")
            plt.xlabel("$delay\, [fs]$"); plt.ylabel("$Signal\, [counts]$")
            plt.grid(); plt.show()
        return res
    
    def get_T2(self, td1, td2, show_plot=False):
        """
        Estimates the dephasing time T2 from the experimental CARS data.
        td_exp : Time delay array from experimental data [fs]
        signal_exp_corrected : Corrected CARS signal from experimental data [a.u.]
        td1, td2 : Time delay range for fitting [fs]
        show_plot : If True, displays the fitting plot
        Returns:
        T2 : Estimated dephasing time [fs]
        dT2 : Uncertainty in the estimated dephasing time [fs]
        """
        x = self.td_arr[(self.td_arr>=td1) & (self.td_arr<=td2)]
        y = np.log(self.signal_exp_corrected[(self.td_arr>=td1) & (self.td_arr<=td2)])
        (m, b), cov = np.polyfit(x, y, 1, cov=True)
        dm = 2.326*np.sqrt(cov[0, 0])  # uncertainty in slope at 98% confidence interval
        db = 2.326*np.sqrt(cov[1, 1])  # uncertainty in intercept at 98% confidence interval
        T2, dT2 = -2/m, 2*dm/m**2
        if show_plot:
            fig = plt.figure(figsize=(12,5))
            plt.subplot(1,3,1)
            plt.semilogy(self.td_arr, self.signal_exp_corrected, 'ko', mfc='none', label='All Transient Data')
            plt.semilogy(x, np.exp(y), 'bo', mfc='none', label='Decay Data')
            plt.xlabel('Time Delay (fs)')
            plt.ylabel('CARS Signal (a.u.)')
            plt.title('CARS Signal: Experimental Data')
            plt.legend()
            plt.subplot(1,3,(2,3))
            plt.plot(x,y,'bo', mfc='none', label='Data for Linear Fit')
            plt.plot(x, m*x + b, 'r-', label='Linear Fit'+r", slope={:.0f}±{:.0f} fs".format(-2/m,2*dm/m**2))
            plt.plot(x, (m+dm)*x + b+db, 'r--', label=r'$\pm$'+r'2.326$\sigma$'+'(98\% CI)')
            plt.plot(x, (m-dm)*x + b-db, 'r--')
            plt.xlabel('Time Delay (fs)')
            plt.ylabel('ln(CARS Signal)')
            plt.title('Logarithmic Decay Fit')
            plt.legend()
            plt.show()
        return T2, dT2