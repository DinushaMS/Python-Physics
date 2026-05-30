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