import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class Surface:
    comment: str = ""
    R: float = np.inf
    thickness: float = 0.0
    n: float = 1.0
    diameter: float = 20

class RayTrace:

    def __init__(self, no_of_rays=5, aperture=10):
        self.no_of_rays = no_of_rays
        self.init_aperture = aperture
        self.surfaces = []

        self.surf_lw = 0.8
        self.ray_lw = 0.6

        self.fig = None
        self.ax = None
        self.ax2 = None

    # -----------------------------
    # Add optical surface
    # -----------------------------
    def surface(self, comment="", rad_curv=np.inf, thickness=0, ref_idx=1, dia=20):
        self.surfaces.append(Surface(comment, rad_curv, thickness, ref_idx, dia))

    # -----------------------------
    # ABCD matrices
    # -----------------------------
    @staticmethod
    def propagate(y, theta, d):
        y2 = y + d*theta
        theta2 = theta
        return y2, theta2

    @staticmethod
    def refract(y, theta, n1, n2, R):
        if np.isinf(R):
            theta2 = (n1/n2)*theta
        else:
            theta2 = (n1/n2)*theta + (n1-n2)/(n2*R)*y
        return y, theta2

    # -----------------------------
    # Surface sag
    # -----------------------------
    @staticmethod
    def sag(R, y):
        if np.isinf(R):
            return np.zeros_like(y)
        return (np.abs(R) - np.sqrt(R**2 - y**2)) * np.sign(R)

    # -----------------------------
    # Effective focal length
    # -----------------------------
    @staticmethod
    def get_EFFL(x_ray, y_ray, theta_ray):
        return x_ray[-1,0] - y_ray[-1,0]/theta_ray[-1,0] - x_ray[-2,0]

    # -----------------------------
    # Ray trace computation (no plotting)
    # -----------------------------
    def trace(self):
        N = len(self.surfaces)
        first_surf = self.surfaces[0]
        aperture = first_surf.diameter

        # initial rays
        y = np.linspace(-aperture/2, aperture/2, self.no_of_rays)
        theta = np.zeros(self.no_of_rays)
        x_pos = 0

        x_ray, y_ray, theta_ray = [], [], []
        x_top_left = None

        for i, surf in enumerate(self.surfaces):
            n1 = self.surfaces[i-1].n if i>0 else 1
            n2 = surf.n

            # surface geometry
            x = x_pos + self.sag(surf.R, y)

            x_ray.append(x.copy())
            y_ray.append(y.copy())
            theta_ray.append(theta.copy())

            # refraction
            y, theta = self.refract(y, theta, n1, n2, surf.R)

            # propagate to next surface
            if i < N-1:
                d = surf.thickness
                y, theta = self.propagate(y, theta, d)
                x_pos += d

        # final propagation (last thickness = screen)
        last = self.surfaces[-1]
        d = last.thickness
        y, theta = self.propagate(y, theta, d)
        x_pos += d
        x = np.ones(self.no_of_rays) * x_pos

        x_ray.append(x.copy())
        y_ray.append(y.copy())
        theta_ray.append(theta.copy())

        # convert to arrays
        x_ray = np.array(x_ray)
        y_ray = np.array(y_ray)
        theta_ray = np.array(theta_ray)

        return x_ray, y_ray, theta_ray

    # -----------------------------
    # Plot ray trace
    # -----------------------------
    def show_trace(self):
        self.fig = plt.figure(figsize=(5,4), dpi=300)
        self.ax = self.fig.add_subplot(2,1,1)
        self.ax2 = self.fig.add_subplot(2,1,2)
        self.ax2.axis("off")
        N = len(self.surfaces)
        x_ray, y_ray, _ = self.trace()

        # Plot surfaces
        x_pos = 0
        x_top_left = None
        for i, surf in enumerate(self.surfaces):
            # y_surf = np.linspace(-surf.diameter/2, surf.diameter/2, 200)
            # x_surf = x_pos + self.sag(surf.R, y_surf)
            # self.ax.plot(x_surf, y_surf, 'b-', lw=self.surf_lw)
            # x_pos += surf.thickness
            n1 = self.surfaces[i-1].n if i>0 else 1
            n2 = surf.n

            # surface geometry
            y_surf = np.linspace(-surf.diameter/2, surf.diameter/2, 200)
            x_surf = x_pos + self.sag(surf.R, y_surf)

            self.ax.plot(x_surf, y_surf, 'b-', lw=self.surf_lw)

            x_vertex = x_pos

            # store lens entry
            if n2 > n1:
                y_edge = surf.diameter/2
                x_top_left = x_vertex + self.sag(surf.R, y_edge)

            # connect lens edges when leaving glass
            if n2 < n1 and x_top_left is not None:
                y_top = surf.diameter/2
                y_bot = -surf.diameter/2

                x_right_edge = x_vertex + self.sag(surf.R, y_top)

                self.ax.plot([x_top_left, x_right_edge],
                            [y_top, y_top],
                            'b-', lw=self.surf_lw)

                self.ax.plot([x_top_left, x_right_edge],
                            [y_bot, y_bot],
                            'b-', lw=self.surf_lw)
            # propagate
            if i < N-1:

                d = surf.thickness

                #y, theta = self.propagate(y, theta, d)

                x_pos += d

        # draw screen
        last = self.surfaces[-1]
        d = last.thickness
        x_pos += d
        y_surf = np.linspace(-last.diameter/2, last.diameter/2, 200)
        x_surf = np.ones_like(y_surf) * x_pos

        self.ax.plot(x_surf, y_surf, 'k-', lw=self.surf_lw)

        # plot rays
        for i in range(self.no_of_rays):
            self.ax.plot(x_ray[:,i], y_ray[:,i], 'r-', lw=self.ray_lw)

        self.ax.axis("equal")
        self.ax.set_title("Ray Trace")
        plt.tight_layout()
        plt.show()

    # -----------------------------
    # Gaussian beam propagation
    # -----------------------------
    def gaussian_beam_trace(self, w0=1.0, lam=0.6328e-3):
        q_list, w_list, R_list = [], [], []

        q = 1j * np.pi * w0**2 / lam
        q_list.append(q)
        w_list.append(w0)
        R_list.append(np.inf)

        for i, surf in enumerate(self.surfaces):
            # free-space propagation
            d = surf.thickness
            M = np.array([[1, d], [0, 1]])
            q = (M[0,0]*q + M[0,1]) / (M[1,0]*q + M[1,1])

            # refraction at curved surface
            n1 = self.surfaces[i-1].n if i>0 else 1
            n2 = surf.n
            R = surf.R
            if not np.isinf(R):
                M_curv = np.array([[1, 0], [(n1-n2)/(n2*R), n1/n2]])
                q = (M_curv[0,0]*q + M_curv[0,1]) / (M_curv[1,0]*q + M_curv[1,1])
            else:
                q = q * n2/n1

            q_list.append(q)
            w = np.sqrt(-lam/(np.pi*np.imag(1/q)))
            R_beam = 1/np.real(1/q) if np.real(1/q)!=0 else np.inf
            w_list.append(w)
            R_list.append(R_beam)

        return w_list, R_list, q_list