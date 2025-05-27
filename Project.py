
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=100):
        self.R = R      # Radius
        self.w = w      # Width of the strip
        self.n = n      # Resolution
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w/2, w/2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self._generate_mesh()

    def _generate_mesh(self):
        u, v = self.U, self.V
        R = self.R
        self.X = (R + v * np.cos(u / 2)) * np.cos(u)
        self.Y = (R + v * np.cos(u / 2)) * np.sin(u)
        self.Z = v * np.sin(u / 2)

    def surface_area(self):
        # Compute partial derivatives
        du = self.u[1] - self.u[0]
        dv = self.v[1] - self.v[0]
        xu = np.gradient(self.X, du, axis=1)
        xv = np.gradient(self.X, dv, axis=0)
        yu = np.gradient(self.Y, du, axis=1)
        yv = np.gradient(self.Y, dv, axis=0)
        zu = np.gradient(self.Z, du, axis=1)
        zv = np.gradient(self.Z, dv, axis=0)

        # Cross product of partials
        cross_x = yu * zv - zu * yv
        cross_y = zu * xv - xu * zv
        cross_z = xu * yv - yu * xv
        norm = np.sqrt(cross_x**2 + cross_y**2 + cross_z**2)

        # Approximate surface area via integration
        area = simpson(simpson(norm, self.v), self.u)
        return area

    def edge_length(self):
        # Trace the edge at v = w/2
        edge_x = (self.R + (self.w/2) * np.cos(self.u / 2)) * np.cos(self.u)
        edge_y = (self.R + (self.w/2) * np.cos(self.u / 2)) * np.sin(self.u)
        edge_z = (self.w/2) * np.sin(self.u / 2)

        dx = np.diff(edge_x)
        dy = np.diff(edge_y)
        dz = np.diff(edge_z)
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        return np.sum(segment_lengths)

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=1, cstride=1,
                        color='cyan', edgecolor='k', alpha=0.8)
        ax.set_title("Möbius Strip")
        plt.tight_layout()
        plt.show()

# Usage
if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=200)
    area = mobius.surface_area()
    length = mobius.edge_length()
    print(f"Surface Area ≈ {area:.4f} units²")
    print(f"Edge Length ≈ {length:.4f} units")
    mobius.plot()
