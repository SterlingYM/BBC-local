"""Luminosity distances for the fixed reference cosmology of SALT2mu (flat, constant w).

SALT2mu integrates 1/E(z) with Romberg integration at relative tolerance 1e-6; here the integral is tabulated once with
composite Simpson on a 1e-5 grid, so distance moduli differ from SALT2mu's by about 1e-6 mag.
"""
import numpy as np
from scipy import integrate

CLIGHT = 299792.458  # km/s


class Cosmology:
    def __init__(self, OL=0.685, w0=-1.0, H0=70.0, Ok=0.0, wa=0.0):
        if Ok != 0.0 or wa != 0.0:
            from .config import UnsupportedOptionError
            raise UnsupportedOptionError("only flat cosmologies with constant w are supported (p10=0, p12=0)")
        self.OL, self.w0, self.H0 = float(OL), float(w0), float(H0)
        self._grid = None

    def key(self):
        return (self.OL, self.w0, self.H0)

    def _table(self, zmax):
        zcap = max(3.0, float(np.ceil(zmax)))
        if self._grid is None or self._grid[0][-1] < zcap:
            zg = np.linspace(0.0, zcap, int(round(zcap / 1e-5)) + 1)
            om = 1.0 - self.OL
            f = 1.0 / np.sqrt(om * (1 + zg) ** 3 + self.OL * (1 + zg) ** (3 * (1 + self.w0)))
            self._grid = (zg, integrate.cumulative_simpson(f, x=zg, initial=0.0))
        return self._grid

    def comoving(self, z):
        """int_0^z dz' / E(z') (dimensionless)."""
        z = np.asarray(z, dtype=float)
        zg, dc = self._table(np.max(z) if z.size else 0.0)
        return np.interp(z, zg, dc)

    def dl(self, zhel, zhd):
        """D_L = (1 + zhel) c / H0 int_0^zhd dz / E, in Mpc."""
        return (1 + np.asarray(zhel, dtype=float)) * self.comoving(zhd) * CLIGHT / self.H0

    def mu(self, zhel, zhd):
        return 5 * np.log10(self.dl(zhel, zhd)) + 25
