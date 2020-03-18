import logging
import numpy as np
from . import timer


def inv_logtransform(plog):
    """ Transform the power spectrum for the log field to the power spectrum of delta.

    Inputs
    ------
    plog - power spectrum  of log field computed at points on a Fourier grid

    Outputs
    -------
    p  - power spectrum of the delta field
    """
    xi_log = np.fft.ifftn(plog)
    xi = np.exp(xi_log) - 1
    p = np.fft.fftn(xi).real.astype('float')
    return p

def logtransform(p, return_corrected=False):
    """ Transform the power spectrum of delta to the power spectrum of the log field.

    Inputs
    ------
    p - power spectrum computed at points on a Fourier grid

    Outputs
    -------
    plog        - power spectrum of the log field
    p_corrected - corrected input power spectrum to match.
    """
    if return_corrected:
        p_corrected = p.copy()

    xi = np.fft.ifftn(p.astype('complex'))

    if not np.min(xi.real) > -1:
        logging.critical("simbox fatal error with log transform! P(k) amp is too high maybe...")
        raise ValueError

    logxi = np.log(1 + xi)
    plog = np.fft.fftn(logxi).real.astype('float')
    plog.flat[0] = 0

    # Set negative modes to 0
    plog[plog<0] = 0

    if return_corrected:
        # Do the inverse transform to compute the corrected input power spectrum
        p_corrected = inv_logtransform(plog)
        return plog, p_corrected

    return plog


def gofft(grid):
    """ Forward FFT """
    with timer.Timer("FFT shape %s time"%str(grid.shape)):
        n = np.prod(grid.shape)
        dk = 1./n*np.fft.fftn(grid)
    return dk


def gofftinv(grid):
    """ inverse FFT """
    with timer.Timer("inv FFT shape %s time"%str(grid.shape)):
        n = np.prod(grid.shape)
        d = n*np.fft.ifftn(grid)
    return d


class SimBox:
    """ """
    def __init__(self, pk_model, shape, length, lognorm=False, apply_window=False):
        """ Generate Gaussian and lognormal simulations in a box.

        Inputs
        ------
        pk_model
        shape
        length
        lognorm
        """
        self.shape = shape
        self.length = length
        self.step = np.array(self.length) / np.array(self.shape)
        self.dim = len(shape)
        assert(self.dim in [1, 2, 3])
        self.volume = np.prod(self.length)
        self.n = np.prod(shape)
        self.cell_volume = self.volume * 1. / self.n

        self.pk_model = pk_model

        self.lognorm = lognorm
        self.apply_window = apply_window

        self._pkgrid = None
        self._kgrid = None
        self._k = None
        self._window = None
        self.reset()

    def reset(self):
        """ """
        self._delta = None
        self._density_k = None
        self._velocity_field = None
        self._window = None

    @property
    def kgrid(self):
        """ Compute k and mu at grid points. """
        if self._kgrid is None:
            with timer.Timer("kgrid time"):
                kk = []
                for i in range(self.dim):
                    kk.append(np.fft.fftfreq(self.shape[i])* 2 * np.pi / self.step[i])
                ky, kx, kz = np.meshgrid(*kk)
                self._kgrid = kx, ky, kz
        return self._kgrid

    @property
    def k(self):
        if self._k is None:
            with timer.Timer("k time"):
                ksq = 0
                for k in self.kgrid:
                    ksq += k * k
                self._k = np.sqrt(ksq)
                self._k.flat[0] = self._k.flat[1]
        return self._k

    @property
    def pkgrid(self):
        if self._pkgrid is None:
            with timer.Timer("pkgrid time"):
                p = self.pk_model.get_pk(self.k)
                p /= self.cell_volume

                p.flat[0] = 0
                p = p.reshape(self.shape)

                if self.apply_window:
                    p *= self.window

                if self.lognorm:
                    p = logtransform(p)

                # compute variance of the Gaussian field
                self.xi0 = np.sum(p)/self.n

                logging.debug("Variance of Gaussian field: %f (from pk)",self.xi0)

                self._pkgrid = p
                self.S0 = np.sum(self._pkgrid) / self.volume

        return self._pkgrid

    @property
    def window(self):
        if self._window is None:
            with timer.Timer("Window calculation"):
                w = np.ones(self.kgrid[0].shape)
                for i, k in enumerate(self.kgrid):
                    x = k * self.step[i] / 2
                    ii = x != 0
                    w[ii] *= np.sin(x[ii])/x[ii]
                self._window = w * w
        return self._window

    def realize(self):
        """realize a random gaussian field"""
        self.reset()


        with timer.Timer("Dreaming of gaussian fields (shape: %s)"%self.shape):
            # random.uniform returns in [0,1)
            # so to exclude the 0 case use 1-uniform
            amp = 1 - np.random.uniform(0, 1, self.n)

            phase = np.random.uniform(0, 2 * np.pi, self.n)

            x = np.sqrt(-2*np.log(amp))*np.exp(1j*phase)

            del amp
            del phase

            x = x.reshape(self.shape)

            grid = np.sqrt(1./self.n*self.pkgrid)*x

            del x

            grid.flat[0] = 0

            out = gofftinv(grid).real.astype('float')

            del grid

            logging.debug("Gauss field min:%f max:%f mean:%f var:%f",out.min(),out.max(),out.mean(),out.var())

            if self.lognorm:
                self._delta = np.exp(out-self.xi0/2.) - 1
            else:
                self._delta = out

        return self._delta

    @property
    def density(self):
        if self._delta is None:
            self._delta = self.realize()
        return self._delta

    @property
    def density_k(self):
        if self._density_k is None:
            with timer.Timer("density k"):
                f = self.pk_model.params['f']
                vel_norm = 1j * f / self.k**2 / self.pk_model.params['bias']
                self._density_k = vel_norm * gofft(self.density)
                self._density_k.flat[0] = 0
            del self._delta
        return self._density_k

    def velocity_component(self, axis=0):
        """ """
        return gofftinv(self.density_k * self.kgrid[axis]).real


    @property
    def velocity_field(self):
        if self._velocity_field is None:
            f = self.pk_model.params['f']
            z = self.pk_model.class_params['z_pk']
            Om = self.pk_model.class_params['Omega_cdm'] + self.pk_model.class_params['Omega_b']
            H = 100 * np.sqrt(Om * (1 + z)**3 + 1 - Om)

            logging.debug(f"Velocity field parameters z={z}, f={f}, H(z)={H}h km/s/Mpc")

            deltak = gofft(self.density)

            vfactor = 1j * f * deltak / self.k**2 / self.pk_model.params['bias']

            vfactor.flat[0] = 0

            self._velocity_field = [gofftinv(vfactor * kk).real for kk in self.kgrid]

        return self._velocity_field
