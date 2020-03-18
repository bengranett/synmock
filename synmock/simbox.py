import logging
import numpy as np
import time

from . import fftutils


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

    logging.debug("> Computing log power spectrum ~~~~~~~~~~")
    xi = np.fft.ifftn(p.astype('complex'))

    if not np.all(xi.real>-1):
        logging.critical("!!!! simbox fatal error with log transform! P(k) amp is too high maybe...")
        raise ValueError

    logxi = np.log(1 + xi)
    plog = np.fft.fftn(logxi).real.astype('float')
    plog.flat[0] = 0

    # Set negative modes to 0
    pmin = plog.min()
    if pmin < 0:
        logging.warning("log pk goes negative! %f",pmin)
        plog[plog<0] = 0

        if return_corrected:
            # Do the inverse transform to compute the corrected input power spectrum
            p_corrected = inv_logtransform(plog)

    if return_corrected:
        return plog, p_corrected

    return plog


class SimBox:
    """ """
    def __init__(self, pk_model, shape, length, lognorm=False):
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

        self._delta = None
        self._pkgrid = None
        self._kgrid = None
        self._k = None
        self._window = None

    @property
    def kgrid(self):
        """ Compute k and mu at grid points. """
        if self._kgrid is None:
            if self.dim==3:
                kgrid = fftutils.kgrid3d(self.shape, self.length) # components of k in physical units
            elif self.dim==2:
                kgrid = fftutils.kgrid2d(self.shape, self.length)
            elif self.dim==1:
                k = fftutils.kgrid1d(self.shape, self.length)
                kgrid = k,
            else:
                logging.critical("Invalid dimension: %s", self.dim)
                raise Exception("Invalid dimension")

            self._kgrid = kgrid

        return self._kgrid

    @property
    def k(self):
        if self._k is None:
            ksq = 0
            for k in self.kgrid:
                ksq += k * k
            self._k = np.sqrt(ksq)
            self._k.flat[0] = self._k.flat[1]
            assert np.all(self._k > 0)
        return self._k

    @property
    def pkgrid(self):
        if self._pkgrid is None:
            p = self.pk_model.get_pk(self.k)
            p /= self.cell_volume

            p.flat[0] = 0
            p = p.reshape(self.shape)

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
            w = np.ones(self.kgrid[0].shape)
            for i, k in enumerate(self.kgrid):
                x = k * self.step[i] / 2
                ii = x != 0
                w[ii] *= np.sin(x[ii])/x[ii]
            self._window = w * w
        return self._window

    def realize(self):
        """realize a random gaussian field"""
        self._velocity_field = None

        logging.debug("Dreaming of gaussian fields (shape: %s)",self.shape)
        t0 = time.time()

        # random.uniform returns in [0,1)
        # so to exclude the 0 case use 1-uniform
        amp = 1 - np.random.uniform(0, 1, self.n)

        phase = np.random.uniform(0, 2 * np.pi, self.n)

        x = np.sqrt(-2*np.log(amp))*np.exp(1j*phase)
        x = x.reshape(self.shape)

        grid = np.sqrt(1./self.n*self.pkgrid)*x

        grid.flat[0] = 0

        t1 = time.time()
        out = fftutils.gofftinv(grid).real.astype('float')

        logging.debug("fft time: %f",time.time()-t1)
        logging.debug("done, seconds: %f",time.time()-t0)

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
    def velocity_field(self):
        if self._velocity_field is None:
            f = self.pk_model.params['f']
            z = self.pk_model.class_params['z_pk']
            Om = self.pk_model.class_params['Omega_cdm'] + self.pk_model.class_params['Omega_b']
            H = 100 * np.sqrt(Om * (1 + z)**3 + 1 - Om)

            logging.debug(f"Velocity field parameters z={z}, f={f}, H(z)={H}h km/s/Mpc")

            deltak = fftutils.gofft(self.density)

            self.k.flat[0] = 1 # remove (0,0,0)

            vfactor = 1j * f * deltak / self.k**2 / self.pk_model.params['bias']

            vfactor.flat[0] = 0

            self._velocity_field = [fftutils.gofftinv(vfactor * kk).real for kk in self.kgrid]

        return self._velocity_field
