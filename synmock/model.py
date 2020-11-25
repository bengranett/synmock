import logging
import numpy as np
from classy import Class
from scipy import interpolate

# default parameters
class_params = {
    'output': 'mPk',
    'non linear': 'halofit',
    'P_k_max_h/Mpc': 300.,
    'z_pk': 0.7,
    'A_s': 2.3e-9,
    'n_s': 0.96,
    'h': 0.7,
    'Omega_b': 0.05,
    'Omega_cdm': 0.25,
}

params = {
    'sigma8': 0.8,
    'bias': 1.,
    'f': 0,
    'fix_sigma8': False,
    'log_kmin': -4,
    'log_kmax': 2,
    'log_ksteps': 10000,
    'logzmax': 1,
    'logzmin': 0,
    'logzsteps': 1000,
    'mpk': None
}


class ModelPk(object):
    """ """
    logger = logging.getLogger(__name__)

    def __init__(self, **param_dict):
        """ """
        self.params = params.copy()

        self.class_params = class_params.copy()
        self.set(**param_dict) #other params for CLASS

        self._cosmo = None
        self._redshift_func = None
        self._logpk_func = None

        if self.params['mpk'] is not None:
            self.load_pk_from_file(self.params['mpk'])

    def load_pk_from_file(self, path):
        """ """
        self.logger.info(f"Loading matter power spectrum from file {path}")
        k, pk = np.loadtxt(path, unpack=True)
        self.logger.info(f"min k: {k.min()}, max k: {k.max()}, steps {len(k)}")
        pk *= self.params['bias']**2
        lk = np.log(k)
        lpk = np.log(pk)
        self._logpk_func = interpolate.interp1d(lk, lpk, bounds_error=False, fill_value=(0,0))

    def set(self, **param_dict):
        """ """
        for key,value in param_dict.items():
            if key == 'non_linear':
                key = 'non linear'
            self.logger.debug("Set %s=%s",key,value)
            found = False
            if key in self.class_params:
                self.class_params[key] = value
                found = True
            if key in self.params:
                self.params[key] = value
                found = True
            if not found:
                continue

    @property
    def cosmo(self):
        """ """
        if not self._cosmo:
            self._cosmo = Class()
            self._cosmo.set(self.class_params)

            self.logger.info("Initializing Class")
            self._cosmo.compute()

            if self.params['fix_sigma8']:
                sig8 = self._cosmo.sigma8()
                A_s = self._cosmo.pars['A_s']
                self._cosmo.struct_cleanup()
                # renormalize to fix sig8
                self.A_s = A_s*(self.params['sigma8']*1./sig8)**2
                self._cosmo.set(A_s=self.A_s)
                self._cosmo.compute()

            sig8 = self._cosmo.sigma8()

            self.params['sigma8'] = sig8
            self.params['A_s'] = self._cosmo.pars['A_s']
            self.params['sigma8z'] = sig8 * self._cosmo.scale_independent_growth_factor(self.class_params['z_pk'])
            self.params['f'] = self._cosmo.scale_independent_growth_factor_f(self.class_params['z_pk'])

            self.logger.info(f"          z: {self.class_params['z_pk']}")
            self.logger.info(f"    sigma_8: {self.params['sigma8']}")
            self.logger.info(f" sigma_8(z): {self.params['sigma8z']}")
            self.logger.info(f"       f(z): {self.params['f']}")
            self.logger.info(f"f sigma8(z): {self.params['f']*self.params['sigma8z']}")
        return self._cosmo

    def class_pk(self, k):
        """ """
        self.logger.info("Computing power spectrum with Class")

        z = np.array([self.class_params['z_pk']]).astype('d')

        shape = k.shape
        k = k.flatten()

        nk = len(k)
        k = np.reshape(k, (nk, 1, 1))

        pk = self.cosmo.get_pk(k * self.class_params['h'], z, nk, 1, 1).reshape((nk,))
        k = k.reshape((nk,))

        # set h units
        pk *= self.class_params['h']**3

        pk *= self.params['bias']**2

        pk = pk.reshape(shape)

        return pk

    def get_pk(self, k):
        ii = k > 0
        out = np.zeros(k.shape, dtype='d')
        out[ii] = np.exp(self.logpk_func(np.log(k[ii])))
        return out

    @property
    def logpk_func(self):
        if self._logpk_func is None:
            k = np.logspace(self.params['log_kmin'], self.params['log_kmax'], self.params['log_ksteps'])
            pk = self.class_pk(k)
            lk = np.log(k)
            lpk = np.log(pk)
            print("logk min", lk.min())
            self._logpk_func = interpolate.interp1d(lk, lpk, bounds_error=False, fill_value=(0,0))
        return self._logpk_func

    def comoving_distance(self, z):
        """ """
        return (1 + z) * self.cosmo.angular_distance(z) * self.class_params['h']

    def redshift_at_comoving_distance(self, r):
        """ """
        try:
            z = 10**self.redshift_func(r) - 1
        except ValueError:
            self.logger.error(f"r min {r.min()} max {r.max()}", file=sys.stderr)
            raise
        return z

    @property
    def redshift_func(self):
        """ """
        if self._redshift_func is None:
            zz = np.logspace(self.params['logzmin'], self.params['logzmax'], self.params['logzsteps']) - 1
            r = np.zeros(len(zz))
            for i, z in enumerate(zz):
                r[i] = self.comoving_distance(z)
            self._redshift_func= interpolate.interp1d(r, np.log10(1+zz))
        return self._redshift_func
