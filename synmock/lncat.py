"""Summary
"""
import logging
import numpy as np
import healpy

from . import simbox, sphere

default_params = {
    'mask': None,
    'nofz': None,
    'box_length': 1024.,
    'nbar': 1e-4,
    'dim': 3,
    'cell_size': 8.,
    'grid_point_weights': False,
    'random': False,
    'random_factor': 10,
}

class LogNormCat:

    """"""

    def __init__(self, pk_model, config=None, **kwargs):
        """
        Parameters
        ----------
        pk_model : model.ModelPk
            Instance of ModelPk

        Keyword arguments
        -----------------
        mask : numpy.ndarray, optional
            Healpix binary mask
        nofz : function, optional
            Selection function
        box_length : float, optional
            Size of box in Mpc/h
        nbar : float, optional
            Mean density in (Mpc/h)^{-3}
        dim : int, optional
            Dimension, eg 3
        cell_size : float, optional
            Size of cell in Mpc/h
        grid_point_weights : bool, optional
            Output the number of galaxies at each grid point
        random : bool, optional
            Generate a random catalog
        random_factor : int, optional
            Number of randoms with respect to nbar
        """
        self.params = default_params.copy()
        if config:
            self.params.update(config)
        self.params.update(kwargs)

        box_length = np.ones(self.params['dim']) * self.params['box_length']

        shape = np.ceil(box_length * 1. / self.params['cell_size']).astype(int)

        assert np.all(shape > 0)

        self.pk_model = pk_model

        self.cell_vol = np.prod(box_length)*1./np.prod(shape)

        self.shape = shape
        self.length = np.array(box_length)
        self.step = self.length*1./self.shape

        self.nbar_grid = self.params['nbar'] * self.cell_vol

        if not self.params['random']:
            self.box = simbox.SimBox(pk_model, shape, box_length, lognorm=True)

        self._mask_exp = None

        if self.params['mask'] is not None:
            theta_grid = 10 * 180/np.pi * self.params['cell_size']/self.pk_model.comoving_distance(pk_model.class_params['z_pk'])
            logging.info(f"cell angular size {theta_grid} deg")
            theta_grid = max(10, theta_grid)
            self._nside = healpy.npix2nside(len(self.params['mask']))
            self._mask_exp = sphere.expand_mask(self.params['mask'], theta_grid)

        self._init()

    def _init(self):
        """"""
        self.delta = None
        self.galaxy_discrete = None
        self.coord = None
        self._velocity_field = None
        self.coord_vel = None
        self._skycoord = None
        self._grid_mask = None

    def realize(self):
        """
        Returns
        -------
        TYPE
            Description
        """
        self._init()
        if not self.params['random']:
            self.box.realize()
        return self.density

    @property
    def density(self):
        """
        Returns
        -------
        TYPE
            Description
        """
        if self.params['random']:
            return 0
        return self.box.density

    @property
    def velocity_field(self):
        """
        Returns
        -------
        TYPE
            Description
        """
        if self.params['random']:
            return 0
        return self.box.velocity_field

    @property
    def galaxy_count(self):
        """
        Returns
        -------
        TYPE
            Description
        """
        if self.galaxy_discrete is None:
            if self.params['random']:
                galaxy = self.grid_mask * self.params['random_factor']
            else:
                galaxy = self.grid_mask * (1 + self.density)

            self.galaxy_discrete = np.zeros(galaxy.shape, dtype='i')

            ii = galaxy > 0
            self.galaxy_discrete[ii] = np.random.poisson(galaxy[ii])

        return self.galaxy_discrete

    @property
    def catalog(self):
        """
        Returns
        -------
        TYPE
            Description
        """
        if self.coord is None:
            nonzero = np.where(self.galaxy_count > 0)
            count = self.galaxy_count[nonzero].flatten()

            xyz = np.transpose(nonzero) * self.step

            if not self.params['grid_point_weights']:

                self.coord = np.repeat(xyz, count, axis=0)

                # add a random offset to move galaxies off the regular grid points
                self.coord += np.random.uniform(0, self.step, self.coord.shape)
                self.weight = 1
            else:
                self.coord = xyz
                self.weight = count

        return self.coord

    @property
    def catalog_velocity(self):
        """ Retrieve the velocity vector from the nearest grid point
        of each mock galaxy.

        In the case of randoms, returns 0

        Returns
        -------
        numpy.ndarray
            (3, N) velocity array
        """
        if self.params['random']:
            return 0
        if self.coord_vel is None:
            # rebin and get the velocity
            b = np.floor(self.coord / self.step).astype(int)
            b = tuple(np.transpose(b))
            self.coord_vel = np.transpose([v[b] for v in self.velocity_field])

        return self.coord_vel

    def mask_sel(self, ra, dec, mask=None):
        """ Apply the angular mask to the sky coordinates.

        Parameters
        ----------
        ra : numpy.ndarray
            Right ascension or longitude coordinate
        dec : numpy.ndarray
            Declination or latitude coordinate
        mask : None, optional
            Healpix mask, if not given the stored mask will be used

        Returns
        -------
        numpy.ndarray
            boolean array indicating inside or outside the mask
        """
        if self.params['mask'] is not None:
            pix = healpy.ang2pix(self._nside, ra, dec, lonlat=True, nest=False)
            if mask is None:
                mask = self.params['mask']
            sel = mask[pix] > 0
            return sel
        else:
            return True

    def selecton_function(self, redshift):
        """ Compute the number density at the given redshift
        using the n(z) function or constant mean density as appropriate.

        Parameters
        ----------
        redshift : numpy.ndarray
            redshift

        Returns
        -------
        numpy.ndarray
            number density
        """
        if self.params['nofz'] is not None:
            return self.params['nofz'](redshift) * self.cell_vol
        return np.ones(len(redshift)) * self.nbar_grid

    @property
    def grid_mask(self):
        """grid_mask is the selection function applied on the grid
        """
        if self.params['nofz'] is None and self.params['mask'] is None:
            return self.nbar_grid

        if self._grid_mask is None:
            center = self.length / 2.
            ll = [np.arange(self.shape[i]) * self.step[i] for i in range(3)]
            y, x, z = np.meshgrid(*ll)
            x = x.flatten() - center[0]
            y = y.flatten() - center[1]
            z = z.flatten() - center[2]
            ra, dec, r = sphere.xyz2lonlat(x,y,z, getr=True)
            redshift = self.pk_model.redshift_at_comoving_distance(r)

            sel = self.mask_sel(ra, dec, self._mask_exp)
            selfunc = self.selecton_function(redshift[sel])

            self._grid_mask = np.zeros(len(x))
            self._grid_mask[sel] = selfunc
            self._grid_mask = self._grid_mask.reshape(self.shape)
        return self._grid_mask

    @property
    def skycoord(self):
        """Return  angular coordinates RA, Dec projected on the sky and redshift

        If generating a galaxy catalog, the output array will contain
        RA, Dec, z, z_s

        In the case of a random catalog, the output array contains
        RA, Dec, z

        Returns
        -------
        numpy.ndarray
            array containing the sky coordinates and redshift
        """
        if self._skycoord is None:
            center = self.length / 2.
            xyz = self.catalog - center
            x, y, z = np.transpose(xyz)
            ra, dec, r = sphere.xyz2lonlat(x, y, z, getr=True)

            if not self.params['random']:
                vel = self.catalog_velocity
                norm = np.sqrt(np.sum(xyz * xyz, axis=1))
                vlos = np.sum(xyz * vel, axis=1) / norm

            if self.params['mask'] is not None:
                sel = self.mask_sel(ra, dec)
                ra = ra[sel]
                dec = dec[sel]
                r = r[sel]
                if not self.params['random']:
                    vlos = vlos[sel]

            redshift = self.pk_model.redshift_at_comoving_distance(r)

            if self.params['random']:
                self._skycoord = np.transpose([ra, dec, redshift])
            else:
                redshift_s = self.pk_model.redshift_at_comoving_distance(r + vlos)
                self._skycoord = np.transpose([ra, dec, redshift, redshift_s])

        return self._skycoord
