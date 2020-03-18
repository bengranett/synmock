import numpy as np
import healpy

ic = 180/np.pi


def xyz2lonlat(x,y,z,norm=True,getr=False):
    """ """
    if norm:
        r = np.sqrt(x*x+y*y+z*z)
    else:
        r = np.ones(x.shape)
    ii = r > 0

    lat = np.zeros(len(r), dtype=float)
    lon = np.zeros(len(r), dtype=float)

    lat[ii] = np.arcsin(z[ii]/r[ii])*ic
    lon[ii] = np.arctan2(y[ii],x[ii])*ic

    lon = lon%360

    if getr:
        return lon,lat,r
    return lon,lat


def expand_mask(mask, fwhm=1, threshold=0.1):
    """ """
    mask_sm = healpy.smoothing(mask, fwhm=fwhm*np.pi/180)
    mask_out = mask * 0
    sel = mask_sm > threshold
    mask_out[sel] = 1
    return mask_out