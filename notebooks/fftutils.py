import sys
import numpy as np


def powerspectrum(grid, length):
    """ Compute the power spectrum
    Inputs:
      grid -- delta values 1, 2 or 3 dimensions
      length -- physical dimensions of box
    Outputs:
      k, pk
    """
    shape = grid.shape
    dim = len(shape)

    if np.shape(length)==():          # True if length is a number
        length = np.ones(dim)*length  # create a list

    step = length / shape    
    
    assert(len(length)==dim)

    dk = gofft(grid)

    dk *= np.sqrt(np.prod(length))

    pk = np.abs(dk**2)
    
    pk = pk.flatten()
    
    return pk


def gofft_numpy(grid, nthreads=1):
    """ Forward FFT """
    n = np.prod(grid.shape)
    dk = 1./n*np.fft.fftn(grid)
    return dk

def gofftinv_numpy(grid, nthreads=1):
    """ inverse FFT """
    n = np.prod(grid.shape)
    d = n*np.fft.ifftn(grid)
    return d

gofft = gofft_numpy
gofftinv = gofftinv_numpy
