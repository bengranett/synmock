import sys
import numpy as N
import time

import scipy


def powerspectrum(grid, length, mask=None, zeropad=None, norm=1, getdelta=False,computek=True,
                  nthreads=1):
    """ Compute the power spectrum
    Inputs:
      grid -- delta values 1, 2 or 3 dimensions
      length -- physical dimensions of box
    Outputs:
      k, pk
    """
    shape = grid.shape
    dim = len(shape)

    if not zeropad==None:
        bigbox = N.zeros(N.array(grid.shape)*zeropad)
        if dim==3: bigbox[:grid.shape[0],:grid.shape[1],:grid.shape[2]] = grid
        if dim==2: bigbox[:grid.shape[0],:grid.shape[1]] = grid

        bigmask = None
        if not mask==None:
            bigmask = N.zeros(N.array(grid.shape)*zeropad)
            bigmask[:grid.shape[0],:grid.shape[1],:grid.shape[2]] = mask

        return powerspectrum(bigbox,N.array(length)*zeropad, mask=bigmask, zeropad=None, getdelta=getdelta, norm=zeropad**3, nthreads=nthreads,
                             computek=computek)

    if N.shape(length)==():          # True if length is a number
        length = N.ones(dim)*length  # create a list

    assert(len(length)==dim)

    t0 = time.time()
    dk = gofft(grid, nthreads=nthreads)

    dk *= N.sqrt(N.prod(length)*norm)

    if not mask==None:
        print ("no use of a mask is implemented!")

    pk = N.abs(dk**2)
    pk = pk.flatten()


    # save significant time if we dont need to recompute k
    if not computek:
        #print "skipping k comptuation"
        if getdelta:
            return pk, dk
        return pk
    if dim==3:
        kgrid = kgrid3d(shape, length)
    elif dim==2:
        kgrid = kgrid2d(shape, length)
    elif dim==1:
        kgrid = kgrid1d(shape, length)
    else:
        print("fftutils: bad grid dimension:",dim, file=sys.stderr)
        raise

    s = 0
    for i in range(dim):
        s += kgrid[i]**2
    k = s.flatten()**.5

    if getdelta:
        return k, pk, (kgrid, dk)

    return k, pk



def gofft_numpy(grid, nthreads=1):
    """ Forward FFT """
    n = N.prod(grid.shape)
    dk = 1./n*N.fft.fftn(grid)
    return dk

def gofftinv_numpy(grid, nthreads=1):
    """ inverse FFT """
    n = N.prod(grid.shape)
    d = n*N.fft.ifftn(grid)
    return d

gofft = gofft_numpy
gofftinv = gofftinv_numpy



kgridcache = {}
def kgrid3d(shape, length):
    """ Return the array of frequencies """
    key = '%s %s'%(shape[0],length[0])
    if key in kgridcache:
        # print "hitting up kgrid cache"
        return kgridcache[key]

    a = N.fromfunction(lambda x,y,z:x, shape)
    a[a >= shape[0]/2] -= shape[0]
    b = N.fromfunction(lambda x,y,z:y, shape)
    b[b >= shape[1]/2] -= shape[1]
    c = N.fromfunction(lambda x,y,z:z, shape)
    c[c >= shape[2]/2] -= shape[2]

    norm = 2*N.pi
    a = a*norm*1./length[0]
    b = b*norm*1./length[1]
    c = c*norm*1./length[2]

    kgridcache[key] = (a,b,c)

    return a,b,c

def kgrid2d(shape, length):
    """ Return the array of frequencies """

    a = N.fromfunction(lambda x,y:x, shape)
    a[a >= shape[0]/2] -= shape[0]
    b = N.fromfunction(lambda x,y:y, shape)
    b[b >= shape[1]/2] -= shape[1]

    norm = 2*N.pi
    a = a*norm*1./(length[0])
    b = b*norm*1./(length[1])

    return a,b

def kgrid1d(shape,length):
    """ Return the array of frequencies """
    a = N.arange(shape[0])
    a[a >= shape[0]/2] -= shape[0]
    a = a*2*N.pi*1./(length[0])
    return N.array([a])
