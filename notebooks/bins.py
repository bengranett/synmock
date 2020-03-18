import numpy as np

def binit(x, y, bands, count=False):
    """ """

    out = np.zeros(len(bands)-1,dtype='d')
    n = out.copy()
    sig = out.copy()
    for i in range(len(out)):
        ii = (x >= bands[i]) & (x < bands[i+1])
        sub = y[ii]
        if sub.size > 0:
            out[i] = np.mean(sub)
            sig[i] = np.std(sub)
            n[i] += sub.size

    if count:
        return out,sig,n
    return out
