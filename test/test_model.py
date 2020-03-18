import sys
import time
import pytest
import numpy as np

from synmock import model

from astropy import cosmology


@pytest.fixture
def Model():
    return model.ModelPk(z_pk=0, bias=2, non_linear='halofit', Omega_cdm=0.25, Omega_b=0.05, h=0.7)


@pytest.fixture
def AstropyModel():
    return cosmology.FlatLambdaCDM(H0=100, Om0=0.3)


def test_comoving_distance(Model, AstropyModel):
    """ """
    print("initialize",file=sys.stderr)
    zz = np.linspace(0.1, 2, 100)
    rr = np.zeros(len(zz))
    for i, z in enumerate(zz):
        rr[i] = Model.comoving_distance(z)

    r_ref = AstropyModel.comoving_distance(zz).value

    assert np.allclose(rr, r_ref, rtol=1e-3)


def test_comoving_distance_inverse(Model, AstropyModel):
    """ """
    rr = np.linspace(100, 3000, 10)
    t0 = time.time()
    zz = Model.redshift_at_comoving_distance(rr)
    print("time", time.time()-t0, file=sys.stderr)

    rr = np.linspace(100, 3000, 10)
    t0 = time.time()
    zz = Model.redshift_at_comoving_distance(rr)
    print("time", time.time()-t0, file=sys.stderr)

    t0 = time.time()
    r_ref = AstropyModel.comoving_distance(zz).value
    print("time", time.time()-t0, file=sys.stderr)

    assert np.allclose(rr, r_ref, rtol=1e-3)

def test_pk(Model):
    """ """
    kk = np.logspace(-2,0,100)
    pk_ref = Model.class_pk(kk)

    pk_test = Model.get_pk(kk)

    assert np.allclose(pk_test, pk_ref, rtol=1e-3)
