import os
import sys
import logging
import argparse
import numpy as np
import yaml
import pandas
from scipy import interpolate
import healpy

from . import __version__
from . import model, lncat, timer


default_params = {
    'z_pk': 0,
    'bias': 1.5,
    'non_linear': 'halofit',
    'P_k_max_h/Mpc': 300.,
    'A_s': 2.3e-9,
    'n_s': 0.96,
    'h': 0.7,
    'Omega_b': 0.05,
    'Omega_cdm': 0.25,
    'sigma8': 0.8,
    'fix_sigma8': False,

    'out': 'out/lncat_%03i.txt',
    'nreal': 1,
    'start': 0,

    'box': 1024,
    'cellsize': 16,
    'nbar': 1e-3,

    'mask': None,
    'nofz': None,
    'sky': False,

    'randoms': False,
    'randoms_factor': 10,
    'randoms_file': 'out/randoms.txt',

    'seed': None,

    'verbose': 1,
}

params_to_write = ('h', 'Omega_cdm', 'Omega_b', 'sigma8', 'A_s', 'n_s', 'non_linear', 'P_k_max_h/Mpc', 'bias', 'z_pk', 'box', 'cellsize', 'mask', 'nofz', 'randoms', 'randoms_factor', 'seed')

def write_header(params, file):
    """ """
    print(f"# This is a lognormal galaxy catalog generated with Synmock v{__version__}", file=file)
    print("# configuration", file=file)
    for key in params_to_write:
        print(f"#    {key} = {params[key]}", file=file)

def write(cats, path, params, model, labels=None):
    """ Write catalog to a file """
    logging.info(f"Writing catalog with size {cats[0].shape[0]} to {path}")

    params['A_s'] = model.params['A_s']
    params['sigma8'] = model.params['sigma8']

    table = np.hstack(cats)

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(path, "w") as out:
        write_header(params, out)
        print("#", file=out)
        print("# " + " ".join(labels), file=out)
        pandas.DataFrame(table).to_csv(out, sep=" ", header=None, index=False)


def load_mask(params):
    """ """
    if params['mask'] is not None:
        mask = healpy.read_map(params['mask'])
        params['sky'] = True
    else:
        mask = None
    return mask

def load_nofz(params):
    """ """
    if params['nofz'] is not None:
        z, n = np.loadtxt(params['nofz'], unpack=True)
        nofz = interpolate.interp1d(z, n, bounds_error=False, fill_value=0)
        params['nbar'] = n.max()
        zmean = np.sum(z*n)/np.sum(n)
        logging.info(f"Loaded n(z) from {params['nofz']}.")
        logging.info(f"zmin={z.min()}, zmax={z.max()}, zmean={zmean}")
        logging.info(f"nmax = {params['nbar']} (h/Mpc)^3")
        params['sky'] = True
    else:
        nofz = None
    return nofz


def go_randoms(M, params, mask=None, nofz=None):
    """ """
    path = params['randoms_file']
    if os.path.exists(path):
        return

    with timer.Timer("go randoms time"):
        L = lncat.LogNormCat(
            M,
            random=True,
            random_factor=params['randoms_factor'],
            mask=mask,
            nofz=nofz,
            box_length=params['box'],
            cell_size=params['cellsize'],
            nbar=params['nbar']
        )

        write((L.skycoord,), path, params, M, labels=('ra','dec','z'))


def go(params):
    """ Run the program """

    n_seed = params['nreal']
    if params['randoms']:
        n_seed += 1
    np.random.seed(params['seed'])
    seeds = list(np.random.randint(2**32, size=n_seed, dtype='uint32'))
    if params['seed'] is not None:
        seeds[-1] = params['seed']

    M = model.ModelPk(**params)

    mask = load_mask(params)
    nofz = load_nofz(params)

    if params['randoms']:
        params['seed'] = seeds.pop()
        np.random.seed(params['seed'])
        go_randoms(M, params, mask=mask, nofz=nofz)

    L = lncat.LogNormCat(M, mask=mask, nofz=nofz, box_length=params['box'], cell_size=params['cellsize'], nbar=params['nbar'])

    stop_flag = False

    for loop in range(params['nreal']):
        with timer.Timer("Realization time"):
            params['seed'] = seeds.pop()

            index = params['start'] + loop
            try:
                path = params['out']%index
            except:
                path = params['out']
                stop_flag = True

            if os.path.exists(path):
                continue

            logging.info(f"Starting realization {loop+1} of {params['nreal']}")

            np.random.seed(params['seed'])

            L.realize()

            if not params['sky']:
                write((L.catalog, L.catalog_velocity), path, params, M, labels=('x','y','z','vx','vy','vz'))
            else:
                write((L.skycoord,), path, params, M, labels=('ra','dec','z','z_s'))

            if stop_flag:
                logging.warning("Stopping early")
                break

    logging.info("Done")


def write_config(params, file=sys.stdout):
    """ write params to screen """
    print(yaml.dump(params), file=file)

def read_config(path):
    """ Read config file """
    with open(path) as file:
        params = yaml.load(file, Loader=yaml.SafeLoader)
    return params

def main():
    """ Executable """
    parser = argparse.ArgumentParser(description='LNMock - Build Lognormal Galaxy Catalogs',
                                    argument_default=argparse.SUPPRESS)

    parser.add_argument('-c', '--config', metavar='PATH', type=str,
                        help='Config file')

    parser.add_argument('-w', action='store_true', help='write config file to screen and exit')

    parser.add_argument('--out', metavar='PATH', type=str,
                        help='output path')

    parser.add_argument('--nreal', metavar='N', type=int,
                        help='Number of independent realizations to generate')

    parser.add_argument('--start', metavar='N', type=int,
                        help='Starting index (default 0)')

    parser.add_argument('--box', metavar='l', type=float,
                        help='Physical size of the cubic box in Mpc/h')

    parser.add_argument('--cellsize', metavar='x', type=float,
                        help='Cell size in Mpc/h')

    parser.add_argument('--nbar', metavar='n', type=float,
                        help='Target galaxy number density in (Mpc/h)^-3')

    parser.add_argument('--veldisp', metavar='v', type=float,
                        help='Velocity dispersion in km/s')

    parser.add_argument('--nofz', metavar='PATH', type=str,
                        help='Path to file with tabulated nbar(z)')

    parser.add_argument('--mask', metavar='PATH', type=str,
                        help='Path to healpix mask')

    parser.add_argument('--sky', type=int,
                        help='Output sky coordinates (RA, Dec, z, zs) instead of (x,y,z,vx,vy,vz)')

    parser.add_argument('--randoms', type=int,
                        help='Generate random catalog'),

    parser.add_argument('--randoms_factor', type=int,
                        help='Density of randoms with respect to galaxies (eg 10)'),

    parser.add_argument('--randoms_file', type=str,
                        help='Output file for randoms'),

    parser.add_argument('--seed', type=int, help='Random seed'),

    parser.add_argument('-v', '--verbose', type=int, help='verbosity level')

    args = parser.parse_args()


    params = default_params.copy()

    if 'config' in args:
        if os.path.exists(args.config):
            config = read_config(args.config)
            params.update(config)

    for key, value in vars(args).items():
        if key == 'w': continue
        if value is None: continue
        if key in params:
            params[key] = value

    if 'w' in args:
        if args.w:
            write_config(params)
            sys.exit(0)

    if params['verbose'] < 1:
        level = logging.CRITICAL
    elif params['verbose'] == 1:
        level=logging.INFO
    else:
        level=logging.DEBUG


    logging.basicConfig(level=level)


    for key, value in params.items():
        logging.info(f"{key} = {value}")

    with timer.Timer("Run time"):
        go(params)


if __name__ == "__main__":
    main()