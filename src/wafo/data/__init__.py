"""Original WAFO measurement assets, loaded without running MATLAB."""
from importlib.resources import files
import numpy as np
from scipy.io import loadmat


def available():
    return sorted(p.name for p in files(__package__).iterdir() if p.name.endswith(('.dat','.mat','.asc')))


def load(name='sea.dat'):
    """Load a bundled dataset. MAT files return a dict of variables."""
    if name not in available():
        raise ValueError(f"Unknown dataset {name!r}; use available()")
    resource = files(__package__).joinpath(name)
    with resource.open('rb') as f:
        return loadmat(f) if name.endswith('.mat') else np.loadtxt(f)
