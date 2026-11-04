"""Finite Markov chains."""
from ._estimation import arfm2mctp, estsmctp, f_smctp
import numpy as np
from ._remaining import smc2rfm,mc2rfm,smctp2arfm,mctp2arfm,scalemat
from ._plotting import hmmplot


def _transition(P):
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1] or not np.all(np.isfinite(P)) or np.any(P < 0) or np.any(P.sum(axis=1) <= 0):
        raise ValueError("Expected a finite nonnegative square transition matrix")
    return P/P.sum(axis=1, keepdims=True)


def mc2stat(P):
    """Stationary probability vector. Nonunique stationary laws raise an error."""
    P = _transition(P)
    n = len(P)
    A = np.vstack((P.T-np.eye(n), np.ones(n)))
    if np.linalg.matrix_rank(A) < n:
        raise ValueError("Stationary distribution is not unique")
    p = np.linalg.lstsq(A, np.r_[np.zeros(n), 1.0], rcond=None)[0]
    p = np.maximum(p, 0)
    return p/p.sum()


from ._extended import (mat2tmat,estmc,chi2cmat,hdcmat,klcmat,loglcmat,
                       tr_p2x,tr_x2p,tr_m2x,tr_x2m,mktestmat,mctp2stat,
                       mctp2reverse,smctp2rfm,mctp2rfm,smctp2joint,smctp2stat,f_funm)
