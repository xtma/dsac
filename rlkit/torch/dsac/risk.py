import numpy as np

import rlkit.torch.pytorch_util as ptu
from scipy.interpolate import interp1d
from scipy.stats import norm


def risk_fn(tau, mode="neutral", param=0):
    if param >= 0:
        if mode == "neutral":
            return tau
        elif mode == "wang":
            x = np.linspace(0., 1., num=100, endpoint=True)
            y = norm.cdf(norm.ppf(x) + param)
            fn = interp1d(y, x, kind='linear')
            return ptu.from_numpy(fn(ptu.get_numpy(tau).clip(0., 1.)))
        elif mode == "cvar":
            return (1. / param * tau).clamp(0., 1.)
        elif mode == "icvar":
            return (1. / (1. - param) * tau).clamp(0., 1.)
        elif mode == "pow":
            return tau.clamp(0., 1.).pow(1. + param)
        elif mode == "cpw":
            x = np.linspace(0., 1., num=100, endpoint=True)
            y = x**param / (x**param + (1. - x)**param)**(1. / param)
            fn = interp1d(y, x, kind='linear')
            return ptu.from_numpy(fn(ptu.get_numpy(tau).clip(0., 1.)))
    else:
        return 1 - risk_fn(1 - tau, mode, -param)
