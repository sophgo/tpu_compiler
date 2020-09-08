from __future__ import division
import math
import numpy as np

def cal_mse(x, y):
    return ((x - y)**2).mean()

def cal_cross_entropy(x, y):
    """ Computes cross entropy between two distributions.
    Input: x: iterabale of N non-negative values
           y: iterabale of N non-negative values
    Returns: scalar
    """

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError('Negative values exist.')

    # Force to proper probability mass function.
    x = np.array(x, dtype=np.float)
    y = np.array(y, dtype=np.float)
    x /= np.sum(x)
    y /= np.sum(y)

    # Ignore zero 'y' elements.
    mask = y > 0
    x = x[mask]
    y = y[mask]    
    ce = -np.sum(x * np.log(y)) 
    return ce

def cal_sigmoid(x):
    return 1. / (1. + np.exp(-x))

def cal_softmax(x, t=-100.):
    x = x - np.max(x)
    if np.min(x) < t:
        x = x / np.min(x) * t
    exp_x = np.exp(x)
    out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    return out

def cal_sqnr(signal_raw, signal_dequant, remove_zero=True):
    # SQNR is non-commutative
    # Unlike other distance function
    # Cannot change the order of signal_raw and signal_dequant
    raw = signal_raw.flatten()
    dequant = signal_dequant.flatten()

    if remove_zero is True:
        idx = dequant != 0
        raw = raw[idx]
        dequant = dequant[idx]

    noise = raw - dequant

    avg_raw = np.sum(raw) / raw.size
    avg_noise = np.sum(noise) / noise.size

    raw_zero_mean = raw - avg_raw
    noise_zero_mean = noise - avg_noise

    var_raw_zero_mean = np.sum(np.square(raw_zero_mean))
    var_noise_zero_mean = np.sum(np.square(noise_zero_mean))

    if var_noise_zero_mean == 0.0:
        return math.inf

    sqnr = 10 * np.log10(var_raw_zero_mean / var_noise_zero_mean)

    return sqnr


if __name__ == '__main__':
    raw = np.array([5.2, -4.1, 3.14, -2.1, 3.8, -1.2])
    dequant = np.array([5, -4, 3, -2, 4, -1])
    print(cal_sqnr(raw, dequant))
