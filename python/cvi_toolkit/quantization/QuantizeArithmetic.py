from functools import partial, wraps
from math import floor, ceil, frexp
import numpy as np
import logging
from ..utils.log_setting import setup_logger

logger = setup_logger('root')
log_flag = logger.level <= logging.INFO

def getRShiftForFilter(max_filter, threshold_x, threshold_y):
    a = max_filter * threshold_x / threshold_y
    if a > 127:
        print("WARNING: findRShiftForFilter, max_filter too large {}".format(max_filter))
        return 0
    rshift = 0
    for _ in range(32):
        if a * (1 << rshift) >= 64:
            return rshift
        rshift += 1
    print("WARNING: findRShiftForFilter, max_filter too small {}".format(max_filter))
    return 0


def getFilterQscale(Sw, Sx, Sy):
    """
        Sx:threshold_x/127
        Sy:threshold_y/127
    """
    return (Sw * Sx) / Sy


def QuantMultipiler(double_multiplier, qdm=False):
    q, shift = frexp(double_multiplier)
    q_fixed = round(q * (1 << 31))
    if q_fixed == (1 << 31):
        q_fixed /= 2
        shift += 1
    if shift < -31:
        q_fixed = 0
        shift = 0

    q_fixed = np.int32(q_fixed).item()  # cast it to 32 bit
    return q_fixed, shift


def getRShiftAndMultiplierFromQScale(Qscale, qdm=False, max_multiplier=127):
    """
        Qscale: 2^rshift * m0(mutlipiler)
    """
    if qdm:
        multiplier, lshift = QuantMultipiler(Qscale)
        rshift = -lshift
        rshift = np.int8(rshift).item()  # cast it to 8 bit
        if rshift < 0:
            raise RuntimeError("rshift less than 0")
        return rshift, multiplier
    else:
        if Qscale > 127:
            raise RuntimeError(
                "Qscale exceed max_multipiler {}".format(max_multiplier))

        rshift = 0
        for _ in range(63):
            if Qscale * (1 << (rshift + 1)) >= max_multiplier:
                multiplier = Qscale * (1 << rshift)
                return rshift, multiplier
            rshift += 1


def getRShiftAndMultiplierFromQScaleArray(QscaleArray, qdm=False):
    rshiftArray = list()
    multiplierArray = list()
    for q in QscaleArray:
        r, m = getRShiftAndMultiplierFromQScale(q, qdm)
        rshiftArray.append(r)
        multiplierArray.append(m)
    return np.array(rshiftArray), np.array(multiplierArray)


def getMultiplierI8FromQScaleAndRShift(Qscale, rshift):
    return np.int8(Qscale * (1 << rshift)).item()  # cast it to 8 bit
