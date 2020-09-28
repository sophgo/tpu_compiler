from functools import partial, wraps
from math import floor, ceil, frexp
import numpy as np
import logging
from ..utils.log_setting import setup_logger

logger = setup_logger('root')
log_flag = logger.level <= logging.INFO

def saturateInt8(float_value):
    if isinstance(float_value, (float, int)):
        q = np.clip(np.floor(float_value+0.5),
               np.iinfo(np.int8).min, np.iinfo(np.int8).max)
        return q
    elif isinstance(float_value, np.ndarray):
        q = np.clip(np.floor(float_value+0.5),
                    np.iinfo(np.int8).min, np.iinfo(np.int8).max)
        return q
    else:
        raise TypeError("{} not support ".format(type(float_value)))


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
        Sx: threshold_x/127
        Sy: threshold_y/127
        Qscale = Sw * Sx / Sy
    """
    return (Sw * Sx) / Sy


def quantizeFilterRShift(weight, threshold_y, threshold_x, rshift):
    factor = (threshold_x / threshold_y) * (1 << rshift)
    quant_weight = weight * factor
    return saturateInt8(quant_weight)


def QuantMultipiler(double_multiplier):
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
        qdm mode:
            qdm is true
            reference to [arxiv 1712.05877]
            choose the int32 value nearest to 2^31 * M0, M0 in [0.5, 1]
            this value is always at least 2^30 and have at least 30 bits accuracy
            the max_multiplier argument is ignored, fixed to (1 << 31)
    """
    if qdm:
        multiplier, lshift = QuantMultipiler(Qscale)
        rshift = -lshift
        rshift = saturateInt8(rshift)  # cast it to 8 bit
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
    return saturateInt8(Qscale * (1 << rshift))
