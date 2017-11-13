import numpy as np

def scale_min_max(y):
    """ Force the values of the array into [0,1] by min/max"""
    too_small = y<0
    too_high = y>1
    return y*(1-too_small)*(1-too_high) + too_high

def scale_linear(y):

    mini = np.min(y)
    maxi = np.max(y)

    return (y-mini)/(maxi-mini)

def gini(actual, pred, cmpcol=0, sortcol=1):
    """Compute Gini coefficient."""
    assert(len(actual) == len(pred))
    all = np.asarray(
        np.c_[actual, pred, np.arange(len(actual))],
        dtype=np.float)
    all = all[np.lexsort((all[:, 2], - 1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini_normalized(a, p):
    """Compute normalized Gini coefficient."""
    return gini(a, p) / gini(a, a)