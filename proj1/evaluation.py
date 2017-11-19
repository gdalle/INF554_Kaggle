import numpy as np

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

def gini_scorer(a,p):
    """Compute normalized Gini coefficent with the probabilities of the two classes """

    return gini_normalized(a,p[:,1])