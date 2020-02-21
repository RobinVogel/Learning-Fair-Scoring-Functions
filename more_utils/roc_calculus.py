import math
import numpy as np


def fpr_tpr(scores, classes):
    """
        Computes the ROC curves from scores and classes of comparisons.
        Does not work if the distributions of the scores are not continuous.
    """
    scores = np.array(scores)
    classes = (np.array(classes) == +1).astype(int)

    n_pos = classes.sum()
    n_neg = classes.shape[0] - n_pos

    ind_sort = np.argsort(scores)  # argsort, increasing order
    cl_sort = classes[ind_sort]

    fnr_ar = cl_sort.cumsum()/n_pos
    tnr_ar = (1-cl_sort).cumsum()/n_neg
    fpr_ar = 1 - tnr_ar

    return fpr_ar, 1 - fnr_ar


def auc(scores, classes, B=None):
    if B is None:
        s = scores.reshape((-1, 1))
        y = classes.reshape((-1, 1))
        ind_0, ind_1 = np.where((y-y.transpose()) > 0)
        delta_s = s[ind_0] - s[ind_1]
    else:
        s = scores
        y = classes
        f_p = (y == 1)
        s_p, s_n = s[f_p], s[~f_p]
        delta_s = np.random.choice(s_p, B) - np.random.choice(s_n, B)
    return (delta_s > 0).mean() + 0.5*(delta_s == 0).mean()


def pointwise_tpr(scores, classes, alpha):
    filt_pos = classes == 1
    n_pos = (filt_pos).sum()
    n_neg = classes.shape[0] - n_pos

    # Increasing order.
    sorted_neg_scores = np.sort(scores[~filt_pos])
    # Score for FPR target.
    threshold = sorted_neg_scores[math.floor(n_neg*(1-alpha))]
    return (scores[filt_pos] > threshold).mean()


def get_auc_from_roc(fpr, tpr):
    # FPR should be increasing
    auc = 0
    n = len(fpr)
    for i in range(0, n-1):
        if fpr[i+1]-fpr[i] > 0:
            auc += ((tpr[i+1]+tpr[i])/2)*(fpr[i+1]-fpr[i])
    return auc


def get_pointwise_from_roc(fpr, tpr, pt):
    # FPR should be increasing
    i = np.where(fpr > pt)[0][0]
    if fpr[i+1]-fpr[i] > 0:
        return tpr[i] + ((tpr[i+1]-tpr[i])/(fpr[i+1]-fpr[i]))*(pt-fpr[i])
    else:
        return (tpr[i] + tpr[i+1])/2


def get_roc_fixed_step(fpr, tpr, x_values):
    res = list()
    i = 0
    for x in x_values:
        found = False
        while not found:
            if x == 1:
                found = True
                res.append(1)
            if fpr[i] <= x < fpr[i+1]:
                found = True
                slope = ((tpr[i+1] - tpr[i])/(fpr[i+1]-fpr[i]))
                val = tpr[i] + slope*(x-fpr[i])
                res.append(val)
            elif x == fpr[i+1]:
                found = True
                res.append(tpr[i+1])
            else:
                i += 1
    return res


def quantile(X, q, axis=0):
    """np.quantile only exists on numpy 1.15 and higher."""
    assert axis == 0
    X = np.array(X)
    return np.sort(X, axis=0)[int(X.shape[0]*q), :]
