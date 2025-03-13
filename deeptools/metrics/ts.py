import numpy as np

def prep_clf(obs, pre, threshold=0.1):
    # 根据阈值分类为 0, 1
    obs = np.where(obs >= threshold, 1, 0)
    pre = np.where(pre >= threshold, 1, 0)
    # True positive (TP)
    hits = np.sum((obs == 1) & (pre == 1))
    # False negative (FN)
    misses = np.sum((obs == 1) & (pre == 0))
    # False positive (FP)
    falsealarms = np.sum((obs == 0) & (pre == 1))
    # True negative (TN)
    correctnegatives = np.sum((obs == 0) & (pre == 0))
    return hits, misses, falsealarms, correctnegatives


def TS(obs, pre, threshold=0.1):
    if np.isnan(pre).all():
        return np.nan
    hits, misses, falsealarms, correctnegatives = prep_clf(obs=obs, pre=pre, threshold=threshold)
    return hits / (hits + falsealarms + misses + 1e-12)

