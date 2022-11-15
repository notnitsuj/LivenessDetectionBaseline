# Ref: https://stackoverflow.com/a/49555212

from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_eer(label, pred):
    fpr, tpr, _ = roc_curve(label, pred, pos_label=1)

    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    return eer