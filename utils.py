"""Utilities for the project."""
import os
import math
import psutil
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from more_utils.roc_calculus import (fpr_tpr, auc, pointwise_tpr, get_auc_from_roc,
                                     get_pointwise_from_roc, get_roc_fixed_step,
                                     quantile)
from more_utils.roc_plot import plot_roc_sec3, plot_roc_sec4


# -------------------- Model loading tools --------------------

if int(tf.__version__.split(".")[0]) == 2:
    tf.compat.v1.disable_eager_execution()
    from models.tfv2_model_auc_cons import AUCCons
    from models.tfv2_model_ptw_cons import PtwCons
    from models.tfv2_model_other_auc_cons import AUCConsBNSP, AUCConsBPSN
else:
    from models.model_auc_cons import AUCCons
    from models.model_ptw_cons import PtwCons
    from models.model_other_auc_cons import AUCConsBNSP, AUCConsBPSN

MODEL_PATH_TO_CLASS = {"auc_cons": AUCCons,
                       "auc_cons_bnsp": AUCConsBNSP,
                       "auc_cons_bpsn": AUCConsBPSN,
                       "ptw_cons": PtwCons}

PROCESS = psutil.Process(os.getpid())

MODELS_CODE = pathlib.Path("models/")
MODELS_RESULTS = pathlib.Path("results/")


def load_model(model_name, param_files=None):
    """Returns an instantiation of the model."""
    return MODEL_PATH_TO_CLASS[model_name](param_files=param_files)


def load_trained_model(model_name, weights_folder, param_files):
    model = MODEL_PATH_TO_CLASS[model_name](param_files=param_files)
    model.load_trained_model(weights_folder)
    return model


def save_array_in_txt(arr, path):
    with path.with_suffix(".txt").open("wt") as f:
        for a in arr:
            f.write(str(a) + "\n")


def load_txt_in_array(path):
    with path.with_suffix(".txt").open("rt") as f:
        return np.array(f.read().split())


def mem_usage():
    return PROCESS.memory_info().rss//10**6

# -------------------- End model loading tools --------------------

# ---------- Saving tools ----------

def save_aucs_to_file(data_train, data_test, filt, f_out,
                      B=500000, val_size=0.40):
    s_train, y_train, Z_train = data_train
    filt0, filt1 = [filt(y_train, Z_train, z_cur) for z_cur in [0, 1]]
    auc_train = auc(s_train, y_train, B=B)
    auc_0_train = auc(s_train[filt0], y_train[filt0], B=B)
    auc_1_train = auc(s_train[filt1], y_train[filt1], B=B)

    _, s_val, _, y_val, _, Z_val = train_test_split(
        s_train, y_train, Z_train, test_size=val_size,
        random_state=42)
    filt0, filt1 = [filt(y_val, Z_val, z_cur) for z_cur in [0, 1]]
    auc_val = auc(s_val, y_val, B=B)
    auc_0_val = auc(s_val[filt0], y_val[filt0], B=B)
    auc_1_val = auc(s_val[filt1], y_val[filt1], B=B)

    s_test, y_test, Z_test = data_test
    filt0, filt1 = [filt(y_test, Z_test, z_cur) for z_cur in [0, 1]]
    auc_test = auc(s_test, y_test, B=B)
    auc_0_test = auc(s_test[filt0], y_test[filt0], B=B)
    auc_1_test = auc(s_test[filt1], y_test[filt1], B=B)

    names = ["AUC_tr", "AUC_0_tr", "AUC_1_tr",
             "AUC_vl", "AUC_0_vl", "AUC_1_vl",
             "AUC_te", "AUC_0_te", "AUC_1_te"]

    vals = [auc_train, auc_0_train, auc_1_train,
            auc_val, auc_0_val, auc_1_val,
            auc_test, auc_0_test, auc_1_test]

    with open(f_out, "wt") as f:
        f.write("B={}\n".format(B))
        for n, v in zip(names, vals):
            s = "{} = {}".format(n, v)
            f.write(s + "\n")


def save_ptw_to_file(data_train, data_test, monitored_pts, f_out,
                     val_size=0.40):
    s_train, y_train, Z_train = data_train
    s_test, y_test, Z_test = data_test
    _, s_val, _, y_val, _, Z_val = train_test_split(
        s_train, y_train, Z_train, test_size=val_size,
        random_state=42)

    def filt(yarr, y):
        return (2*yarr-1)*(2*y-1) > 0

    names, vals = list(), list()
    for y, alphas in monitored_pts:
        for alpha in alphas:
            # Compute the ptw values
            names.append("ROC(Y={}/alpha={})_tr".format(y, alpha))
            val = pointwise_tpr(
                s_train[filt(y_train, int(y))], Z_train[filt(y_train, int(y))],
                alpha)
            vals.append(val)
            names.append("ROC(Y={}/alpha={})_vl".format(y, alpha))
            val = pointwise_tpr(
                s_val[filt(y_val, int(y))], Z_val[filt(y_val, int(y))],
                alpha)
            vals.append(val)
            names.append("ROC(Y={}/alpha={})_te".format(y, alpha))
            val = pointwise_tpr(
                s_test[filt(y_test, int(y))], Z_test[filt(y_test, int(y))],
                alpha)
            vals.append(val)

    with open(f_out, "wt") as f:
        for n, v in zip(names, vals):
            s = "{} = {}".format(n, v)
            f.write(s + "\n")

# ---------- End saving tools ----------

# ---------- Plotting tools ----------


def plot_2d_dist(X, Y, Z, n=500):
    X, Y, Z = X[:n], Y[:n], Z[:n]

    filt = np.logical_and(Y == +1, Z == 1)
    print("n_pos_1={}".format(np.sum(filt)))
    plt.scatter(X[filt, 0], X[filt, 1],
                color="green", marker="x", alpha=0.50, label="Y=+1, Z=1")
    filt = np.logical_and(Y == +1, Z == 0)
    print("n_pos_0={}".format(np.sum(filt)))
    plt.scatter(X[filt, 0], X[filt, 1],
                color="green", marker="o", alpha=0.50, label="Y=+1, Z=0")

    filt = np.logical_and(Y != +1, Z == 1)
    print("n_neg_1={}".format(np.sum(filt)))
    plt.scatter(X[filt, 0], X[filt, 1],
                color="red", marker="x", alpha=0.50, label="Y=-1, Z=1")
    filt = np.logical_and(Y != +1, Z == 0)
    print("n_neg_0={}".format(np.sum(filt)))
    plt.scatter(X[filt, 0], X[filt, 1],
                color="red", marker="o", alpha=0.50, label="Y=-1, Z=0")

# ---------- Other plotting tools ----------
