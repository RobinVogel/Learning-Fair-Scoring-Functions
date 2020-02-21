import pathlib
import numpy as np
import matplotlib.pyplot as plt

from more_utils.roc_calculus import fpr_tpr

LEGEND_SIZE = (10.5, 1)


class RocGen:
    def __init__(self, s_1, s_2, out_fold):
        self.s_1 = s_1
        self.s_2 = s_2
        self.out_fold = out_fold
        self.prefix = ""

    def save_roc(self, fpr, tpr):
        fold = pathlib.Path("{}/{}".format(self.out_fold, "rocs"))
        if not fold.exists():
            fold.mkdir()
        path = fold/("{}".format(self.prefix))
        with open(str(path) + "fpr.npy", "wb") as f:
            np.save(f, fpr)
        with open(str(path) + "tpr.npy", "wb") as f:
            np.save(f, tpr)


class RocGenSec3(RocGen):
    def f_1(self, y, z):
        return z == 0

    def f_2(self, y, z):
        return z == 1

    def y_1(self, y, z):
        return y[self.f_1(y, z)]

    def y_2(self, y, z):
        return y[self.f_2(y, z)]


class RocGenSec4(RocGen):
    def f_1(self, y, z):
        return y != 1

    def f_2(self, y, z):
        return y == 1

    def y_1(self, y, z):
        return z[self.f_1(y, z)]

    def y_2(self, y, z):
        return z[self.f_2(y, z)]


def plot_roc(s, y, z, rocgen, ls="-"):
    fpr, tpr = fpr_tpr(s, y)

    fpr_1, tpr_1 = fpr_tpr(s[rocgen.f_1(y, z)], rocgen.y_1(y, z))
    fpr_2, tpr_2 = fpr_tpr(s[rocgen.f_2(y, z)], rocgen.y_2(y, z))
    fpr, tpr = fpr_tpr(s, y)

    plt.plot(fpr_1, tpr_1, label=rocgen.s_1, color="green", linestyle=ls)
    plt.plot(fpr_2, tpr_2, label=rocgen.s_2, color="blue", linestyle=ls)
    plt.plot(fpr, tpr, label="$ROC_{H_s, G_s}$", color="black", linestyle=ls)

    orig_prefix = rocgen.prefix
    rocgen.prefix = orig_prefix + "-1-"
    rocgen.save_roc(fpr_1, tpr_1)
    rocgen.prefix = orig_prefix + "-2-"
    rocgen.save_roc(fpr_2, tpr_2)
    rocgen.prefix = orig_prefix + "-main-"
    rocgen.save_roc(fpr, tpr)


def legend_get_handles(ls="-"):
    handles = list()
    for color in ["green", "blue", "black"]:
        handles.append(plt.Line2D([0], [0], color=color, linestyle=ls))
    return handles


def plot_roc_sec3(path_analysis, data_train, data_test, save_data=True):
    s_train, y_train, Z_train = data_train
    s_test, y_test, Z_test = data_test

    plt.figure(figsize=(3, 3))

    s_1 = "$ROC_{H_s^{(0)}, G_s^{(0)}}$"
    s_2 = "$ROC_{H_s^{(1)}, G_s^{(1)}}$"
    rocgen = RocGenSec3(s_1, s_2, path_analysis)
    rocgen.prefix = "sec3-test"
    plot_roc(s_test, y_test, Z_test, rocgen)
    rocgen.prefix = "sec3-train"
    plot_roc(s_train, y_train, Z_train, rocgen, ls="--")

    plt.xlabel("$FPR$")
    plt.ylabel("$TPR$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_analysis/"roc_sec3.pdf")

    plt.figure(figsize=LEGEND_SIZE)
    labels = [s_1, s_2, "$ROC_{H_s, G_s}$"]
    leg = plt.legend(legend_get_handles("--"), labels, title="Train",
                     ncol=3, loc="center left")
    plt.legend(legend_get_handles("-"), labels, title="Test",
               ncol=3, loc="center right")
    plt.gca().axis('off')
    plt.gca().add_artist(leg)
    plt.savefig(path_analysis/"leg_sec3.pdf")


def plot_roc_sec4(path_analysis, data_train, data_test, save_data=True):
    s_train, y_train, Z_train = data_train
    s_test, y_test, Z_test = data_test

    plt.figure(figsize=(3, 3))

    plt.plot([0, 1], [0, 1], color="grey")
    s_1, s_2 = "$ROC_{H_s^{(0)}, H_s^{(1)}}$", "$ROC_{G_s^{(0)}, G_s^{(1)}}$"
    rocgen = RocGenSec4(s_1, s_2, path_analysis)
    rocgen.prefix = "sec4-test"
    plot_roc(s_test, y_test, Z_test, rocgen)
    rocgen.prefix = "sec4-train"
    plot_roc(s_train, y_train, Z_train, rocgen, ls="--")

    plt.xlabel("$FPR$")
    plt.ylabel("$TPR$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path_analysis/"roc_sec4.pdf")

    plt.figure(figsize=LEGEND_SIZE)
    labels = [s_1, s_2, "$ROC_{H_s, G_s}$"]
    leg = plt.legend(legend_get_handles("--"), labels, title="Train",
                     ncol=3, loc="center left")
    plt.legend(legend_get_handles("-"), labels, title="Test",
               ncol=3, loc="center right")
    plt.gca().axis('off')
    plt.gca().add_artist(leg)
    plt.savefig(path_analysis/"leg_sec4.pdf")
