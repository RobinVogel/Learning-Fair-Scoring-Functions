import sys
import numpy as np
import matplotlib.pyplot as plt

from utils import (get_auc_from_roc, get_pointwise_from_roc,
                   get_roc_fixed_step, quantile)


def plot_all_rocs(path, n_runs=100):
    """Plots all ROCs on a single figure (looks messy)."""

    plt.figure(figsize=(3, 3))

    for i in range(0, n_runs):
        basedir = "{}/run_{}/final_analysis/rocs/sec4-test".format(path, i)
        with open(basedir + "-1-fpr.npy", "rb") as f:
            fpr_1 = np.load(f)
        with open(basedir + "-1-tpr.npy", "rb") as f:
            tpr_1 = np.load(f)
        with open(basedir + "-2-fpr.npy", "rb") as f:
            fpr_2 = np.load(f)
        with open(basedir + "-2-tpr.npy", "rb") as f:
            tpr_2 = np.load(f)
        with open(basedir + "-main-fpr.npy", "rb") as f:
            fpr = np.load(f)
        with open(basedir + "-main-tpr.npy", "rb") as f:
            tpr = np.load(f)

        s_1 = "$ROC_{H_s^{(0)}, H_s^{(1)}}$"
        s_2 = "$ROC_{G_s^{(0)}, G_s^{(1)}}$"
        plt.plot(fpr_1, tpr_1, label=s_1, color="green", alpha=0.50)
        plt.plot(fpr_2, tpr_2, label=s_2, color="blue", alpha=0.50)
        plt.plot(fpr, tpr, label="$ROC_{H_s, G_s}$", color="black", alpha=0.50)

    plt.xlabel("$FPR$")
    plt.ylabel("$TPR$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + "/roc_curves.pdf")


def get_AUC_and_ptw_ROC(path, n_runs=100):
    """Gives the important statistics for the toy experiments."""
    AUC_HG = list()
    AUC_H0G0 = list()
    AUC_H1G1 = list()
    ROC_H_ptw = list()

    for i in range(0, n_runs):
        basedir = "{}/run_{}/final_analysis/rocs/sec3-test".format(path, i)
        with open(basedir + "-1-fpr.npy", "rb") as f:
            fpr_1 = np.load(f)[::-1]
        with open(basedir + "-1-tpr.npy", "rb") as f:
            tpr_1 = np.load(f)[::-1]
        with open(basedir + "-2-fpr.npy", "rb") as f:
            fpr_2 = np.load(f)[::-1]
        with open(basedir + "-2-tpr.npy", "rb") as f:
            tpr_2 = np.load(f)[::-1]
        with open(basedir + "-main-fpr.npy", "rb") as f:
            fpr = np.load(f)[::-1]
        with open(basedir + "-main-tpr.npy", "rb") as f:
            tpr = np.load(f)[::-1]

        AUC_HG.append(get_auc_from_roc(fpr, tpr))
        AUC_H0G0.append(get_auc_from_roc(fpr_1, tpr_1))
        AUC_H1G1.append(get_auc_from_roc(fpr_2, tpr_2))

        basedir = "{}/run_{}/final_analysis/rocs/sec4-test".format(path, i)
        with open(basedir + "-1-fpr.npy", "rb") as f:
            fpr_1 = np.load(f)[::-1]
        with open(basedir + "-1-tpr.npy", "rb") as f:
            tpr_1 = np.load(f)[::-1]

        ROC_H_ptw.append(get_pointwise_from_roc(fpr_1, tpr_1, 0.75))

    with open(path + "/statistics.txt", "wt") as f:
        f.write("AUC_H,G = {} (+- {})\n".format(np.mean(AUC_HG),
                                                np.std(AUC_HG)))
        f.write("AUC_H_0,G_0 = {} (+- {})\n".format(np.mean(AUC_H0G0),
                                                    np.std(AUC_H0G0)))
        f.write("AUC_H_1,G_1 = {} (+- {})\n".format(np.mean(AUC_H1G1),
                                                    np.std(AUC_H1G1)))
        f.write("ROC_H0,G0(0.75) = {} (+- {})\n".format(np.mean(ROC_H_ptw),
                                                        np.std(ROC_H_ptw)))


def plot_all_rocs_quant(path, labels, ident,
                        n_pts=101, alpha=0.05, n_runs=100):
    plt.figure(figsize=(3, 3))
    if ident == "sec4":
        plt.plot([0, 1], [0, 1], color="grey", alpha=0.5)

    x_vals = np.linspace(0, 1, n_pts)
    c_1 = list()
    c_2 = list()
    c = list()
    for i in range(0, n_runs):
        print("run {}".format(i))
        basedir = "{}/run_{}/final_analysis/rocs/{}-test".format(path, i, ident)
        with open(basedir + "-1-fpr.npy", "rb") as f:
            fpr_1 = np.load(f)
        with open(basedir + "-1-tpr.npy", "rb") as f:
            tpr_1 = np.load(f)

        c_1.append(get_roc_fixed_step(fpr_1[::-1], tpr_1[::-1],  x_vals))

        with open(basedir + "-2-fpr.npy", "rb") as f:
            fpr_2 = np.load(f)
        with open(basedir + "-2-tpr.npy", "rb") as f:
            tpr_2 = np.load(f)

        c_2.append(get_roc_fixed_step(fpr_2[::-1], tpr_2[::-1], x_vals))

        with open(basedir + "-main-fpr.npy", "rb") as f:
            fpr = np.load(f)
        with open(basedir + "-main-tpr.npy", "rb") as f:
            tpr = np.load(f)

        c.append(get_roc_fixed_step(fpr[::-1], tpr[::-1], x_vals))

    s_1, s_2, s, sa_1, sa_2, sa = labels

    area1 = plt.fill_between(x_vals, quantile(c_1, (1-alpha/2), axis=0),
                             quantile(c_1, alpha/2, axis=0), color="green",
                             label=sa_1, alpha=0.25)
    area2 = plt.fill_between(x_vals, quantile(c_2, (1-alpha/2), axis=0),
                             quantile(c_2, alpha/2, axis=0), color="blue",
                             label=sa_2, alpha=0.25)
    area = plt.fill_between(x_vals, quantile(c, (1-alpha/2), axis=0),
                            quantile(c, alpha/2, axis=0), color="black",
                            label=sa, alpha=0.25)

    cur1 = plt.plot(x_vals, np.median(c_1, axis=0),
                    label=s_1, color="green", alpha=0.50)
    cur2 = plt.plot(x_vals, np.median(c_2, axis=0),
                    label=s_2, color="blue", alpha=0.50)
    cur = plt.plot(x_vals, np.median(c, axis=0),
                   label=s, color="black", alpha=0.50)

    # plt.legend()
    plt.xlabel("$FPR$")
    plt.ylabel("$TPR$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + "/roc_curves_quant_" + ident + ".pdf")

    plt.figure(figsize=(4, 1))
    area1 = plt.fill_between([0], [0], [0],
                             color="green", alpha=0.25, label=sa_1)
    area2 = plt.fill_between([0], [0], [0],
                             color="blue", alpha=0.25, label=sa_2)
    area = plt.fill_between([0], [0], [0], color="black", alpha=0.25, label=sa)

    cur1 = plt.Line2D([0], [0], label=s_1, color="green", alpha=0.50)
    cur2 = plt.Line2D([0], [0], label=s_2, color="blue", alpha=0.50)
    cur = plt.Line2D([0], [0], label=s, color="black", alpha=0.50)
    handles = (area1, area2, area, cur1, cur2, cur)
    labels = (sa_1, sa_2, sa, s_1, s_2, s)
    plt.gca().axis('off')
    plt.legend(handles, labels, loc='center', ncol=2)
    plt.savefig(path + "/legend-" + ident + ".pdf")


def main():
    path = sys.argv[1]
    # plot_all_rocs(path)
    labels_sec4 = ("med $ROC_{H_s^{(0)}, H_s^{(1)}}$",
                   "med $ROC_{G_s^{(0)}, G_s^{(1)}}$",
                   "med $ROC_{H_s, G_s}$",
                   "95% CI $ROC_{H_s^{(0)}, H_s^{(1)}}$",
                   "95% CI $ROC_{G_s^{(0)}, G_s^{(1)}}$",
                   "95% CI $ROC_{H_s, G_s}$")

    labels_sec3 = ("med $ROC_{H_s^{(0)}, G_s^{(0)}}$",
                   "med $ROC_{H_s^{(1)}, G_s^{(1)}}$",
                   "med $ROC_{H_s, G_s}$",
                   "95% CI $ROC_{H_s^{(0)}, G_s^{(0)}}$",
                   "95% CI $ROC_{H_s^{(1)}, G_s^{(1)}}$",
                   "95% CI $ROC_{H_s, G_s}$")
    plot_all_rocs_quant(path, labels_sec4, "sec4")
    plot_all_rocs_quant(path, labels_sec3, "sec3")
    get_AUC_and_ptw_ROC(path)


if __name__ == "__main__":
    main()
