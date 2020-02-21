import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from monitoring_plot_auc import (plot_cost, plot_auc, plot_l2)

DEF_COLORS = ["blue", "green", "red", "black"]


def plot_fprs(df, iters):
    plt.figure(figsize=(3, 2))
    n_min = min(df.shape[0], len(iters))
    for i, col in enumerate(df.columns):
        plt.plot(iters[:n_min], df[col][:n_min],
                 label="#{}".format(i), color=DEF_COLORS[i])
    plt.xlabel("iterations")
    plt.ylabel("FPR")
    plt.legend(title="Constraint")
    plt.grid()
    plt.tight_layout()


def plot_tprs(df, iters):
    plt.figure(figsize=(3, 2))
    n_min = min(df.shape[0], len(iters))
    for i, col in enumerate(df.columns):
        plt.plot(iters[:n_min], df[col][:n_min],
                 label="#{}".format(i), color=DEF_COLORS[i])
    plt.xlabel("iterations")
    plt.ylabel("TPR")
    plt.legend(title="Constraint")
    plt.grid()
    plt.tight_layout()


def plot_cs(df, iters):
    plt.figure(figsize=(3, 2))
    n_min = min(df.shape[0], len(iters))
    for i, col in enumerate(df.columns):
        plt.plot(iters[:n_min], df[col][:n_min],
                 label="#{}".format(i), color=DEF_COLORS[i])
    plt.xlabel("iterations")
    plt.ylabel("c")
    plt.legend(title="Constraint")
    plt.ylim([-1.1, 1.1])
    plt.grid()
    plt.tight_layout()


def plot_thre(df, iters):
    plt.figure(figsize=(3, 2))
    n_min = min(df.shape[0], len(iters))
    for i, col in enumerate(df.columns):
        plt.plot(iters[:n_min], df[col][:n_min],
                 label="#{}".format(i), color=DEF_COLORS[i])
    plt.xlabel("iterations")
    plt.ylabel("Threshold t")
    plt.legend(title="Constraint")
    plt.grid()
    plt.tight_layout()


def main():
    outfolder = sys.argv[1]
    with open("{}/dyn_analysis/files/iter.txt".format(outfolder), "rt") as f:
        iters = [int(x) for x in f.read().split("\n") if len(x) > 0]
    df = pd.read_csv("{}/dyn_analysis/files/data.csv".format(outfolder))

    plot_cost(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/cost.pdf".format(outfolder))
    plt.close()

    plot_auc(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/auc.pdf".format(outfolder))
    plt.close()

    plot_l2(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/l2.pdf".format(outfolder))
    plt.close()

    df = pd.read_csv("{}/dyn_analysis/files/{}.csv".format(
        outfolder, "fpr"), header=None)
    plot_fprs(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/fprs.pdf".format(outfolder))
    plt.close()

    df = pd.read_csv("{}/dyn_analysis/files/{}.csv".format(
        outfolder, "tpr"), header=None)
    plot_tprs(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/tprs.pdf".format(outfolder))
    plt.close()

    df = pd.read_csv("{}/dyn_analysis/files/{}.csv".format(
        outfolder, "c"), header=None)
    plot_cs(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/cs.pdf".format(outfolder))
    plt.close()

    df = pd.read_csv("{}/dyn_analysis/files/{}.csv".format(
        outfolder, "biases"), header=None)
    plot_thre(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/thres.pdf".format(outfolder))
    plt.close()


if __name__ == "__main__":
    main()
