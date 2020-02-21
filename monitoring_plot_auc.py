import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_cost(df, iters):
    plt.figure(figsize=(4, 2))
    n_min = min(df.shape[0], len(iters))
    plt.plot(iters[:n_min], df["cost"][:n_min], color="green", label="cost")
    plt.plot(iters[:n_min], df["r_cost"][:n_min],
             color="blue", label="relaxed cost")
    plt.xlabel("iterations")
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_auc(df, iters):
    plt.figure(figsize=(4, 2))
    n_min = min(df.shape[0], len(iters))
    plt.plot(iters[:n_min], 1-df["auc"][:n_min], color="green", label="AUC")
    plt.plot(iters[:n_min], 1-df["r_auc"][:n_min],
             color="blue", label="relaxed AUC")
    plt.xlabel("iterations")
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_f_auc(df, iters):
    plt.figure(figsize=(4, 2))
    n_min = min(df.shape[0], len(iters))
    plt.plot(iters[:n_min], df["f_auc"][:n_min], color="green",
             label="fair $\Delta$AUC")
    plt.plot(iters[:n_min], df["r_f_auc"][:n_min],
             color="blue", label="relaxed fair $\Delta$AUC")
    plt.xlabel("iterations")
    plt.legend()
    plt.grid()
    plt.tight_layout()


def plot_c(df, iters):
    plt.figure(figsize=(4, 2))
    n_min = min(df.shape[0], len(iters))
    plt.plot(iters[:n_min], df["c"][:n_min], color="blue")
    plt.ylabel("c")
    plt.xlabel("iterations")
    plt.ylim([-1.1, 1.1])
    plt.grid()
    plt.tight_layout()


def plot_l2(df, iters):
    plt.figure(figsize=(4, 2))
    n_min = min(df.shape[0], len(iters))
    plt.plot(iters[:n_min], df["l2"][:n_min], color="blue")
    plt.ylabel(r"$\|\|W\|\|^2_2$")
    plt.xlabel("iterations")
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

    plot_f_auc(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/f_auc.pdf".format(outfolder))
    plt.close()

    plot_c(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/c.pdf".format(outfolder))
    plt.close()

    plot_l2(df, iters)
    plt.savefig("{}/dyn_analysis/plots_fancy/l2.pdf".format(outfolder))
    plt.close()


if __name__ == "__main__":
    main()
