"""
    Plots a first version of the dynamics.
"""
import sys
import matplotlib.pyplot as plt
import pandas as pd

DICT_YLABEL = {"r_cost": r"$\widetilde{L}_\lambda$",
               "cost": r"$\widehat{L}_\lambda$",
               "auc": r"$1 - AUC_{\widehat{H}_s, \widehat{G}_s}$",
               "r_auc": r"$1 - \widetilde{AUC}_{H, G}$",
               "f_auc": r"Diff $\widehat{AUC}_{H^{(z)}, G^{(z)}}$'s",
               "r_f_auc": r"Diff $\widetilde{AUC}_{H^{(z)}, G^{(z)}}$'s",
               "l2": "L2 norm of weights",
               "c": "$c$",

               "mean_c": "mean_c",
               "var_c": "var_c",
               "dFPR_m": "dFPR_m",
               "dFPR_v": "dFPR_v",
               "dFNR_m": "dFNR_m",
               "dFNR_v": "dFNR_v"
               }

DICT_YMIN_YMAX = {"r_cost": [0., 0.75],  # [-0.1, 1.2],
                  "cost": [0., 0.75],  # [0, 1],
                  "auc": [-0.05, 1.05],  # [0, 1],
                  "r_auc": [-0.05, 1.05],
                  "f_auc": [-0.5, 0.5],  # [-1, 1]
                  "r_f_auc": [-0.5, 0.5],  # [-1, 1],
                  "l2": [0, 5],  # [0, 10],
                  "c": [-1.05, 1.05],


                  "mean_c": None,
                  "var_c":  None,
                  "dFPR_m": None,
                  "dFPR_v": None,
                  "dFNR_m": None,
                  "dFNR_v": None
                  }


def plot_val(outfolder, val, dyn_ana, form="pdf", csv_name="data"):
    with open("{}/{}/files/iter.txt".format(outfolder, dyn_ana), "rt") as f:
        iters = [int(x) for x in f.read().split("\n") if len(x) > 0]
        n_iter = len(iters)

    plt.figure(figsize=(4, 2))
    if csv_name == "data":
        df = pd.read_csv("{}/{}/files/data.csv".format(
            outfolder, dyn_ana))
        n_min = min(df.shape[0], n_iter)
        plt.plot(iters[:n_min], df[val][:n_min])
        plt.ylabel(DICT_YLABEL[val])
    else:
        df = pd.read_csv("{}/{}/files/{}.csv".format(
            outfolder, dyn_ana, csv_name), header=None)
        n_min = min(df.shape[0], n_iter)
        for col in df.columns:
            plt.plot(iters[:n_min], df[col][:n_min], label=col, alpha=0.5)
        plt.ylabel(val)
        plt.legend()
    plt.xlabel("iterations")
    plt.grid()
    plt.tight_layout()
    # plt.ylim(DICT_YMIN_YMAX[val])
    if csv_name != "data":
        val = val + "-all"

    plt.savefig("{}/{}/plots/{}.{}".format(
        outfolder, dyn_ana, val, form))
    plt.close()


if __name__ == "__main__":
    outfolder = sys.argv[1]
    val = sys.argv[2]
    dyn_ana = sys.argv[3]
    if len(sys.argv) > 4:
        csv_name = sys.argv[4]
    else:
        csv_name = "data"
    plot_val(outfolder, val, dyn_ana, csv_name=csv_name)
