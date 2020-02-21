import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils import auc, pointwise_tpr
from load_data import load_toy1, load_toy2


def get_aucs(db):
    # B = 10000
    B = 100000
    aucs = list()
    aucs0 = list()
    aucs1 = list()
    assert db in {"toy1", "toy2"}
    if db == "toy1":
        data_tr, data_te = load_toy1(n=10000)
    else:
        data_tr, data_te = load_toy2(n=10000)

    X_tr, Y_tr, Z_tr = data_tr
    X_te, Y_te, Z_te = data_te
    X = np.vstack([X_tr, X_te])
    Y = np.concatenate([Y_tr, Y_te])
    Z = np.concatenate([Z_tr, Z_te])

    cs = np.linspace(0, 1, 101)
    for c in cs:
        if db == "toy1":
            S = c*X[:, 0] + (1-c)*X[:, 1]
        else:
            S = - c*X[:, 0] + (1-c)*X[:, 1]
        aucs.append(auc(S, Y, B=B))
        aucs0.append(auc(S[Z == 0], Y[Z == 0], B=B))
        aucs1.append(auc(S[Z == 1], Y[Z == 1], B=B))

    return np.array(aucs), np.array(aucs0), np.array(aucs1)


def get_ptws(alpha, y_val):
    B = 100000
    aucs = list()
    ptws = list()
    data_tr, data_te = load_toy2(n=10000)

    X_tr, Y_tr, Z_tr = data_tr
    X_te, Y_te, Z_te = data_te
    X = np.vstack([X_tr, X_te])
    Y = np.concatenate([Y_tr, Y_te])
    Z = np.concatenate([Z_tr, Z_te])

    cs = np.linspace(0, 1, 101)
    for c in cs:
        if db == "toy1":
            S = c*X[:, 0] + (1-c)*X[:, 1]
        else:
            S = - c*X[:, 0] + (1-c)*X[:, 1]
        aucs.append(auc(S, Y, B=B))
        ptws.append(pointwise_tpr(S[Y == y_val], Z[Y == y_val], alpha))

    return np.array(aucs), np.array(ptws)


def original_figure(db="toy1", lagrangian=False, lamb=1):
    aucs, aucs0, aucs1 = get_aucs(db)

    cs = np.linspace(0, 1, 101)
    plt.figure(figsize=(5, 2))
    plt.plot(cs, aucs0, label="$AUC_{H_s^{(0)}, G_s^{(0)}}$", color="green")
    plt.plot(cs, aucs1, label="$AUC_{H_s^{(1)}, G_s^{(1)}}$", color="blue")
    plt.plot(cs, aucs,  label="$AUC_{H_s, G_s}$", color="black")
    if lagrangian:
        plt.plot(cs, aucs - lamb*np.abs(aucs0 - aucs1),
                 label=r"$L_\lambda$", color="red")
    plt.xlabel("$c$")
    plt.grid()
    plt.legend(ncol=2)
    plt.tight_layout()
    if lagrangian:
        plt.savefig("figures/supp/synth_data/{}/aucs_n_lag.pdf".format(db))
    else:
        plt.savefig("figures/supp/synth_data/{}/aucs.pdf".format(db))


def method_conv(db="toy1"):
    aucs, aucs0, aucs1 = get_aucs(db)

    lag_to_sol = dict()
    for lamb_no in [0, 3]:
        c_file = ("results/avg_{}"
                  "/auc_cons/lambda_{}/run_0/final_analysis/"
                  "c_val.txt").format(db, lamb_no)
        with open(c_file, "rb") as f:
            lamb_val, c_val = list(json.load(f).items())[0]
            lag_to_sol[float(lamb_val)] = c_val

    colors = ["blue", "green"]  # ["red", "green", "blue", "black"]
    plt.figure(figsize=(5, 2.5))
    lines = list()
    lines_label = list()
    cs = np.linspace(0, 1, 101)
    for lamb, color in zip(lag_to_sol.keys(), colors):
        plt.plot(cs, aucs - lamb*np.abs(aucs0 - aucs1),
                 color=color)
        plt.axvline(lag_to_sol[lamb], color=color, linestyle="--")
        lines.append(Line2D([0, 0], [0, 0], color=color))
        lines_label.append(r"$L_\lambda(s_c)$, $\lambda = {}$".format(lamb))
    lines.append(Line2D([0, 0], [0, 0], color="black", linestyle="--"))
    lines_label.append("GD solution")
    plt.xlabel("$c$")
    plt.grid()
    plt.gca().legend(lines, lines_label, ncol=2)
    plt.tight_layout()
    plt.savefig("figures/supp/synth_data/{}/sgd_solutions.pdf".format(db))


def method_conv_ptw_toy2(alpha=0.75):
    aucs, ptws = get_ptws(alpha, -1)

    lag_to_sol = dict()
    for lamb_no in [0, 3]:
        c_file = ("results/avg_{}"
                  "/ptw_cons/lambda_{}/run_0/final_analysis/"
                  "c_val.txt").format(db, lamb_no)
        with open(c_file, "rb") as f:
            lamb_val, c_val = list(json.load(f).items())[0]
            lag_to_sol[float(lamb_val)] = c_val

    colors = ["blue", "green"]  # ["red", "green", "blue", "black"]
    plt.figure(figsize=(5, 2.5))
    lines = list()
    lines_label = list()
    cs = np.linspace(0, 1, 101)
    for lamb, color in zip(lag_to_sol.keys(), colors):
        plt.plot(cs, aucs - lamb*np.abs(ptws - alpha),
                 color=color)
        plt.axvline(lag_to_sol[lamb], color=color, linestyle="--")
        lines.append(Line2D([0, 0], [0, 0], color=color))
        lines_label.append(r"$L_\Lambda(s_c)$, $\lambda_H = {}$".format(lamb))
    lines.append(Line2D([0, 0], [0, 0], color="black", linestyle="--"))
    lines_label.append("GD solution")
    plt.xlabel("$c$")
    plt.grid()
    plt.gca().legend(lines, lines_label, ncol=2)
    plt.tight_layout()
    plt.savefig("figures/supp/synth_data/{}/ptw_sgd_sol.pdf".format(db))


if __name__ == "__main__":
    db = sys.argv[1]
    original_figure(db)
    original_figure(db, lagrangian=True)
    method_conv(db)
    if db == "toy2":
        method_conv_ptw_toy2()
