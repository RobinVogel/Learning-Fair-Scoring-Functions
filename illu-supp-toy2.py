import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from utils import plot_2d_dist
from load_data import load_toy2


def s_1(x):
    return - np.sin(np.pi/4)*x[:, 0] + np.cos(np.pi/4)*x[:, 1]


def s_2(x):
    return - np.sin(np.pi/8)*x[:, 0] + np.cos(np.pi/8)*x[:, 1]


def roc_plots_sec3(X, Y, Z, score, num):
    plt.figure(figsize=(3, 3))
    # plt.figure(figsize=(5, 5))
    # plt.figure(figsize=(10, 10))
    ls = 0
    for i, s_fun in zip([num], [score]):
        linestyle = ["solid", "dashed"][ls]
        S = s_fun(X)
        print("mean_1: " + str(np.mean(S[Z == 1])))
        print("mean_0: " + str(np.mean(S[Z == 0])))
        for z in [0, 1]:
            s = S[Z == z]
            fpr, tpr, _ = roc_curve(Y[Z == z], s)
            if z == 0:
                label = (r"$ROC_{H_{s_" + str(i) + r"}^{(0)}, G_{s_" + str(i) +
                         r"}^{(0)}}(\alpha)$")
                color = "red"
            else:
                label = (r"$ROC_{H_{s_" + str(i) + r"}^{(1)}, G_{s_" + str(i) +
                         r"}^{(1)}}(\alpha)$")
                color = "blue"
            plt.plot(fpr, tpr, label=label, color=color,
                     linestyle=linestyle)

        fpr, tpr, _ = roc_curve(Y, S)
        label = (r"$ROC_{H_{s_" + str(i) + r"}, G_{s_" + str(i) +
                 r"}}(\alpha)$")
        plt.plot(fpr, tpr, label=label, color="green", linestyle=linestyle)
        ls += 1
    plt.plot([0, 1], [0, 1], color="black")
    plt.xlabel(r"$\alpha$")
    plt.legend()  # ncol=2)  # ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/supp/synth_data/toy2/roc_sec3_s{}.pdf".format(num))


def roc_plots_sec4(X, Y, Z, score, num):
    plt.figure(figsize=(3, 3))
    # plt.figure(figsize=(10, 10))
    ls = 0
    for i, s_fun in zip([num], [score]):
        linestyle = ["solid", "dashed"][ls]
        S = s_fun(X)
        print("mean_1: " + str(np.mean(S[Z == 1])))
        print("mean_0: " + str(np.mean(S[Z == 0])))
        for y in [-1, +1]:
            s0 = S[np.logical_and(Y == y, Z == 0)]
            s1 = S[np.logical_and(Y == y, Z == 1)]
            s_tpr = np.concatenate([s0, s1])
            y_tpr = np.concatenate([[-1]*s0.shape[0], [+1]*s1.shape[0]])
            fpr, tpr, _ = roc_curve(y_tpr, s_tpr)
            if y == +1:
                label = (r"$ROC_{G_{s_" + str(i) + r"}^{(0)}, G_{s_" + str(i) +
                         r"}^{(1)}}(\alpha)$")
                color = "red"
            else:
                label = (r"$ROC_{H_{s_" + str(i) + r"}^{(0)}, H_{s_" + str(i) +
                         r"}^{(1)}}(\alpha)$")
                color = "blue"
            plt.plot(fpr, tpr, label=label, color=color,
                     linestyle=linestyle)

        fpr, tpr, _ = roc_curve(Y, S)
        label = (r"$ROC_{H_{s_" + str(i) + r"}, G_{s_" + str(i) +
                 r"}}(\alpha)$")
        plt.plot(fpr, tpr, label=label, color="green", linestyle=linestyle)
        ls += 1
    plt.plot([0, 1], [0, 1], color="black")
    plt.xlabel(r"$\alpha$")
    plt.legend()  # ncol=2)
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/supp/synth_data/toy2/roc_sec4_s{}.pdf".format(num))

def all_plots():
    (X_tr, Y_tr, Z_tr), (X_te, Y_te, Z_te) = load_toy2(n=10000)
    X, Y, Z = (np.concatenate([X_tr, X_te]),
               np.concatenate([Y_tr, Y_te]),
               np.concatenate([Z_tr, Z_te]))

    roc_plots_sec3(X, Y, Z, s_1, 1)
    roc_plots_sec3(X, Y, Z, s_2, 2)

    roc_plots_sec4(X, Y, Z, s_1, 1)
    roc_plots_sec4(X, Y, Z, s_2, 2)

    plt.figure(figsize=(5, 5))
    plot_2d_dist(X, Y, Z, n=500)
    plt.legend()
    plt.savefig("figures/supp/synth_data/toy2/dist.pdf")


if __name__ == "__main__":
    all_plots()
