"""
    This script generates:
    * figures/sec4/limits-AUC/dist.pdf
    * figures/sec4/limits-AUC/roc.pdf
"""
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from utils import get_auc_from_roc

# az(pz+1) + bz/2 = cst
# az/pz + bz = 1
# ==> az(1/(pz+1) - 1/2pz) = cst
# a0 <= p0 & q1 <= a1

# Eg 2
p0 = 2
p1 = 20
a0 = 2

a1 = a0*((1/(p0+1) - 1/(2*p0))/(1/(p1+1) - 1/(2*p1)))
b0 = 1 - a0/p0
b1 = 1 - a1/p1

DEF_A = 0.5


def h_a(x, a=DEF_A):
    assert 0 <= a <= 1
    eps = 1 if 2*x > 1 else -1
    return 1/2 + (1/2)*eps*np.power(np.abs(2*x-1), (1-a)/a)


def h_s(t):
    return 1


def invH_s(t):
    return t


def H_s(t):
    return t 


def g0_s_vec(t):
    return np.array([g0_s(v) for v in t])


def g1_s_vec(t):
    return np.array([g1_s(v) for v in t])


def g0_s(t):
    return a0*t**(p0-1) + b0


def g1_s(t):
    return a1*t**(p1-1) + b1


def G0_s(t):
    return np.array([integrate.quad(g0_s, 0, v)[0] for v in t])


def G1_s(t):
    return np.array([integrate.quad(g1_s, 0, v)[0] for v in t])


def fpr_tpr_plots():
    n_x = 101
    x_s = np.linspace(0, 1, n_x)

    plt.figure(figsize=(3, 3))

    plt.plot(x_s, H_s(x_s), label=r"$H_s(x)$", color="red")
    plt.plot(x_s, G0_s(x_s), label=r"$G_s^{(0)}(x)$", color="blue")
    plt.plot(x_s, G1_s(x_s), label=r"$G_s^{(1)}(x)$", color="green")

    plt.xlabel(r"$x$")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig("figures/sec4/limits-AUC/dist.pdf")


def roc_plots():
    n_x = 101
    x_s = np.linspace(0, 1, n_x)

    plt.figure(figsize=(3, 3))

    tpr0 = 1 - G0_s(invH_s(1-x_s))
    plt.plot(x_s, tpr0, label=r"$ROC_{H_s, G_s^{(0)}}(\alpha)$",
             color="blue")
    tpr1 = 1 - G1_s(invH_s(1-x_s))
    plt.plot(x_s, tpr1, label=r"$ROC_{H_s, G_s^{(1)}}(\alpha)$",
             color="green")

    plt.plot([0, 1], [0, 1], color="black")
    plt.xlabel(r"$\alpha$")
    plt.legend()  # ncol=2
    plt.grid()
    plt.tight_layout()

    plt.savefig("figures/sec4/limits-AUC/roc.pdf")

    print("AUC0: " + str(get_auc_from_roc(x_s, tpr0)))
    print("AUC1: " + str(get_auc_from_roc(x_s, tpr1)))

    print("sup: " + str(np.max(tpr0-tpr1)))


if __name__ == "__main__":
    roc_plots()
    fpr_tpr_plots()
