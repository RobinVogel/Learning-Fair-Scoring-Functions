import os
import json
import numpy as np

DB_TO_TXTFILES = {
    "german": ["auc_bnsp.txt", "ptw_summaries.txt"],
    "adult": ["auc_subgroup_auc.txt", "ptw_summaries.txt"],
    "compas": ["auc_bpsn.txt", "ptw_summaries.txt"],
    "bank": ["auc_subgroup_auc.txt", "ptw_summaries.txt"]
}


def crit_auc_cons(d, lamb):
    crit_train = d["AUC_tr"] - lamb*np.abs(d["AUC_0_tr"] - d["AUC_1_tr"])
    crit_val = d["AUC_vl"] - lamb*np.abs(d["AUC_0_vl"] - d["AUC_1_vl"])
    crit_test = d["AUC_te"] - lamb*np.abs(d["AUC_0_te"] - d["AUC_1_te"])
    return crit_train, crit_val, crit_test


def crit_ptw_cons(d, lamb):
    train_cons_1 = np.abs(d["ROC(Y=-1/alpha=0.125)_tr"] - 0.125)
    train_cons_2 = np.abs(d["ROC(Y=-1/alpha=0.25)_tr"] - 0.25)

    valid_cons_1 = np.abs(d["ROC(Y=-1/alpha=0.125)_vl"] - 0.125)
    valid_cons_2 = np.abs(d["ROC(Y=-1/alpha=0.25)_vl"] - 0.25)

    test_cons_1 = np.abs(d["ROC(Y=-1/alpha=0.125)_te"] - 0.125)
    test_cons_2 = np.abs(d["ROC(Y=-1/alpha=0.25)_te"] - 0.25)

    crit_train = d["AUC_tr"] - lamb*(train_cons_1 + train_cons_2)
    crit_valid = d["AUC_vl"] - lamb*(valid_cons_1 + valid_cons_2)
    crit_test = d["AUC_te"] - lamb*(test_cons_1 + test_cons_2)
    return crit_train, crit_valid, crit_test


def write_table(f, lambdas, regs, datas):
    f.write("\\begin{tabular}{cccccccc} \n")
    f.write("\\toprule \n")
    for type_data in ["te"]:  # ["Test", "Valid", "Train"]:
        # "on {} data".format(type_data)
        header = (r"$\lambda / \lambda_H$ & $\lambda_\text{reg}$ & "
                  r"$\auc$ & $\Delta \auc$ "
                  "\n"
                  r"& $|\Delta_H(\frac{1}{8})|$ & $|\Delta_H(\frac{3}{4})|$ "
                  r"& $L_\lambda$ & $L_\Lambda$")
        f.write(header + " \\\\ \n")  # type_data + " & " +
        f.write("\\midrule \n")
        for lamb, reg, data in zip(lambdas, regs, datas):
            AUC = data["AUC_{}".format(type_data)]

            AUC_0 = data["AUC_0_{}".format(type_data)]
            AUC_1 = data["AUC_1_{}".format(type_data)]
            deltaAUC = np.abs(AUC_0 - AUC_1)

            ptw_cons_1 = np.abs(
                data["ROC(Y=-1/alpha=0.125)_{}".format(type_data)] - 0.125)
            ptw_cons_2 = np.abs(
                data["ROC(Y=-1/alpha=0.25)_{}".format(type_data)] - 0.25)

            L_lamb = AUC - lamb*deltaAUC
            L_Lamb = AUC - lamb*(ptw_cons_1 + ptw_cons_2)

            l1 = ("$" + "{:.2f}".format(lamb) + "$ & $"
                  + "{:.3f}".format(reg) + "$ & $")
            l2 = ("{:.2f}".format(AUC) + "$ & $"
                  + "{:.2f}".format(deltaAUC) + "$ & $")
            l3 = ("{:.2f}".format(ptw_cons_1) + "$ & $"
                  + "{:.2f}".format(ptw_cons_2) + r"$ & $")
            l4 = ("{:.2f}".format(L_lamb) + "$ & $"
                  + "{:.2f}".format(L_Lamb) + r"$ ")

            f.write(l1 + l2 + l3 + l4 + "\\\\ \n")  # type_data + " & " +
        f.write("\\bottomrule \n")
    f.write("\\end{tabular} \n")


def gen_table(db_name="adult", model="auc_cons", on_split="Val"):
    datas = list()
    lambdas = list()
    regs = list()
    for lamb_no in range(0, 6):
        max_criterion = -np.Infinity
        max_reg_no = None
        lamb = json.load(open("params/lambda/lambda_{}.json".format(
            lamb_no), "rt"))["coef_lagrange"]
        for reg_no in range(0, 7):
            # We recover the data from the textfiles:
            d = dict()
            for txtfile in DB_TO_TXTFILES[db_name]:
                path_txtfile = ("results/{}/{}/reg_{}/lambda_{}/"
                                "final_analysis/{}").format(
                                    db_name, model, reg_no, lamb_no, txtfile)
                with open(path_txtfile, "rt") as f:
                    data = [a.strip().split(" = ") for a in f.readlines()]
                    data = [a for a in data if len(a) > 1]
                    for k, v in data:
                        d[k] = float(v)

            # We compute the criterion on the data:
            if model.startswith("auc_cons"):
                current_crit = crit_auc_cons(d, lamb)
            else:
                current_crit = crit_ptw_cons(d, lamb)

            if on_split == "Val":
                current_crit = current_crit[1]
            elif on_split == "Train":
                current_crit = current_crit[0]
            elif on_split == "Test":
                current_crit = current_crit[2]
            else:
                msg = "Invalid split <{}> selected !".format(on_split)
                raise ValueError(msg)

            # We memorize the parameters if we get a better value:
            if current_crit > max_criterion:
                max_criterion = current_crit
                max_reg_no = reg_no
                max_data = d
        datas.append(max_data)
        lambdas.append(lamb)
        max_reg = json.load(open("params/reg/reg_{}.json".format(
            max_reg_no), "rt"))["reg"]
        regs.append(max_reg)

    outpath = "figures/supp/real_data/{}/{}".format(db_name, model)
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    with open("{}/table.tex".format(outpath), "wt") as f:
        write_table(f, lambdas, regs, datas)


if __name__ == "__main__":
    gen_table("german", "auc_cons_bnsp")
    gen_table("adult", "auc_cons")
    gen_table("compas", "auc_cons_bpsn")
    gen_table("bank", "auc_cons")
    gen_table("german", "ptw_cons")
    gen_table("adult", "ptw_cons")
    gen_table("compas", "ptw_cons")
    gen_table("bank", "ptw_cons")
