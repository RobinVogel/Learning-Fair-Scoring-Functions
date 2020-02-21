import datetime
import pathlib
import json
import argparse

import numpy as np
from utils import (load_trained_model, save_array_in_txt,
                   save_aucs_to_file, save_ptw_to_file,
                   plot_roc_sec3, plot_roc_sec4)
from load_data import load_db_by_name

DEFAULT_B = 500000


def gen_plots_return_scores(model_name, weights_folder, db_name, param_files,
                            B=DEFAULT_B, save_scores=True):
    print("Working on {} - {}".format(model_name, datetime.datetime.now()),
          flush=True)
    model = load_trained_model(model_name, pathlib.Path(weights_folder),
                               param_files)

    # Evaluation
    data_train, data_test = load_db_by_name(db_name)
    X_train, y_train, z_train = data_train
    X_test, y_test, z_test = data_test

    s_test = np.array(model.score(X_test)).ravel().astype(float)
    s_train = np.array(model.score(X_train)).ravel().astype(float)

    path_expes = weights_folder
    path_analysis = path_expes/"final_analysis"

    if not path_analysis.exists():
        path_analysis.mkdir()

    # 1. Plot ROC curves for all problems on the same plot
    plot_roc_sec3(path_analysis,
                  (s_train, y_train, z_train), (s_test, y_test, z_test))

    # 2. Other view on ROCs
    plot_roc_sec4(path_analysis,
                  (s_train, y_train, z_train), (s_test, y_test, z_test))

    # 3. Give all interesting AUC values
    def filt_subgroup_AUC(y, z, z_val):
        return z == z_val

    def filt_bnsp(y, z, z_val):
        return np.logical_or(y != +1, z == z_val)

    def filt_bpsn(y, z, z_val):
        return np.logical_or(y == +1, z == z_val)

    data_train = (s_train, y_train, z_train)
    data_test = (s_test, y_test, z_test)

    save_aucs_to_file(data_train, data_test, filt_subgroup_AUC,
                      path_analysis/"auc_subgroup_auc.txt",
                      val_size=model.validation_size)

    save_aucs_to_file(data_train, data_test, filt_bnsp,
                      path_analysis/"auc_bnsp.txt",
                      val_size=model.validation_size)

    save_aucs_to_file(data_train, data_test, filt_bpsn,
                      path_analysis/"auc_bpsn.txt",
                      val_size=model.validation_size)

    # 3. Special monitorings
    if db_name in {"toy1", "toy2"}:
        x1 = model.score(np.array([[1, 0]])).ravel().astype(float)[0]
        x2 = model.score(np.array([[0, 1]])).ravel().astype(float)[0]
        if db_name == "toy1":
            c_val = x1/(np.abs(x1) + np.abs(x2))
        else:
            c_val = -x1/(np.abs(x1) + np.abs(x2))

        with open(path_analysis/"c_val.txt", "wt") as f:
            json.dump({model.coef_lagrange: c_val}, f)

    if hasattr(model, "mon_pts"):  # The model is a pointwise model
        save_ptw_to_file(data_train, data_test,
                         model.mon_pts.items(),
                         path_analysis/"ptw_summaries.txt",
                         val_size=model.validation_size)

    if save_scores:
        print("Done evaluation !", flush=True)

        path_scores = path_expes / "scorefiles"

        if not path_scores.exists():
            path_scores.mkdir()

        save_array_in_txt(s_train, path_scores/"s_train")
        save_array_in_txt(s_test, path_scores/"s_test")

        save_array_in_txt(y_train, path_scores/"y_train")
        save_array_in_txt(y_test, path_scores/"y_test")

        save_array_in_txt(z_train, path_scores/"z_train")
        save_array_in_txt(z_test, path_scores/"z_test")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates the performances of the model.")
    parser.add_argument("-w", "--weights_folder", required=True,
                        help="Weight folder each model (where fit_* are).",
                        default="results/auc_cons/tmp")
    parser.add_argument("-m", "--model", required=True,
                        help="Models to run.", default="auc_cons")
    parser.add_argument("-b", "--db", default="sim-ex-tradeoff",
                        help="Database name.")
    parser.add_argument("-f", "--param_files", type=str, default=None,
                        nargs='+', help="Loads a file with many parameters.")
    args = parser.parse_args()

    gen_plots_return_scores(model_name=args.model,
                            weights_folder=pathlib.Path(args.weights_folder),
                            db_name=args.db,
                            param_files=args.param_files,
                            B=DEFAULT_B,
                            save_scores=False)


if __name__ == "__main__":
    main()
