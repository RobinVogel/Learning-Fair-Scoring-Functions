import datetime
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt

from utils import (load_trained_model, plot_2d_dist, MODELS_RESULTS)
from load_data import load_db_by_name

DEFAULT_B = 500000
LEGEND_SIZE = (2, 1)


def plot_01_square(fun, gran=100):
    """Plots a function values on the [0, 1]x[0, 1] square."""
    x = np.linspace(0, 1, gran+1).reshape((gran+1, 1))
    X = np.hstack([x]*(gran+1))
    all_points = np.array([X.transpose().ravel(), X.ravel()]).transpose()
    z = fun(all_points)
    Z = z.reshape(gran+1, gran+1)
    plt.imshow(Z, origin="lower", cmap='gray', extent=[0, 1, 0, 1])
    # , interpolation='bilinear')
    cbar = plt.colorbar()
    cbar.set_label("score")


def gen_analysis_and_save(model_name="auc_cons",
                          weights_folder=MODELS_RESULTS/"auc_cons"/"tmp",
                          db_name="toy2",
                          param_files=None, n=500):

    print("Starting eval...- {}".format(datetime.datetime.now()), flush=True)
    data_train, _ = load_db_by_name(db_name)
    X_train, Y_train, Z_train = data_train
    X_train, Y_train, Z_train = X_train[:n], Y_train[:n], Z_train[:n]

    plot_2d_dist(X_train, Y_train, Z_train, n=500)
    model = load_trained_model(model_name, pathlib.Path(weights_folder),
                               param_files)

    plot_01_square(lambda x: np.array(model.score(x)).ravel().astype(float))

    path_expes = weights_folder
    path_analysis = path_expes/"final_analysis"

    if not path_analysis.exists():
        path_analysis.mkdir()

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.savefig("{}/decision_plot.pdf".format(path_analysis))

    plt.figure(figsize=LEGEND_SIZE)
    labels = (["Y=+1, Z={}".format(i) for i in [0, 1]]
              + ["Y=-1, Z={}".format(i) for i in [0, 1]])
    handles = [plt.scatter([], [], color="green", marker="x", alpha=0.50),
               plt.scatter([], [], color="green", marker="o", alpha=0.50),
               plt.scatter([], [], color="red", marker="x", alpha=0.50),
               plt.scatter([], [], color="red", marker="o", alpha=0.50)]
    plt.legend(handles, labels, loc="center")
    plt.gca().axis('off')
    plt.savefig("{}/legend_decision_plot.pdf".format(path_analysis))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates the model on 2D grid.")
    parser.add_argument("-m", "--model", required=True,
                        help="Models to run.", default="auc_cons")
    parser.add_argument("-w", "--weights_folder", required=True,
                        help="Weight folder each model (where fit_* are).",
                        default="results/auc_cons/tmp")
    parser.add_argument("-b", "--db", default="toy2",
                        help="Database name.")
    parser.add_argument("-f", "--param_files", type=str, default=None,
                        nargs='+', help="Loads a file with many parameters.")
    args = parser.parse_args()

    gen_analysis_and_save(model_name=args.model,
                          weights_folder=pathlib.Path(args.weights_folder),
                          db_name=args.db, param_files=args.param_files)


if __name__ == "__main__":
    main()
