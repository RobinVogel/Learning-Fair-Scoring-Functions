"""
    Fits the model.
"""
import datetime
import pickle
import pathlib
import argparse
from load_data import load_db_by_name
from utils import mem_usage, load_model
# import ipdb; ipdb.set_trace()

SEED = 42
RESULTS_PATH = pathlib.Path("results")
DEFAULT_DB = "german"


def fit(weights_folder, model_name="test_model", db_name=DEFAULT_DB,
        param_files=None, loaded_model_folder=None):

    print("Start fitting... - {}".format(datetime.datetime.now()), flush=True)
    print("Model: {}".format(model_name), flush=True)
    print("DB name: {}".format(db_name), flush=True)
    time_1 = datetime.datetime.now()

    print("Reading {} - mem {} Mb - {}".format(
        db_name, mem_usage(), datetime.datetime.now()), flush=True)

    data_train, data_test = load_db_by_name(db_name)
    X_train, Z_train, y_train = data_train
    X_test, Z_test, y_test = data_test

    print("Training... - mem {} Mb- {}".format(
        mem_usage(), datetime.datetime.now()), flush=True)

    model = load_model(model_name, param_files=param_files)

    exp_path = pathlib.Path(weights_folder)

    if not exp_path.exists():
        exp_path.mkdir()

    time_2 = datetime.datetime.now()
    model.fit(data_train, model_folder=loaded_model_folder)
    time_3 = datetime.datetime.now()

    print(exp_path)
    # Write a logging file with important info.
    with (exp_path / "log.txt").open("wt") as f:
        f.write("db_name: {}\n".format(db_name))
        f.write("time_to_load: {}\n".format(time_2-time_1))
        f.write("time_to_fit: {}\n".format(time_3-time_2))

    # Save the model.
    model.save_model(exp_path)


def main():
    parser = argparse.ArgumentParser(
        description="Fits the model.")
    parser.add_argument("-w", "--weights_folder", type=str, required=True,
                        help="Name of the experiment.")
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Models to run.")
    parser.add_argument("-b", "--db_name", type=str, default=DEFAULT_DB,
                        help="Database names.")
    parser.add_argument("-f", "--param_files", type=str, default=None,
                        nargs='+', help="Loads a file with many parameters.")
    parser.add_argument("-r", "--load_trained_model", type=str, default=None,
                        help="Loads a previously trained model.")
    args = parser.parse_args()

    fit(weights_folder=args.weights_folder,
        model_name=args.model,
        db_name=args.db_name,
        param_files=args.param_files,
        loaded_model_folder=args.load_trained_model)


if __name__ == "__main__":
    main()
