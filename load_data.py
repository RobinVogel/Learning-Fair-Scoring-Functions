import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEF_TEST_SIZE = 0.20
BIG_TEST_SIZE = 0.40
TEST_SIZE_GERMAN = BIG_TEST_SIZE
TEST_SIZE_COMPAS = DEF_TEST_SIZE
TEST_SIZE_BANK = DEF_TEST_SIZE
TEST_SIZE_YOW = BIG_TEST_SIZE
TEST_SIZE_ARRHYTHMIA = BIG_TEST_SIZE

N_TRAIN_TOY = 10000


def load_german_data():
    def preprocess_z(x):
        assert x in {"A91", "A92", "A93", "A94", "A95"}
        if x in {"A91", "A93", "A94"}:
            return 1
        else:
            return 0

    def preprocess_y(x):
        assert x in {1, 2}
        return 2*int(x == 1) - 1

    # Generates a dataset with 48 covariates with
    df = pd.read_csv("data/german_credit_data/german.data", sep=" ",
                     header=None)
    df.columns = ["check account", "duration", "credit history", "purpose",
                  "credit amount", "savings/bonds", "employed since",
                  "installment rate", "status and sex",
                  "other debtor/guarantor", "residence since", "property",
                  "age", "other plans", "housing", "existing credits",
                  "job", "number liable people", "telephone",
                  "foreign worker", "credit decision"]
    ind_sex = 8
    Z = np.array([preprocess_z(x) for x in df[df.columns[ind_sex]]])
    Y = np.array([preprocess_y(x) for x in df[df.columns[-1]]])

    cols_X = df.columns[:-1]
    ind_quali = {0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 18, 19}
    ind_all = set(range(len(cols_X)))

    X_quanti = df[df.columns[list(ind_all.difference(ind_quali))]].values

    X_quali = df[df.columns[list(ind_quali)]].values

    quali_encoder = OneHotEncoder(categories="auto")
    quali_encoder.fit(X_quali)
    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    X_train, X_test, Z_train, Z_test, Y_train, Y_test = train_test_split(
        X, Z, Y, test_size=TEST_SIZE_GERMAN, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)



def load_adult_dataset():
    # The continuous variable fnlwgt represents final weight, which is the
    # number of units in the target population that the responding unit
    # represents.
    df_train = pd.read_csv("data/adult_dataset/adult.data", header=None)
    columns = ["age", "workclass", "fnlwgt", "education", "education-num",
               "marital-status", "occupation", "relationship", "race", "sex",
               "capital-gain", "capital-loss", "hours-per-week",
               "native-country", "salary"]
    df_train.columns = columns
    df_test = pd.read_csv("data/adult_dataset/adult.test", header=None, comment="|")
    df_test.columns = columns

    def proc_z(Z):
        return np.array([1 if "Male" in z else 0 for z in Z])

    def proc_y(Y):
        return np.array([1 if ">50K" in y else -1 for y in Y])

    Z_train, Z_test = [proc_z(s["sex"]) for s in [df_train, df_test]]
    Y_train, Y_test = [proc_y(s["salary"]) for s in [df_train, df_test]]

    col_quanti = ["age", "education-num", "capital-gain",
                  "capital-loss", "hours-per-week"]  # "fnlwgt",
    col_quali = ["workclass", "education", "marital-status", "occupation",
                 "relationship", "race", "sex", "native-country"]

    X_train_quali = df_train[col_quali].values
    X_test_quali = df_test[col_quali].values

    X_train_quanti = df_train[col_quanti]
    X_test_quanti = df_test[col_quanti]

    quali_encoder = OneHotEncoder(categories="auto")  # drop="first")
    quali_encoder.fit(X_train_quali)

    X_train_quali_enc = quali_encoder.transform(X_train_quali).toarray()
    X_test_quali_enc = quali_encoder.transform(X_test_quali).toarray()

    X_train = np.concatenate([X_train_quali_enc, X_train_quanti], axis=1)
    X_test = np.concatenate([X_test_quali_enc, X_test_quanti], axis=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)


def load_compas_data():
    # See https://www.propublica.org/article/how-we-analyzed-the-compas-recidivism-algorithm
    # Load the two-year data
    df = pd.read_csv('data/compas_data/compas-analysis-master/compas-scores.csv')

    # vr = violent recidivism
    # r = recidivism
    # Types of crimes in the USA: felonies and misdemeanors
    interesting_cols = [  # 'compas_screening_date',
        'sex',  # 'dob',
        'age', 'race',
        'juv_fel_count', 'decile_score', 'juv_misd_count',
        'juv_other_count', 'priors_count',
        'days_b_screening_arrest',
        'c_jail_in', 'c_jail_out',
        # 'c_offense_date', 'c_arrest_date',
        # 'c_days_from_compas',
        'c_charge_degree',
        # 'c_charge_desc',
        'is_recid',
        # 'r_charge_degree',
        # 'r_days_from_arrest', 'r_offense_date',  # 'r_charge_desc',
        # 'r_jail_in', 'r_jail_out',
        # 'is_violent_recid', 'num_vr_cases',  # 'vr_case_number',
        # 'vr_charge_degree', 'vr_offense_date',
        # 'vr_charge_desc', 'v_type_of_assessment',
        'v_decile_score',  # 'v_score_text',
        # 'v_screening_date',
        # 'type_of_assessment',
        'decile_score.1',  # 'score_text',
        # 'screening_date'
        ]
    df = df[interesting_cols]
    df = df[np.logical_and(df["days_b_screening_arrest"] >= -30,
                           df["days_b_screening_arrest"] <= 30)]
    df["days_in_jail"] = [a.days for a in (pd.to_datetime(df["c_jail_out"]) -
                                           pd.to_datetime(df["c_jail_in"]))]
    df = df[df["is_recid"] >= 0]
    df = df[df["c_charge_degree"] != "O"]
    # df = df[[x in {"Caucasian", "African-American"} for x in df["race"]]]
    Z = np.array([int(x == "African-American") for x in df["race"]])
    Y = 2*df["is_recid"].values - 1

    cols_to_delete = ["c_jail_out", "c_jail_in", "days_b_screening_arrest"]
    df = df[[a for a in df.columns if a not in cols_to_delete]]

    col_quanti = ["age", "juv_fel_count", "decile_score", "juv_misd_count",
                  "priors_count", "v_decile_score", "decile_score.1",
                  "days_in_jail"]
    col_quali = ["race", "c_charge_degree"]

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto")
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=TEST_SIZE_COMPAS, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)


def load_bank_data():
    # It is bank marketing data.
    # bank.csv 462K lines 450 Ko
    # bank-full 4M 614K lines 4.4 Mo
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing
    df = pd.read_csv("data/bank_marketing_data/bank-additional"
                     + "/bank-additional-full.csv", sep=";")

    Y = np.array([2*int(y == "yes") - 1 for y in df["y"]])
    Z = np.logical_and(df["age"].values <= 60,
                       df["age"].values >= 25).astype(int)

    col_quanti = ["age", "duration", "campaign", "pdays", "previous",
                  "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                  "euribor3m", "nr.employed"]
    col_quali = ["job", "education", "default", "housing", "loan", "contact",
                 "month", "day_of_week", "poutcome"]

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto")
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    X_train, X_test, Y_train, Y_test, Z_train, Z_test = train_test_split(
        X, Y, Z, test_size=TEST_SIZE_BANK, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (X_train, Y_train, Z_train), (X_test, Y_test, Z_test)


def load_toy1(n=4000):
    n_tr = n
    n_te = n*2
    n_tot = n_tr + n_te

    # q0 = 1/10
    q1 = 17 / 20

    X = np.random.uniform(0, 1, (n_tot, 2))
    Z = np.random.binomial(1, q1, n_tot)
    Y = np.zeros_like(Z)

    Y[Z == 0] = 2*np.random.binomial(1, X[Z == 0, 0]) - 1
    Y[Z == 1] = 2*np.random.binomial(1, X[Z == 1, 1]) - 1

    return (X[:n_tr], Y[:n_tr], Z[:n_tr]), (X[n_tr:], Y[n_tr:], Z[n_tr:])


def load_toy2(n=4000, q1=1/2):
    n_tr = n
    n_te = n*2
    n_tot = n_tr + n_te

    Z = np.random.binomial(1, q1, n_tot)
    thetas = np.random.uniform(0, np.pi/2, n_tot)
    rs = np.random.uniform(0, 0.5, n_tot) + 0.5*Z
    X = np.array([rs*np.cos(thetas), rs*np.sin(thetas)]).transpose()
    Y = 2*np.random.binomial(1, 2*thetas/np.pi) - 1

    return (X[:n_tr], Y[:n_tr], Z[:n_tr]), (X[n_tr:], Y[n_tr:], Z[n_tr:])


def load_db_by_name(db_name):
    if db_name == "german":
        return load_german_data()
    elif db_name == "adult":
        return load_adult_dataset()
    elif db_name == "compas":
        return load_compas_data()
    elif db_name == "bank":
        return load_bank_data()
    elif db_name == "toy1":
        return load_toy1(n=N_TRAIN_TOY)
    elif db_name == "toy2":
        return load_toy2(n=N_TRAIN_TOY)
    raise ValueError("Wrong db name...")


if __name__ == "__main__":
    for db_name in ["german", "adult", "compas", "bank"]:
        data_train, _ = load_db_by_name(db_name)
        print("{}:".format(db_name), data_train[0].shape[1])
    # data_german = load_german_data()
    # data_train, data_test = load_adult_dataset()
    # data_bank = load_bank_data()
    # data_compas = load_compas_data()
