# The adaptative step is related to the evaluation step
# since the adaptation is calibrated on its results.
import os
import datetime
import pickle
import json
import numpy as np
import tensorflow as tf
from models.general_model import GeneralModel
from sklearn.model_selection import train_test_split

DEF_N_ITER = 50000
DEF_ADAPT_STEP = 0.01
DEF_LR = 0.001
DEF_MOMENTUM = 0.9
DEF_N_BATCH = 1000
DEF_DISPLAY_STEP = 500
DEF_C = 0.
DEF_PREC = 5
DEF_VAR_INIT = 0.01
DEF_REG = 0.001
DEF_OPTIMIZER = "momentum"
DEF_COOLING_C = 0
DEF_BALANCED = 0
DEF_VALIDATION_SIZE = 0.10
DEF_COEF_LAGRANGE = 1.
DEF_NET_DEPTH = 3
DEF_USE_INC_AUC_TRAIN = 1
DEF_N_INCOMPLETE_TRAIN = DEF_N_BATCH
DEF_USE_INC_AUC_VALID = 1
DEF_N_INCOMPLETE_VALID = DEF_N_BATCH*10


def gen_batch(X, y, z, dataset_name=None, n_batch=-1, balanced=False):
    """data_type in {"train", "test"}, n_batch = -1 means all the data"""
    n = len(y)
    if n_batch >= 0:
        if balanced:
            n_pos = n_batch//2
            n_neg = n_batch - n_pos
            filt_y = y == 1
            positives = np.where(filt_y)[0]
            negatives = np.where(~filt_y)[0]
            selected = np.concatenate([np.random.choice(positives, n_pos),
                                       np.random.choice(negatives, n_neg)])
        else:
            selected = np.random.randint(0, n, n_batch)
        return X[selected, :], y[selected], z[selected]
    return X, y, z


def full_auc(sc, y, rel):
    y = tf.reshape(y, (-1, 1))
    delta_y = y - tf.transpose(a=y)
    n_diff = tf.reduce_sum(input_tensor=tf.cast(tf.not_equal(delta_y, 0), tf.int32))

    sc_loc = sc

    def compute_value():
        sc = tf.reshape(sc_loc, (-1, 1))
        delta_sc = sc - tf.transpose(a=sc)
        filt = tf.greater(delta_y, 0.)
        # We only select relevant instances to multiply:
        prod = delta_sc[filt]*delta_y[filt]
        rel_auc = tf.reduce_mean(input_tensor=rel(prod))
        auc = (tf.reduce_mean(input_tensor=tf.cast(tf.greater(0., prod), tf.float32)) +
               0.5*tf.reduce_mean(input_tensor=tf.cast(tf.equal(prod, 0.), tf.float32)))
        return rel_auc, auc

    cond = tf.equal(n_diff, 0)
    return tf.cond(pred=cond, true_fn=lambda: (0, 0), false_fn=compute_value)


def incomplete_auc(sc, y, rel, B=1000):
    # y = tf.constant([-1, 1, -1, 1])
    # sc = tf.constant([0.1, 0.9, 0.2, 0.8])
    p_filt = tf.equal(y, 1)
    n_filt = tf.logical_not(p_filt)  # tf.equal(y, -1)

    p_sc, p_y = sc[p_filt], y[p_filt]
    n_sc, n_y = sc[n_filt], y[n_filt]

    num_n = tf.shape(input=n_y)[0]
    num_p = tf.shape(input=p_y)[0]

    def compute_value():
        n_ind = tf.random.uniform([B], minval=0, maxval=num_n, dtype=tf.int32)
        p_ind = tf.random.uniform([B], minval=0, maxval=num_p,
                                  dtype=tf.int32)

        n_sc_inc = tf.gather(n_sc, n_ind)
        p_sc_inc = tf.gather(p_sc, p_ind)

        prod = 2*(p_sc_inc - n_sc_inc)
        rel_auc = tf.reduce_mean(input_tensor=rel(prod))
        auc = (tf.reduce_mean(input_tensor=tf.cast(tf.greater(0., prod), tf.float32)) +
               0.5*tf.reduce_mean(input_tensor=tf.cast(tf.equal(prod, 0.), tf.float32)))
        return rel_auc, auc

    cond = tf.logical_or(tf.equal(num_n, 0), tf.equal(num_p, 0))
    return tf.cond(pred=cond, true_fn=lambda: (0., 0.), false_fn=compute_value)


class AUCCons(GeneralModel):

    def __init__(self, param_files=None):
        self.n_iter = DEF_N_ITER
        self.n_batch = DEF_N_BATCH
        self.display_step = DEF_DISPLAY_STEP
        self.lr = DEF_LR
        self.adapt_step = DEF_ADAPT_STEP
        self.momentum = DEF_MOMENTUM
        self.default_c = DEF_C
        self.default_var_init = DEF_VAR_INIT
        self.reg = DEF_REG
        self.optimizer = DEF_OPTIMIZER
        self.weight_fun = "softmax"
        self.cooling_c = DEF_COOLING_C
        self.balanced = DEF_BALANCED
        self.validation_size = DEF_VALIDATION_SIZE
        self.coef_lagrange = DEF_COEF_LAGRANGE
        self.net_depth = DEF_NET_DEPTH
        self.use_inc_auc_train = DEF_USE_INC_AUC_TRAIN
        self.n_incomplete_train = DEF_N_INCOMPLETE_TRAIN
        self.use_inc_auc_valid = DEF_USE_INC_AUC_VALID
        self.n_incomplete_valid = DEF_N_INCOMPLETE_VALID
        self.c = DEF_C

        if param_files is not None:
            for param_file in param_files:
                if os.path.exists(param_file):
                    with open(param_file, "rt") as f:
                        d_params = json.load(f)

                    for k, v in d_params.items():
                        setattr(self, k, v)
                else:
                    msg = "The file {} does not exist !".format(param_file)
                    raise ValueError(msg)

        print("Parameters:")
        for k, v in vars(self).items():
            print("{:s} : {:s}".format(str(k), str(v)))

        # - other:
        self.ph_c = None
        self.ph_incomplete = None
        self.l2_weights = None
        self.sc = None

        # Define the quantities to monitor
        self.rel_cost = None
        self.cost = None

        # Conveniences
        self.var_are_initialized = False
        self.saver = None
        self.sess = None
        self.cur_iter = 0

        self.weights = None
        self.biases = None

        assert self.use_inc_auc_train == self.use_inc_auc_valid

        super().__init__()

    def fit(self, data_train, model_folder=None):
        """
        Fits the model.
        """
        X_tot, y_tot, z_tot = data_train
        self.n_features = X_tot.shape[1]
        X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(
            X_tot, y_tot, z_tot, test_size=self.validation_size,
            random_state=42)

        print("Number of features: {}".format(self.n_features))
        print("Size of the train set: {}".format(len(y_train)))
        print("Size of the validation set: {}".format(len(y_val)))

        self.define_model()
        self.define_cost(self.l2_weights)

        if self.optimizer == "momentum":
            optim_step = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.lr,
                                                    momentum=self.momentum)
        elif self.optimizer == "adam":
            optim_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr,
                                                beta1=self.momentum)
        elif self.optimizer == "adagrad":
            optim_step = tf.compat.v1.train.AdamOptimizer(learning_rate=self.lr,
                                                beta1=self.momentum)
        else:
            raise ValueError("Unknown optimizer, choose adam or momentum.")

        optim_step = optim_step.minimize(self.rel_cost)
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        optim_step = tf.group([optim_step, update_ops])

        step = 0

        self.saver = tf.compat.v1.train.Saver()

        if self.sess is None:
            self.sess = tf.compat.v1.Session()

        if model_folder is not None:
            self.load_model_for_training(model_folder)

        if not self.var_are_initialized:
            self.sess.run(tf.compat.v1.global_variables_initializer())

        tf.compat.v1.get_default_graph().finalize()

        try:
            while step < self.n_iter:
                X_tr, y_tr, z_tr = gen_batch(X_train, y_train, z_train,
                                             n_batch=self.n_batch,
                                             balanced=self.balanced)
                train_dict = self.base_train_dict(X_tr, y_tr, z_tr)

                if self.use_inc_auc_train:
                    train_dict = dict(
                        list(train_dict.items()) + [(self.ph_incomplete,
                                                     self.n_incomplete_train)])

                self.sess.run(optim_step, feed_dict=train_dict)

                bool_monitor = ((step == self.n_iter - 1)
                                or (step % self.display_step == 0))
                if bool_monitor:
                    self.cur_iter = step
                    self.monitor_fun(X_tr, y_tr, z_tr, "train", header=True)
                    # Compute the loss for each dataset and add it.
                    val_fair = self.monitor_fun(X_val, y_val, z_val, "test")
                    self.adaptive_learning(val_fair)
                    self.print_summary_monitor(step)

                step += 1
        except KeyboardInterrupt:
            print("Learning process stopped manually.")

    def base_train_dict(self, X_tr, y_tr, z_tr):
        return {self.ph_x: X_tr, self.ph_y: y_tr, self.ph_z: z_tr,
                self.ph_c: self.c, self.is_train: True}

    def load_trained_model(self, model_folder):
        n_features_loc = str(model_folder) + "/n_features.pickle"
        with open(n_features_loc, "rb") as f:
            self.n_features = pickle.load(f)

        if self.sess is None:
            self.sess = tf.compat.v1.Session()

        self.define_model()
        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess, "{}/model.ckpt".format(model_folder))
        self.var_are_initialized = True

    def load_model_for_training(self, model_folder):
        if self.sess is None:
            self.sess = tf.compat.v1.Session()

        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess, "{}/model.ckpt".format(model_folder))

        cs_loc = str(model_folder) + "/c.pickle"
        with open(cs_loc, "rb") as f:
            self.c = pickle.load(f)

        self.var_are_initialized = True

    def save_model(self, model_folder):
        self.saver.save(self.sess, "{}/model.ckpt".format(model_folder))
        cs_loc = str(model_folder) + "/c.pickle"
        with open(cs_loc, "wb") as f:
            pickle.dump(self.c, f)

        n_features_loc = str(model_folder) + "/n_features.pickle"
        with open(n_features_loc, "wb") as f:
            pickle.dump(self.n_features, f)

    def score(self, X):
        feed_dict = {self.ph_x: X, self.is_train: False}
        return self.sess.run(self.sc, feed_dict=feed_dict)

    # ---------- Private methods ----------

    def print_summary_monitor(self, step):
        s1 = "Iter {1:^5} " + 10*"-"
        dt = datetime.datetime.now()
        s2 = " {}/{}/{} - {}:{}:{} ".format(
            dt.day, dt.month, dt.year % 100, dt.hour, dt.minute, dt.second)
        s2 = s2 + 5*"-"
        print((s1 + s2).format(DEF_PREC, step), flush=True)

    def monitor_fun(self, X, y, z, data_type, header=False):
        if header:
            s = ("dt | {1:^{0}} | {2:^{0}} | {3:^{0}} | {4:^{0}} "
                 + "| {5:^{0}} | {6:^{0}} | {7:^{0}} | {8:^{0}}")
            print(s.format(DEF_PREC+3, "r_cost", "cost", "auc", "r_auc",
                           "f_auc", "r_f_auc", "l2", "c"))

        if data_type == "test":
            X, y, z = gen_batch(X, y, z)
        else:
            X, y, z = gen_batch(X, y, z, n_batch=self.n_batch,
                                balanced=self.balanced)

        feed_dict = {self.ph_x: X,
                     self.ph_y: y,
                     self.ph_z: z,
                     self.ph_c: self.c,
                     self.is_train: False}

        if data_type == "test" and self.use_inc_auc_valid:
            n_inc = self.n_incomplete_valid
            feed_dict = dict(list(feed_dict.items())
                             + [(self.ph_incomplete, n_inc)])
        if data_type == "train" and self.use_inc_auc_train:
            n_inc = self.n_incomplete_train
            feed_dict = dict(list(feed_dict.items())
                             + [(self.ph_incomplete, n_inc)])

        monitored = [self.rel_cost, self.cost, self.auc, self.rel_auc,
                     self.fair_auc, self.rel_fair_auc, self.l2_weights]
        eval_test = self.sess.run(monitored, feed_dict=feed_dict)
        s_eval = " | ".join(["{0:+.{1}f}".format(a, DEF_PREC)
                             for a in (eval_test + [self.c])])
        print("{0:^2} | {1}".format(
            data_type[:2], s_eval))

        return eval_test[4]

    def init_weights(self, datasets=None):
        self.weights = {
            'w_out': tf.Variable(tf.random.normal(
                [self.n_features, 1], stddev=self.default_var_init),
                name="w_out")
        }

        self.biases = {'b_out': tf.Variable(
            tf.random.normal([1], stddev=self.default_var_init), name="b_out")}

        if self.net_depth > 0:
            other_weights = {"w_{}".format(i): tf.Variable(
                tf.random.normal([self.n_features, self.n_features],
                                 stddev=self.default_var_init),
                name="w_{}".format(i))
                             for i in range(0, self.net_depth)}

            other_biases = {'b_{}'.format(i): tf.Variable(
                tf.random.normal([self.n_features],
                                 stddev=self.default_var_init),
                name="b_{}".format(i))
                            for i in range(0, self.net_depth)}

            self.weights = dict(list(self.weights.items()) +
                                list(other_weights.items()))
            self.biases = dict(list(self.biases.items())
                               + list(other_biases.items()))

        self.l2_weights = 0
        for v in self.weights.values():
            self.l2_weights += tf.nn.l2_loss(v)
        for v in self.biases.values():
            self.l2_weights += tf.nn.l2_loss(v)

    def define_model(self):
        self.init_weights()

        # Define the placeholders:
        self.ph_x = tf.compat.v1.placeholder(tf.float32,
                                   [None, self.n_features],
                                   name="features")
        self.ph_y = tf.compat.v1.placeholder(tf.float32, [None], name="y")
        self.ph_z = tf.compat.v1.placeholder(tf.float32, [None], name="z")

        self.sc = self.define_net(self.ph_x)

    def auc_est_train(self, sc, y):
        if self.use_inc_auc_train:
            return incomplete_auc(sc, y, self.rel_indicator,
                                  self.ph_incomplete)
        else:
            return full_auc(sc, y, self.rel_indicator)

    def define_main_obj(self):
        if self.use_inc_auc_train:
            self.ph_incomplete = tf.compat.v1.placeholder(tf.int32, (), name="B")

        return self.auc_est_train(self.sc, self.ph_y)

    def get_filters_auc_defn(self):
        filt_0 = tf.equal(self.ph_z, 0)
        filt_1 = tf.equal(self.ph_z, 1)
        return filt_0, filt_1

    def define_cost(self, l2_reg):
        self.rel_auc, self.auc = self.define_main_obj()

        self.ph_c = tf.compat.v1.placeholder(tf.float32, (), name="c")

        filt_0, filt_1 = self.get_filters_auc_defn()
        rel_auc_0, auc_0 = self.auc_est_train(self.sc[filt_0],
                                              self.ph_y[filt_0])
        rel_auc_1, auc_1 = self.auc_est_train(self.sc[filt_1],
                                              self.ph_y[filt_1])

        self.rel_fair_auc = rel_auc_0 - rel_auc_1
        self.fair_auc = auc_0 - auc_1

        self.rel_cost = (self.rel_auc + self.reg*self.l2_weights)
        self.cost = self.auc

        if self.coef_lagrange > 0:
            self.rel_cost += self.coef_lagrange*self.ph_c*self.rel_fair_auc
            self.cost += self.coef_lagrange*self.fair_auc

    def define_net(self, ph_x):
        fc1 = ph_x
        self.is_train = tf.compat.v1.placeholder(tf.bool, name="is_train")

        for i in range(self.net_depth):
            fc1 = tf.add(tf.matmul(fc1, self.weights['w_{}'.format(i)]),
                         self.biases['b_{}'.format(i)])
            fc1 = tf.nn.relu(fc1)

        s_out = tf.add(tf.matmul(fc1, self.weights['w_out']),
                       self.biases['b_out'])
        s_out = tf.compat.v1.layers.batch_normalization(s_out, trainable=False,
                                              training=self.is_train)

        return s_out

    def rel_indicator(self, s):
        # 1/(1 + exp(x))
        return tf.sigmoid(-s)

    def adaptive_learning(self, val_fair_auc):
        if self.cooling_c:
            delta = self.adapt_step*(1 - self.cur_iter/self.n_iter)
            print(delta)
        else:
            delta = self.adapt_step

        if val_fair_auc > 0:
            self.c = min(self.c + delta, 1)
        else:
            self.c = max(self.c - delta, -1)
