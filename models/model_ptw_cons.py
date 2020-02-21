# The adaptative step is related to the evaluation step
# since the adaptation is calibrated on its results.
import itertools
from collections.abc import Iterable
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.model_auc_cons import (AUCCons, gen_batch, incomplete_auc,
                                   full_auc, DEF_PREC, DEF_C)

DEF_MON_PTS = {-1: [0.25], +1: [0.25]}
EXPLICIT_C_AND_RATES = True
DEF_ADAPT_STEP_T = 0.01


class PtwCons(AUCCons):

    def __init__(self, param_files=None):
        self.mon_pts = DEF_MON_PTS
        self.adapt_step_t = DEF_ADAPT_STEP_T

        super().__init__(param_files=param_files)

        d = dict()
        for k, v in self.mon_pts.items():
            d[int(k)] = v
        self.mon_pts = d

        self.adapt_step_t = float(self.adapt_step_t)

        n_mon = len(list(itertools.chain.from_iterable(self.mon_pts.values())))
        self.coefs_lagrange = [self.coef_lagrange]*n_mon
        self.c = np.array([DEF_C]*n_mon)
        assert len(self.c) == n_mon

        self.n_c = sum([len(v) for v in self.mon_pts.values()])
        self.thre = np.array([float(v) for v in self.c])

        self.rel_ptw_val = None
        self.ptw_val = None
        self.tprs = None
        self.fprs = None

        self.ph_thre = None

    # ---------- Private methods ----------

    def monitor_fun(self, X, y, z, data_type, header=False):
        if header:
            s = "dt | {1:^{0}} | {2:^{0}} | {3:^{0}} | {4:^{0}} | {5:^{0}}"
            base_s = s.format(
                DEF_PREC+3, "r_cost", "cost", "auc", "r_auc", "l2")
            s = (" | {1:^{0}} | {2:^{0}} | {3:^{0}}"
                 + " | {4:^{0}} | {5:^{0}} | {6:^{0}}")
            summaries_s = s.format(DEF_PREC+3,
                                   "mean_c", "var_c",
                                   "dFPR_m", "dFPR_v",
                                   "dTPR_m", "dTPR_v")
            print(base_s + summaries_s)

        if data_type == "test":
            X, y, z = gen_batch(X, y, z)
        else:
            X, y, z = gen_batch(X, y, z, n_batch=self.n_batch,
                                balanced=self.balanced)

        test_dict = {self.ph_x: X,
                     self.ph_y: y,
                     self.ph_z: z,
                     self.ph_c: self.c,
                     self.ph_thre: self.thre,
                     self.is_train: False}

        if data_type == "test" and self.use_inc_auc_valid:
            n_inc = self.n_incomplete_valid
            test_dict = dict(list(test_dict.items())
                             + [(self.ph_incomplete, n_inc)])
        if data_type == "train" and self.use_inc_auc_train:
            n_inc = self.n_incomplete_train
            test_dict = dict(list(test_dict.items())
                             + [(self.ph_incomplete, n_inc)])

        monitored = [self.rel_cost, self.cost, self.auc,  # 0, 1, 2
                     self.rel_auc, self.l2_weights,  # 3, 4
                     self.fprs, self.tprs]  # 5, 6
        eval_test = self.sess.run(monitored, feed_dict=test_dict)

        fprs, tprs = eval_test[5:]
        fprs, tprs = np.array(fprs), np.array(tprs)

        mon_vals = np.array(list(itertools.chain.from_iterable(
            self.mon_pts.values())))

        target_tprs = mon_vals
        dev_tpr = tprs - target_tprs
        dtprs = np.abs(dev_tpr)

        target_fprs = mon_vals
        dev_fpr = fprs - target_fprs
        dfprs = np.abs(dev_fpr)

        other_eval = [np.mean(self.c), np.var(self.c),
                      np.mean(dfprs), np.var(dfprs),
                      np.mean(dtprs), np.var(dtprs)]

        final_eval = eval_test[:5] + other_eval

        s_eval = " | ".join(["{0:+.{1}f}".format(a, DEF_PREC)
                             for a in final_eval])

        print("{0:^2} | {1}".format(data_type[:2], s_eval))

        if EXPLICIT_C_AND_RATES and data_type == "test":
            c_string = " | ".join(["{0:+.{1}f}".format(a, DEF_PREC)
                                   for a in self.c])
            biases_string = " | ".join(["{0:+.{1}f}".format(a, DEF_PREC)
                                        for a in self.thre])
            fpr_string = " | ".join(["{0:+.{1}f}".format(a, DEF_PREC)
                                     for a in fprs])
            tpr_string = " | ".join(["{0:+.{1}f}".format(a, DEF_PREC)
                                     for a in tprs])
            print("c: {}".format(c_string))
            print("biases: {}".format(biases_string))
            print("FPRs: {}".format(fpr_string))
            print("TPRs: {}".format(tpr_string))

        return fprs, tprs

    def base_train_dict(self, X_tr, y_tr, z_tr):
        return {self.ph_x: X_tr, self.ph_y: y_tr, self.ph_z: z_tr,
                self.ph_c: self.c, self.ph_thre: self.thre,
                self.is_train: True}

    def define_cost(self, l2_reg):
        self.rel_auc, self.auc = self.define_main_obj()

        self.ph_c = tf.placeholder(tf.float32, [self.n_c], name="c")
        self.ph_thre = tf.placeholder(tf.float32, [self.n_c], name="thre")

        self.rel_ptw_val = list()
        self.ptw_val = list()
        self.tprs = list()
        self.fprs = list()

        ind = 0
        for y_val in self.mon_pts.keys():
            # First we select values corresponding to y == y_val
            filt_sel = tf.equal(self.ph_y, y_val)
            z_sel = self.ph_z[filt_sel]
            sc_sel = self.sc[filt_sel]
            for mon_pt in self.mon_pts[y_val]:
                # Then we compute the asymmetric cost value
                thre = self.ph_thre[ind]
                c = self.ph_c[ind]
                biased_sc = sc_sel - thre

                filt_1 = tf.equal(z_sel, 1)
                filt_0 = tf.logical_not(filt_1)

                # Check whether the batch has any samples from Z=0 or Z=1
                n_z_1 = tf.reduce_sum(tf.cast(filt_1, tf.int32))
                no_z_1 = tf.equal(n_z_1, 0)
                no_z_0 = tf.equal(tf.shape(filt_1)[0] - n_z_1, 0)

                # rel_indicator is sigmoid(-s), i.e. I{x<0}
                # z = 0 negative # z = 1 positive
                # 1/(1 + exp(-x)) -> I{ x > 0 }
                r_p_pred = self.rel_indicator(-biased_sc)

                rel_tpr = tf.cond(no_z_1, lambda: 0.,
                                  lambda: tf.reduce_mean(r_p_pred[filt_1]))
                rel_fpr = tf.cond(no_z_0, lambda: 0.,
                                  lambda: tf.reduce_mean(r_p_pred[filt_0]))

                # Monitored values:
                tpr = tf.cond(no_z_1, lambda: 0.,
                              lambda: tf.cast(tf.greater(biased_sc[filt_1], 0),
                                              tf.float32))
                fpr = tf.cond(no_z_0, lambda: 0.,
                              lambda: tf.cast(tf.greater(biased_sc[filt_0], 0),
                                              tf.float32))

                mean_tpr = tf.reduce_mean(tpr)
                mean_fpr = tf.reduce_mean(fpr)

                # The cost is fpr - tpr
                rel_cost = c*(rel_fpr - rel_tpr)
                cost = c*(mean_fpr - mean_tpr)

                # We save the monitored values
                self.rel_ptw_val.append(rel_cost)
                self.ptw_val.append(cost)
                self.tprs.append(mean_tpr)
                self.fprs.append(mean_fpr)

                ind += 1

        self.rel_cost = self.rel_auc + self.reg*self.l2_weights
        self.cost = self.auc

        for i in range(self.n_c):
            if self.coefs_lagrange[i] > 0:
                self.rel_cost += self.coefs_lagrange[i]*self.rel_ptw_val[i]
                self.cost += self.coefs_lagrange[i]*self.ptw_val[i]

    def adaptive_learning(self, dev_ptw_val):
        fprs, tprs = dev_ptw_val

        if self.cooling_c:
            # deltac = self.adapt_step*(1 - self.cur_iter/self.n_iter)
            # The two parameters are determined with exp(-10^4/T) = adapt_step
            # and exp(-5*10^4/T) = adapt_step/100
            T = (2*10000)/np.log(10)
            lambd = np.sqrt(10)*self.adapt_step
            deltac = lambd*np.exp(-self.cur_iter/T)
            print("deltac: {}".format(deltac))
        else:
            deltac = self.adapt_step
            deltat = self.adapt_step_t

        mon_vals = np.array(list(itertools.chain.from_iterable(
            self.mon_pts.values())))

        devs = (fprs - tprs)/2
        means = (fprs + tprs)/2
        for i in range(self.n_c):
            if np.abs(means[i] - mon_vals[i]) > np.abs(devs[i]):
                # Change the threshold
                if means[i] > mon_vals[i]:
                    self.thre[i] = self.thre[i] + deltat
                else:
                    self.thre[i] = self.thre[i] - deltat
            else:
                if devs[i] > 0:
                    self.c[i] = min(self.c[i] + deltac, 1)
                else:
                    self.c[i] = max(self.c[i] - deltac, -1)
