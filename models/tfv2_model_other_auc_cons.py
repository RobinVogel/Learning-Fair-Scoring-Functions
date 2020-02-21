import tensorflow as tf
from models.tfv2_model_auc_cons import AUCCons


class AUCConsBNSP(AUCCons):
    # For German, eq 4
    def define_cost(self, l2_reg):
        self.rel_auc, self.auc = self.define_main_obj()

        self.ph_c = tf.compat.v1.placeholder(tf.float32, (), name="c")

        filt_0 = tf.logical_or(tf.equal(self.ph_y, -1),
                               tf.equal(self.ph_z, 0))
        rel_auc_0, auc_0 = self.auc_est_train(self.sc[filt_0],
                                              self.ph_y[filt_0])

        filt_1 = tf.logical_or(tf.equal(self.ph_y, -1),
                               tf.equal(self.ph_z, 1))
        rel_auc_1, auc_1 = self.auc_est_train(self.sc[filt_1],
                                              self.ph_y[filt_1])

        self.rel_fair_auc = rel_auc_0 - rel_auc_1
        self.fair_auc = auc_0 - auc_1

        self.rel_cost = (self.rel_auc + self.reg*self.l2_weights)
        self.cost = self.auc

        if self.coef_lagrange > 0:
            self.rel_cost += self.coef_lagrange*self.ph_c*self.rel_fair_auc
            self.cost += self.coef_lagrange*self.fair_auc


class AUCConsBPSN(AUCCons):
    # For Compas, eq 5
    def define_cost(self, l2_reg):
        self.rel_auc, self.auc = self.define_main_obj()

        self.ph_c = tf.compat.v1.placeholder(tf.float32, (), name="c")

        filt_0 = tf.logical_or(tf.equal(self.ph_y, +1),
                               tf.equal(self.ph_z, 0))
        rel_auc_0, auc_0 = self.auc_est_train(self.sc[filt_0],
                                              self.ph_y[filt_0])

        filt_1 = tf.logical_or(tf.equal(self.ph_y, +1),
                               tf.equal(self.ph_z, 1))
        rel_auc_1, auc_1 = self.auc_est_train(self.sc[filt_1],
                                              self.ph_y[filt_1])

        self.rel_fair_auc = rel_auc_0 - rel_auc_1
        self.fair_auc = auc_0 - auc_1

        self.rel_cost = (self.rel_auc + self.reg*self.l2_weights)
        self.cost = self.auc

        if self.coef_lagrange > 0:
            self.rel_cost += self.coef_lagrange*self.ph_c*self.rel_fair_auc
            self.cost += self.coef_lagrange*self.fair_auc
