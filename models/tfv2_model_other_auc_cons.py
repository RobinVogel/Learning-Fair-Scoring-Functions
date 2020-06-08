import tensorflow as tf
from models.tfv2_model_auc_cons import AUCCons


class AUCConsBNSP(AUCCons):
    def get_filters_auc_defn(self):
        filt_0 = tf.logical_or(tf.equal(self.ph_y, -1),
                               tf.equal(self.ph_z, 0))
        filt_1 = tf.logical_or(tf.equal(self.ph_y, -1),
                               tf.equal(self.ph_z, 1))
        return filt_0, filt_1


class AUCConsBPSN(AUCCons):
    def get_filters_auc_defn(self):
        filt_0 = tf.logical_or(tf.equal(self.ph_y, +1),
                               tf.equal(self.ph_z, 0))
        filt_1 = tf.logical_or(tf.equal(self.ph_y, +1),
                               tf.equal(self.ph_z, 1))
        return filt_0, filt_1


class AUCConsXAUC(AUCCons):
    def get_filters_auc_defn(self):
        filt_0 = tf.logical_or(
            tf.logical_and(tf.equal(self.ph_z, 0), tf.equal(self.ph_y, -1)),
            tf.logical_and(tf.equal(self.ph_z, 1), tf.equal(self.ph_y, +1))
        )
        filt_1 = tf.logical_or(
            tf.logical_and(tf.equal(self.ph_z, 1), tf.equal(self.ph_y, -1)),
            tf.logical_and(tf.equal(self.ph_z, 0), tf.equal(self.ph_y, +1))
        )
        return filt_0, filt_1
