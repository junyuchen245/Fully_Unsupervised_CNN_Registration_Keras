import tensorflow as tf
from keras import backend as K
import numpy as np
import sys

'''
normalized local cross-correlation
Modified from https://github.com/voxelmorph/voxelmorph
'''
class NCC():
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps


    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [7] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return - self.ncc(I, J)

'''
Structural Similarity Index
'''
class SSIM():
    def __init__(self, maxVal=1.0):
        self.maxVal = maxVal


    def ssim(self, I, J):
        SSIM_idx = tf.image.ssim(I, J, max_val=self.maxVal)
        return SSIM_idx

    def loss(self, I, J):
        return 1-self.ssim(I,J)

'''
Pearson's Correlation Coefficient
'''
class PCC():
    def __init__(self, maxVal=1.0):
        self.maxVal = maxVal


    def pcc(self, y_true, y_pred):
        A_bar = tf.reduce_mean(y_pred)
        B_bar = tf.reduce_mean(y_true)
        num = tf.reduce_sum((y_pred - A_bar) * (y_true - B_bar))
        den = K.sqrt(tf.reduce_sum((y_pred - A_bar) ** 2) * tf.reduce_sum((y_true - B_bar) ** 2))
        return num/den

    def loss(self, I, J):
        return 1-self.pcc(I,J)

'''
Weighted PCC + SSIM
'''
class PCC_SSIM():
    def __init__(self, pcc_wt=0.5, maxVal = 1.0):
        self.pcc_wt = pcc_wt
        self.ssim_wt = 1.0 - pcc_wt
        self.maxVal = maxVal

    def pcc(self, y_true, y_pred):
        A_bar = tf.reduce_mean(y_pred)
        B_bar = tf.reduce_mean(y_true)
        num = tf.reduce_sum((y_pred - A_bar) * (y_true - B_bar))
        den = K.sqrt(tf.reduce_sum((y_pred - A_bar) ** 2) * tf.reduce_sum((y_true - B_bar) ** 2))
        return num/den

    def ssim(self, I, J):
        SSIM_idx = tf.image.ssim(I, J, max_val=self.maxVal)
        return SSIM_idx

    def loss(self, I, J):
        return self.pcc_wt*(1-self.pcc(I,J)) + self.ssim_wt*(1-self.ssim(I,J))

