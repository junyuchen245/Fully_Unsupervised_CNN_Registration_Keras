import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

'''
normalized local cross-correlation
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
            self.win = [9] * ndims
        else:
            self.win = [self.win] * ndims

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
        u_I = I_sum/win_size # average_I
        u_J = J_sum/win_size # average_J

        cross = tf.math.reduce_sum((I-u_I)*(J-u_J))
        I_var = tf.math.reduce_sum((I-u_I)*(I-u_I))
        J_var = tf.math.reduce_sum((J-u_J)*(J-u_J))

        try:
            print(type(J_var))
        except tf.errors.InvalidArgumentError as e:
            print(e)

        # ncc
        cc = -cross / (tf.math.sqrt(I_var * J_var) + self.eps) + 1

        # return negative cc.
        return cc

    def loss(self, I, J):
        return self.ncc(I, J)

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
        top = tf.reduce_sum((y_pred - A_bar) * (y_true - B_bar))
        bottom = K.sqrt(tf.reduce_sum((y_pred - A_bar) ** 2) * tf.reduce_sum((y_true - B_bar) ** 2))
        return top/bottom

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
        top = tf.reduce_sum((y_pred - A_bar) * (y_true - B_bar))
        bottom = K.sqrt(tf.reduce_sum((y_pred - A_bar) ** 2) * tf.reduce_sum((y_true - B_bar) ** 2))
        return top/bottom

    def ssim(self, I, J):
        SSIM_idx = tf.image.ssim(I, J, max_val=self.maxVal)
        return SSIM_idx

    def loss(self, I, J):
        return self.pcc_wt*(1-self.pcc(I,J)) + self.ssim_wt*(1-self.ssim(I,J))

'''
Mutual information
'''
class MutualInformation():
    """
    Mutual Information for image-image pairs

    This function assumes that y_true and y_pred are both (batch_sizexheightxwidthxdepthxchan)

    """
    def __init__(self, sigma_ratio=1, max_clip=1, crop_background=False):


        """ prepare MI. """
        bin_centers = np.linspace(0.0, 1.0, num=32)#list(bin_centers)
        vol_bin_centers = K.variable(bin_centers)
        num_bins = len(bin_centers)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = K.variable(1 / (2 * np.square(sigma)))
        self.bin_centers = bin_centers
        self.max_clip = max_clip
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers
    def mi(self, y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, self.max_clip)
        y_true = K.clip(y_true, 0, self.max_clip)

        y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true = K.expand_dims(y_true, 2)
        y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(self.vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(self.vol_bin_centers, o)

        # compute image terms
        I_a = K.exp(- self.preterm * K.square(y_true - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        I_b = K.exp(- self.preterm * K.square(y_pred - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)

        return mi

    def loss(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)

class ParzenMutualInformation():
    """
    Parzen Mutual Information for image-image pairs
    """

    def __init__(self, sigma_ratio=1, max_clip=1, crop_background=False):

        """ prepare MI. """
        bin_centers = np.linspace(0.0, 1.0, num=32)  # list(bin_centers)
        vol_bin_centers = K.variable(bin_centers)
        num_bins = len(bin_centers)
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = K.variable(1 / (2 * np.square(sigma)))
        self.bin_centers = bin_centers
        self.max_clip = max_clip
        self.crop_background = crop_background
        self.num_bins = num_bins
        self.vol_bin_centers = vol_bin_centers

    def mi(self, y_true, y_pred):
        """ soft mutual info """
        y_pred = K.clip(y_pred, 0, self.max_clip)
        y_true = K.clip(y_true, 0, self.max_clip)

        y_true = K.reshape(y_true, (-1, K.prod(K.shape(y_true)[1:])))
        y_true = K.expand_dims(y_true, 2)
        y_pred = K.reshape(y_pred, (-1, K.prod(K.shape(y_pred)[1:])))
        y_pred = K.expand_dims(y_pred, 2)

        nb_voxels = tf.cast(K.shape(y_pred)[1], tf.float32)

        # reshape bin centers to be (1, 1, B)
        o = [1, 1, np.prod(self.vol_bin_centers.get_shape().as_list())]
        vbc = K.reshape(self.vol_bin_centers, o)

        # compute image terms
        #I_a =np.sqrt(6/(np.pi*1))*K.exp(-6*self.preterm*((y_true - vbc)**2)/1)#K.abs(y_true - vbc)#np.sqrt(6/(np.pi*1))*K.exp(-6*(y_true - vbc)**2/1)#K.abs(y_true - vbc)#K.exp(- self.preterm * K.square(y_true - vbc))
        I_a = K.exp(- self.preterm * K.square(y_true - vbc))
        I_a /= K.sum(I_a, -1, keepdims=True)

        #I_b = np.sqrt(6/(np.pi*4))*K.exp(-6*self.preterm*((y_pred - vbc)**2)/4)#K.exp(- self.preterm * K.square(y_pred - vbc))#K.abs(y_pred - vbc)#1/6*(4-6*(y_true - vbc)**2+3*K.abs(y_pred - vbc)**3)##K.exp(- self.preterm * K.square(y_pred - vbc))
        I_b = K.exp(- self.preterm * K.square(y_pred - vbc))
        I_b /= K.sum(I_b, -1, keepdims=True)

        # compute probabilities
        I_a_permute = K.permute_dimensions(I_a, (0, 2, 1))
        pab = K.batch_dot(I_a_permute, I_b)  # should be the right size now, nb_labels x nb_bins
        pab /= nb_voxels
        pa = tf.reduce_mean(I_a, 1, keep_dims=True)
        pb = tf.reduce_mean(I_b, 1, keep_dims=True)

        papb = K.batch_dot(K.permute_dimensions(pa, (0, 2, 1)), pb) + K.epsilon()
        mi = K.sum(K.sum(pab * K.log(pab / papb + K.epsilon()), 1), 1)

        return mi

    def loss(self, y_true, y_pred):
        return -self.mi(y_true, y_pred)