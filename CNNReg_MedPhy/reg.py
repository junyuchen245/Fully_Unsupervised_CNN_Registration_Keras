from keras import backend as K
import tensorflow as tf

class l2Reg():
    '''
    l2
    '''
    def __init__(self, wt=0):
        self.wt = K.cast_to_floatx(wt)

    def __call__(self, x):
        regularization = self.wt * tf.reduce_mean(x * x)
        return regularization


class TVNormReg():
    '''
    Total Variation Norm
    '''
    def __init__(self, wt=0):
        self.wt = K.cast_to_floatx(wt)

    def __call__(self, x):
        xx = x[:, :, :, 0]
        yy = x[:, :, :, 1]
        regularization = self.wt * tf.reduce_sum(tf.image.total_variation(xx)+tf.image.total_variation(yy))
        return regularization

class NJ_reg():
    '''
    Determinant of Jacobian regularization

    obtained from:
    https://github.com/dykuang/Medical-image-registration
    '''
    def __init__(self, wt=0):
        self.wt = K.cast_to_floatx(wt)

    def Get_Ja(self, v):
        D_x = (v[:, 1:, :-1, :] - v[:, :-1, :-1, :])

        D_y = (v[:, :-1, 1:, :] - v[:, :-1, :-1, :])

        D1 = (D_x[..., 0] + 1) * (D_y[..., 1] + 1) - (D_x[..., 1]) * (D_y[..., 0])

        # D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

        # D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

        # D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

        return D1  # -D2+D3

    def __call__(self, v) :
        '''
        Penalizing locations where Jacobian has negative determinants
        '''
        Neg_Jac = 0.5*(tf.abs(self.Get_Ja(v)) - self.Get_Ja(v))
        return self.wt*tf.reduce_sum(Neg_Jac)

class Grad():
    """
    N-D gradient loss

    obtained from:
    https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, penalty='l1', wt=0):
        self.penalty = penalty
        self.wt = wt

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def __call__(self, y_pred):
        if self.penalty == 'l1':
            df = [tf.reduce_mean(tf.abs(f)) for f in self._diffs(y_pred)]
        elif self.penalty == 'l2':
            df = [tf.reduce_mean(f * f) for f in self._diffs(y_pred)]

        return self.wt*tf.add_n(df) / len(df)


class Grad_NJ_reg():
    def __init__(self, wt1=1, wt2=0.0001):
        self.wt_diff = K.cast_to_floatx(wt1)
        self.wt_npj = K.cast_to_floatx(wt2)

    def _diffs(self, y):
        vol_shape = y.get_shape().as_list()[1:-1]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 1
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d + 1, ndims + 2)]
            y = K.permute_dimensions(y, r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            df[i] = K.permute_dimensions(dfi, r)

        return df

    def Get_Ja(self, v):

        '''
        Calculate the Jacobian value at each point of the displacement map having
        size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
        '''
        D_x = (v[:,1:,:-1,:] - v[:,:-1,:-1,:])

        D_y = (v[:,:-1,1:,:] - v[:,:-1,:-1,:])

        D1 = (D_x[...,0]+1)*(D_y[...,1]+1)-(D_x[...,1])*(D_y[...,0])

        #D1 = (D_x[...,0]+1)*( (D_y[...,1]+1)*(D_z[...,2]+1) - D_z[...,1]*D_y[...,2])

        #D2 = (D_x[...,1])*(D_y[...,0]*(D_z[...,2]+1) - D_y[...,2]*D_x[...,0])

        #D3 = (D_x[...,2])*(D_y[...,0]*D_z[...,1] - (D_y[...,1]+1)*D_z[...,0])

        return D1#-D2+D3

    def __call__(self, v) :
        df = [tf.reduce_mean(f * f) for f in self._diffs(v)]
        Neg_Jac = 0.5*(tf.abs(self.Get_Ja(v)) - self.Get_Ja(v))
        return self.wt_diff*(tf.add_n(df) / len(df))+self.wt_npj*tf.reduce_sum(Neg_Jac)