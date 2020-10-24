import keras.layers as KL
from keras.layers import *
import sys, reg, image_warp
from keras.models import Model, load_model
import numpy as np
import scipy.stats as st
import tensorflow as tf
from scipy import signal
from keras.initializers import RandomNormal
import keras
from keras.layers.advanced_activations import PReLU

def concLayer(input):
    u = input[0]
    v = input[1]
    output = concatenate([u,v])
    return output
"""
Gaussian kernel
"""
def gkern2(n=41, std=20., normalised=True):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2*np.pi*(std**2))
    return gaussian2D

def gkern3(n=41, std=20., normalised=True):
    '''
    Generates a n x n matrix with a centered gaussian
    of standard deviation std centered on it. If normalised,
    its volume equals 1.'''
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    if normalised:
        gaussian3D /= (2*np.sqrt(2)*(np.pi*np.sqrt(np.pi))*(std**3))
    return gaussian3D

def kernel_init(shape):
    kernel = np.zeros(shape)
    kernel[:,:,0,0] = gkern2(shape[0], 160)
    return kernel

def kernel_init3d(shape):
    kernel = np.zeros(shape)
    kernel[:,:,0,0] = gkern3(shape[0])
    return kernel

def gaussian2d_deform(x_in, kernlen=(21, 21), ndims=2):
    Conv = getattr(KL, 'Conv%dD' % ndims)
    kernlen = list(kernlen)

    g1 = Conv(1, kernlen, kernel_initializer=kernel_init, padding="same", activity_regularizer=reg.Grad('l2',0))
    g1.trainable = False
    v = g1(x_in)
    return v

def gaussian3d_deform(x_in, kernlen=(21, 21, 21), ndims=2):
    Conv = getattr(KL, 'Conv%dD' % ndims)
    kernlen = list(kernlen)

    v = Conv(filters=ndims, kernel_size=1, padding='same')(x_in)
    g1 = Conv(1, kernlen, kernel_initializer=kernel_init3d, padding="same")
    g1.trainable = False
    v = g1(v)
    return v

def custom_act(x):
    mu = 0
    sigma = 20
    return x*K.exp(-0.5*((x-mu)/sigma)**2)#1/(sigma*np.sqrt(2*np.pi))*


def conv_block(x_in, nf, ndims=2, strides=1):
    Conv = getattr(KL, 'Conv%dD' % ndims)
    conv_out = Conv(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(x_in)
    conv_out = BatchNormalization()(conv_out)
    conv_out = Conv(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(conv_out)
    x_out = BatchNormalization()(conv_out)
    return x_out

def convPool_block(x_in, nf, ndims=2, pool_size=(2, 2)):
    MaxPool = getattr(KL, 'MaxPool%dD' % ndims)


    conv_out = conv_block(x_in, nf, ndims)
    x_out = MaxPool(pool_size=pool_size)(conv_out)
    return x_out

def convUp_block(x_in1, x_in2, nf, ndims = 2, up_size=(2, 2)):
    Conv = getattr(KL, 'Conv%dD' % ndims)
    UpSampling = getattr(KL, 'UpSampling%dD' % ndims)


    up_out = UpSampling(size=up_size)(x_in1)
    conv_out = Conv(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal', dilation_rate=1)(up_out)
    x_in = concatenate([conv_out, x_in2])
    x_out = conv_block(x_in, nf, ndims)
    return x_out


def unetND(input_size = (384,384,1), ndims=2, up_size=(2, 2), pool_size=(2, 2), reg_wt = 0):
    enc_nf = [24, 24, 24, 24]
    dec_nf = [24, 24, 24]
    gauss_filt_flag = False
    moving = Input(input_size)
    target = Input(input_size)

    x_in = concatenate([moving, target])
    Conv = getattr(KL, 'Conv%dD' % ndims)

    # encoder
    x_enc = [x_in]
    for i in range(len(enc_nf)-1):
        x_enc.append(convPool_block(x_enc[-1], enc_nf[i], ndims, pool_size))
    x_bot = conv_block(x_enc[-1], enc_nf[-1], ndims)
    # up-sample path (decoder)
    for i in range(2,len(dec_nf)+2):
        x = convUp_block(x_bot, x_enc[-i], dec_nf[i-2], ndims, up_size)
        x_bot = x

    # form deformation field
    x = Conv(filters=32, kernel_size=3, activation='relu', padding='same', dilation_rate=1)(x) #reg.NJ_reg(0.00001)#)#reg.TVNormReg(0.0000003)#reg.GradReg(0.0000003)
    x = BatchNormalization()(x) #reg.Grad('l2',1)
    if not gauss_filt_flag:
        x = Conv(filters=ndims, kernel_size=1, padding='same', name='deformField', activity_regularizer=reg.Grad('l2', reg_wt))(x)#0.0000000008

    else:
        '''
        Gaussian smoothing
        '''
        if ndims == 2:
            v1 = Conv(filters=1, kernel_size=1, padding='same', activity_regularizer=reg.Grad('l2',0))(x)#reg.Grad('l2',1))(x)#0.0000000008
            v1 = gaussian2d_deform(v1, (21, 21), ndims)
            v2 = Conv(filters=1, kernel_size=1, padding='same', activity_regularizer=reg.Grad('l2', 0))(x)
            v2 = gaussian2d_deform(v2, (21, 21), ndims)
            x = concatenate([v1, v2], name='deformField')
        else:
            v1 = Conv(filters=1, kernel_size=1, padding='same', activity_regularizer=reg.Grad('l2', 0))(x)  # reg.Grad('l2',1))(x)#0.0000000008
            v1 = gaussian3d_deform(v1, (21, 21, 21), ndims)
            v2 = Conv(filters=1, kernel_size=1, padding='same', activity_regularizer=reg.Grad('l2', 0))(x)
            v2 = gaussian3d_deform(v2, (21, 21, 21), ndims)
            v3 = Conv(filters=1, kernel_size=1, padding='same', activity_regularizer=reg.Grad('l2', 0))(x)
            v3 = gaussian3d_deform(v3, (21, 21, 21), ndims)
            x = concatenate([v1, v2, v3], name='deformField')

    # deform moving image
    Mapping = getattr(image_warp, 'Mapping%dD' % ndims)
    deformMoving = Lambda(Mapping, name='Mapping')([moving, x])

    model = Model(inputs=[moving, target], outputs=deformMoving)
    return model

def mapping(input_sz1 = (384, 384, 1), input_sz2 = (384, 384, 2), ndims=2):
    Mapping_nn = getattr(image_warp, 'Mapping%dD_nn' % ndims)
    moving    = Input(input_sz1)
    vec_field = Input(input_sz2)
    deformMoving = Lambda(Mapping_nn, name='Mapping_nn')([moving, vec_field])
    #deformMoving = nrn_layers.SpatialTransformer(interp_method='nearest', indexing='ij')([moving, vec_field])
    model = Model(inputs=[moving, vec_field], outputs=deformMoving)
    return model

def mapping_bl(input_sz1 = (384, 384, 1), input_sz2 = (384, 384, 2), ndims=2):
    Mapping = getattr(image_warp, 'Mapping%dD' % ndims)
    moving    = Input(input_sz1)
    vec_field = Input(input_sz2)
    deformMoving = Lambda(Mapping)([moving, vec_field])
    #deformMoving = nrn_layers.SpatialTransformer(interp_method='nearest', indexing='ij')([moving, vec_field])
    model = Model(inputs=[moving, vec_field], outputs=deformMoving)
    return model