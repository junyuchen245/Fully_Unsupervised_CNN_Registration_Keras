"""
Junyu Chen
jchen245@jhmi.edu
"""
import keras.layers as KL
from keras.layers import *
import sys, image_warp
from keras.models import Model, load_model
import numpy as np
import scipy.stats as st
import tensorflow as tf
from scipy import signal

def Mapping_nn(input):
    a = input[0]
    b = input[1]
    output = image_warp.image_warp(a,b,'NN',name='dense_image_warp')
    return output

def Mapping(input):
    a = input[0]
    b = input[1]
    output = tf.contrib.image.dense_image_warp(a,b,name='dense_image_warp')
    return output

def concLayer(input):
    u = input[0]
    v = input[1]
    output = concatenate([u,v])
    return output

def conv_block(x_in, nf, ndims=2, strides=1):
    Conv = getattr(KL, 'Conv%dD' % ndims)


    conv_out = Conv(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x_in)
    conv_out = BatchNormalization()(conv_out)
    conv_out = Conv(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv_out)
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
    conv_out = Conv(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up_out)
    x_in = concatenate([conv_out, x_in2])
    x_out = conv_block(x_in, nf, ndims)
    return x_out


def unetND(input_size = (384,384,1), ndims=2, up_size=(2, 2), pool_size=(2, 2)):
    #enc_nf = [16, 32, 64, 128, 256] # 384 192 96 48 24
    #dec_nf = [128, 64, 32, 16]
    enc_nf = [24, 24, 24, 24]
    dec_nf = [24, 24, 24]

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
    x = Conv(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x) #reg.Grad('l2',1)
    x = Conv(filters=ndims, kernel_size=1, padding='same', name='deformField')(x)

    # deform moving image
    deformMoving = Lambda(Mapping, name='Mapping')([moving, x])

    model = Model(inputs=[moving, target], outputs=deformMoving)
    return model

def mapping(input_sz1 = (384, 384, 1), input_sz2 = (384, 384, 2)):
    moving    = Input(input_sz1)
    vec_field = Input(input_sz2)
    deformMoving = Lambda(Mapping, name='Mapping_nn')([moving, vec_field])
    #deformMoving = nrn_layers.SpatialTransformer(interp_method='nearest', indexing='ij')([moving, vec_field])
    model = Model(inputs=[moving, vec_field], outputs=deformMoving)
    return model




