from tensorflow.keras import backend as K
import losses, nets
import sys, os, nrrd, pickle
from tensorflow.keras.optimizers import Adam
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import numpy as np
from skimage.transform import rescale, resize
from scipy.ndimage import gaussian_filter

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

'''
Model parameters
'''
sz_x = 160
sz_y = 160
sz_z = 256
ndim = 3
'''
Initialize GPU
'''
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Automatically choose an existing and supported device if the specified one does not exist
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)
    print('GPU Setup done')

'''
Initialize Models
'''
# Registration model
net = nets.unetND((sz_x, sz_y, sz_z, 1), ndim, up_size=(2, 2, 2), pool_size=(2, 2, 2), reg_wt = 0.8)
print(net.summary())
reg_model = Model(inputs=net.inputs, outputs=net.outputs)
reg_model.compile(optimizer=Adam(lr=1e-4), loss=losses.PCC_SSIM().loss)#'mean_squared_error')#NCC().loss losses.PCC_SSIM().loss)
# Apply deformation model
def_model = nets.mapping((sz_x, sz_y, sz_z, 1), (sz_x, sz_y, sz_z, 3), ndim)
def_atn_model = nets.mapping_bl((sz_x, sz_y, sz_z, 1), (sz_x, sz_y, sz_z, 3), ndim)
# Deformation visualization model
vis_model = Model(inputs=reg_model.inputs, outputs=reg_model.get_layer('deformField').output)

'''
Start registration
'''
# load moving image:
img_dir = 'pat.pkl'
moving, target = pkload(img_dir)
moving_org = moving
moving_org = resize(moving_org, (sz_x, sz_y, sz_z), anti_aliasing=False).reshape(1, sz_x, sz_y, sz_z, 1)


# load target image:
target = target
#target = target.reshape(1, sz_x, sz_y, sz_z,1)
target = gaussian_filter(target, sigma=0.1)
moving = gaussian_filter(moving, sigma=0.1)
# normalize images:
moving = (moving - moving.min())/(moving.max() - moving.min())
target = (target - target.min())/(target.max() - target.min())

# resize images:
moving       = resize(moving, (sz_x, sz_y, sz_z), anti_aliasing=True)
target = resize(target, (sz_x, sz_y, sz_z), anti_aliasing=False, order=1)

moving       = moving.reshape(1,sz_x, sz_y, sz_z,1)
target       = target.reshape(1,sz_x, sz_y, sz_z,1)

for iter_i in range(8000):
    reg_model.train_on_batch([moving, target], target)
    loss = reg_model.test_on_batch([moving, target], target)
    print('loss = ' + str(loss))
    if iter_i % 100 == 0:
        print(iter_i)
        def_moving = reg_model.predict([moving, target])
        vec_field = vis_model.predict([moving, target])
        def_moving_atn = def_atn_model.predict([moving_org, vec_field])
        plt.figure(num=None, figsize=(46, 6), dpi=150, facecolor='w', edgecolor='k')
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.imshow(def_moving_atn[0, :, :, 100, 0], cmap='gray')
        plt.title('Deformed Moving Image')
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.imshow(moving_org[0, :, :,100, 0], cmap='gray')
        plt.title('Moving Image')
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.imshow(target[0, :, :,100, 0], cmap='gray')
        plt.title('Target Image')
        plt.savefig('out_l2_0.8_p9.png')
        plt.close()
def_moving_atn = def_moving_atn.reshape(sz_x, sz_y, sz_z)
nrrd.write('/netscratch/jchen/CT_data_9_patients_3D/registered/p9_atn.nrrd', def_moving_atn)
