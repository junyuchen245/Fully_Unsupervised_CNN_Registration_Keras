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

def detJacobian(Axij,Ayij, **kwargs):
    [xFX, xFY] = np.gradient(Axij)
    [yFX, yFY] = np.gradient(Ayij)
    jac_det = np.zeros(Axij.shape)
    for i in range(384):
        for j in range(384):
            jac_mij = [[xFX[i, j], xFY[i, j]], [yFX[i, j], yFY[i, j]]]
            jac_det[i, j] =  np.linalg.det(jac_mij)
    return jac_det

def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], **kwargs)

def display_deformed_grid(feature_maps, layer_name):
    plt.figure()
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.imshow(feature_maps[0, :, :, 0], cmap='gray')
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.imshow(feature_maps[0, :, :, 1], cmap='gray')
    plt.savefig(layer_name + '.png')
    plt.close()

    x = np.arange(0, 384, 1)
    y = np.arange(0, 384, 1)
    X, Y = np.meshgrid(x, y)
    u =feature_maps[0, :, :, 0].reshape(384, 384)
    v =feature_maps[0, :, :, 1].reshape(384, 384)
    print('max u: '+str(np.max(u)))

    phix = X; phiy = Y
    down = 4
    for i in range(0, 384):
        for j in range(0, 384):
            # add the displacement for each p(k) in the sum
            phix[i, j] = phix[i, j] - u[i, j]
            phiy[i, j] = phiy[i, j] - v[i, j]
    phixdown = phix[0:-1:down, 0:-1:down]
    phiydown = phiy[0:-1:down, 0:-1:down]
    plt.figure(num=None, figsize=(46, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.subplot(1, 6, 1)
    plot_grid(phixdown,phiydown, color="C0")
    plt.gca().invert_yaxis()
    plt.title('Deformed grid')
    plt.subplot(1, 6, 2)
    detJac = detJacobian(phix, phiy)
    print('Min det(Jac): '+str(np.min(np.abs(detJac))))
    print('# det(Jac)<=0: ' + str((detJac == 0).sum()))
    plt.imshow(detJac); plt.title('det(Jacobian)')
    plt.colorbar()

def save_act_figs(act_model, img1, img2):
    activation_maps = act_model.predict([img1, img2])
    print(activation_maps.shape)
    display_deformed_grid(activation_maps, 'conv11')

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
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    # To constrain the use of gpu memory, otherwise all memory is used
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)
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
moving_file = np.load('/netscratch/jchen/CT_data_9_patients_3D/phan/xcat_phan.npz')
moving = moving_file['atnRef']
moving_atlas = moving_file['actRef']
moving_org = np.swapaxes(moving,0,2)
moving_org = np.rot90(moving_org,-1)
moving_org = np.flip(moving_org,1)
print(moving_org.shape)
moving_org = resize(moving_org, (sz_x, sz_y, sz_z), anti_aliasing=False).reshape(1, sz_x, sz_y, sz_z, 1)


# load target image:
target = np.load('/netscratch/jchen/CT_data_9_patients_3D/CT/CT_p9.npz')
target = target['a']
#target = target.reshape(1, sz_x, sz_y, sz_z,1)
target = gaussian_filter(target, sigma=0.3)
moving = gaussian_filter(moving, sigma=0.5)
# normalize images:
moving = (moving - moving.min())/(moving.max() - moving.min())
target = (target - target.min())/(target.max() - target.min())

moving = np.swapaxes(moving,0,2)
moving = np.rot90(moving,-1)
moving = np.flip(moving,1)
moving_atlas = np.swapaxes(moving_atlas,0,2)
moving_atlas = np.rot90(moving_atlas,-1)
moving_atlas = np.flip(moving_atlas,1)
target = np.swapaxes(target,0,2)
target = np.rot90(target,-1)

# resize images:
moving       = resize(moving, (sz_x, sz_y, sz_z), anti_aliasing=True)
moving_atlas = resize(moving_atlas, (sz_x, sz_y, sz_z), anti_aliasing=False, order=0)
target = resize(target, (sz_x, sz_y, sz_z), anti_aliasing=False, order=1)

moving       = moving.reshape(1,sz_x, sz_y, sz_z,1)
moving_atlas = moving_atlas.reshape(1,sz_x, sz_y, sz_z,1)
target       = target.reshape(1,sz_x, sz_y, sz_z,1)

for iter_i in range(8000):
    reg_model.train_on_batch([moving, target], target)
    loss = reg_model.test_on_batch([moving, target], target)
    print('loss = ' + str(loss))
    if iter_i % 100 == 0:
        print(iter_i)
        save_act_figs(vis_model, moving, target)
        def_moving = reg_model.predict([moving, target])
        vec_field = vis_model.predict([moving, target])
        def_moving_atlas = def_model.predict([moving_atlas, vec_field])
        def_moving_atn = def_atn_model.predict([moving_org, vec_field])
        plt.figure(num=None, figsize=(46, 6), dpi=150, facecolor='w', edgecolor='k')
        plt.subplot(1, 4, 1)
        plt.axis('off')
        plt.imshow(def_moving_atn[0, :, :, 100, 0], cmap='gray')
        plt.title('Deformed Moving Image')
        plt.subplot(1, 4, 2)
        plt.axis('off')
        plt.imshow(moving_org[0, :, :,100, 0], cmap='gray')
        plt.title('Moving Image')
        plt.subplot(1, 4, 3)
        plt.axis('off')
        plt.imshow(target[0, :, :,100, 0], cmap='gray')
        plt.title('Target Image')
        plt.subplot(1, 4, 4)
        plt.axis('off')
        atlas_tmp = def_moving_atlas[0, :, :,100, 0]
        plt.imshow((atlas_tmp / np.max(atlas_tmp)) ** 0.5, cmap='gray')
        plt.title('Diff. Image')
        plt.savefig('out_l2_0.8_p9.png')
        plt.close()
def_moving_atn = def_moving_atn.reshape(sz_x, sz_y, sz_z)
def_moving_atlas = def_moving_atlas.reshape(sz_x, sz_y, sz_z)
nrrd.write('/netscratch/jchen/CT_data_9_patients_3D/registered/p9_atn.nrrd', def_moving_atn)
nrrd.write('/netscratch/jchen/CT_data_9_patients_3D/registered/p9_act.nrrd', def_moving_atlas)
