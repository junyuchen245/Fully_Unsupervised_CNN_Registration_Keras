"""
Junyu Chen
jchen245@jhmi.edu
"""
from keras import backend as K
import losses, nets
import sys, os, nrrd
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, load_model
import numpy as np
from skimage.transform import rescale, resize

def reset_weights(model):
    session = K.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

def detJacobian(Axij,Ayij, **kwargs):
    [xFX, xFY] = np.gradient(Axij)
    [yFX, yFY] = np.gradient(Ayij)
    jac_det = np.zeros(Axij.shape)
    for i in range(384):
        for j in range(384):
            jac_mij = [[xFX[i, j], xFY[i, j]], [yFX[i, j], yFY[i, j]]]
            jac_det[i, j] =  -np.linalg.det(jac_mij)
    return jac_det

def plot_grid(gridx,gridy, **kwargs):
    for i in range(gridx.shape[1]):
        plt.plot(gridx[i,:], gridy[i,:], **kwargs)
    for i in range(gridx.shape[0]):
        plt.plot(gridx[:,i], gridy[:,i], **kwargs)


def display_activation(activation_map, filter_num, img_size, layer_name):

    x = np.arange(0, 384, 1)
    y = np.arange(0, 384, 1)
    X, Y = np.meshgrid(x, y)

    u = activation_map[0, :, :, 0].reshape(384, 384)
    v = activation_map[0, :, :, 1].reshape(384, 384)

    phix = X
    phiy = Y
    down = 4
    for i in range(0, 384):
        for j in range(0, 384):
            # add the displacement for each p(k) in the sum
            phix[i, j] = phix[i, j] - u[i, j]
            phiy[i, j] = phiy[i, j] - v[i, j]
    phixdown = phix[0:-1:down, 0:-1:down]
    phiydown = phiy[0:-1:down, 0:-1:down]

    plt.figure(num=None, figsize=(45, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.subplot(1, 6, 1)
    plot_grid(phixdown, phiydown, color="C0")
    plt.gca().invert_yaxis()
    plt.title('Deformed grid')
    plt.subplot(1, 6, 2)
    detJac = detJacobian(phix, phiy)
    plt.imshow(detJac);
    plt.title('det(Jacobian)')
    plt.colorbar()
    return (detJac <= 0).sum()

def save_act_figs(act_model, img1, img2):
    activation_maps = act_model.predict([img1, img2])
    print(activation_maps.shape)
    NPdetJac = display_activation(activation_maps, 2, 384, 'conv11')
    return NPdetJac

'''
Model parameters
'''
sz_x = 384
sz_y = 384
sz_z = 1
ndim = 2
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
net = nets.unetND((sz_x, sz_y, sz_z), ndim)
print(net.summary())
reg_model = Model(inputs=net.inputs, outputs=net.outputs)
reg_model.compile(optimizer=Adam(lr=1e-3), loss=losses.PCC_SSIM().loss)#'mean_squared_error')#NCC().loss
# Apply deformation model
def_model = nets.mapping((sz_x, sz_y, 1), (sz_x, sz_y, 2))
# Deformation visualization model
vis_model = Model(inputs=reg_model.inputs, outputs=reg_model.get_layer('deformField').output)

'''
Start registration
'''
s_i = 10
# load moving image:
movingFile = np.load('data/xcat_pat_1.npz')
moving3D = movingFile['atnRef']
moving = moving3D[s_i,:,:]
moving3D_atlas = movingFile['actRef']
moving_atlas = moving3D_atlas[s_i,:,:]

# load target image:
target3D = np.load('data/CT_p1.npz')
target3D = target3D['a']
target   = target3D[s_i,:,:]

# normalize images:
moving = (moving - moving.min())/(moving.max() - moving.min())
target = (target - target.min())/(target.max() - target.min())
print(moving.shape)

# resize images:
moving       = resize(moving, (sz_x, sz_y), anti_aliasing=True)
moving_atlas = resize(moving_atlas, (sz_x, sz_y), anti_aliasing=False, order=0)

moving       = moving.reshape(1,sz_x, sz_y,1)
moving_atlas = moving_atlas.reshape(1,sz_x, sz_y,1)
target       = target.reshape(1,sz_x,sz_y,1)

for iter_i in range(4000):
    #reset_weights(reg_model)
    reg_model.train_on_batch([moving, target], target)
    loss = reg_model.test_on_batch([moving, target], target)
    print('loss = ' + str(loss))
    if (iter_i+1) % 100 == 0:
        print(iter_i)
        NPdetJac = save_act_figs(vis_model, moving, target)
        def_moving = reg_model.predict([moving, target])
        vec_field = vis_model.predict([moving, target])
        def_moving_atlas = def_model.predict([moving_atlas, vec_field])
        plt.subplot(1, 6, 3)
        plt.axis('off')
        plt.imshow(def_moving[0, :, :, 0], cmap='gray')
        plt.title('Deformed Moving Image')
        plt.subplot(1, 6, 4)
        plt.axis('off')
        plt.imshow(moving[0, :, :, 0], cmap='gray')
        plt.title('Moving Image')
        plt.subplot(1, 6, 5)
        plt.axis('off')
        plt.imshow(target[0, :, :, 0], cmap='gray')
        plt.title('Target Image')
        plt.subplot(1, 6, 6)
        plt.axis('off')
        def_moving_atlas = def_moving_atlas.reshape(384, 384)
        # img_bone = scipy.ndimage.binary_closing(img_bone).astype(img_bone.dtype)
        # img_bone = scipy.ndimage.binary_erosion(img_bone, structure=np.ones((3,3))).astype(img_bone.dtype)
        # img_bone = scipy.ndimage.binary_dilation(img_bone, structure=np.ones((3, 3))).astype(img_bone.dtype)
        plt.imshow((def_moving_atlas / np.max(def_moving_atlas)) ** 0.5, cmap='gray')
        plt.title('Deformed SPECT phantom')
        plt.savefig('out.png')
        plt.close()
        print(NPdetJac)
        a = input('Press Enter')

