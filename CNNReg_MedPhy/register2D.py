from keras import backend as K
import losses, nets
import sys, os, nrrd, utils
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model, load_model
import numpy as np
from skimage.transform import rescale, resize
import imageio
from scipy.ndimage import gaussian_filter

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
    down = 7
    for i in range(0, 384):
        for j in range(0, 384):
            # add the displacement for each p(k) in the sum
            phix[i, j] = phix[i, j] - u[i, j]
            phiy[i, j] = phiy[i, j] - v[i, j]
    phixdown = phix[0:-1:down, 0:-1:down]
    phiydown = phiy[0:-1:down, 0:-1:down]
    plt.figure(num=None, figsize=(46, 6), dpi=150, facecolor='w', edgecolor='k')
    plt.subplot(1, 7, 1)
    plot_grid(phixdown,phiydown, color="k")
    plt.gca().invert_yaxis()
    plt.title('Deformed grid')
    plt.subplot(1, 7, 2)
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
sz_x = 384
sz_y = 384
sz_z = 1
ndim = 2
'''
Initialize GPU
'''
if K.backend() == 'tensorflow':
    # Use only gpu #X (with tf.device(/gpu:
    # X) does not work)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
net = nets.unetND((sz_x, sz_y, sz_z), ndim, reg_wt=0.2)
print(net.summary())
reg_model = Model(inputs=net.inputs, outputs=net.outputs)
reg_model.compile(optimizer=Adam(lr=1e-3), loss=losses.PCC_SSIM(0.5).loss)#losses.ParzenMutualInformation().loss)#losses.PCC_SSIM(0.7).loss)#'mean_squared_error')#NCC().loss
# Apply deformation model
def_model = nets.mapping((sz_x, sz_y, 1), (sz_x, sz_y, 2))
# Deformation visualization model
vis_model = Model(inputs=reg_model.inputs, outputs=reg_model.get_layer('deformField').output)

'''
Start registration
'''
# load moving image:
imgdir = '/netscratch/jchen/deep_learning_projects/MEDPHY/TCIA_processed_data/processed/'
movingFile = np.load(imgdir+'xcat_p1.npz')
moving = movingFile['atnRef']
moving_atlas = movingFile['actRef']
print(moving.shape)


# load target image:
target3d = np.load(imgdir+'CT_p1_noArm.npz')
target3d = target3d['a']
target = target3d[75, :, :]
target_atlas = np.zeros_like(target3d);#np.load(imgdir+'p1_ctseg_torso.npz'); target_atlas = target_atlas['a']
target_atlas[target_atlas>0] = 1
target_atlas = target_atlas[75,:,:]

# normalize images:
moving = (moving - moving.min())/(moving.max() - moving.min())
target = (target - target.min())/(target.max() - target.min())
print(moving.shape)
moving       = moving[80,:,:].reshape(384, 384) #35
moving_atlas = moving_atlas[80,:,:].reshape(384, 384) #35

# resize images:
moving       = resize(moving, (sz_x, sz_y), anti_aliasing=True)
moving_atlas = resize(moving_atlas, (sz_x, sz_y), anti_aliasing=False, order=0)
target       = resize(target, (sz_x, sz_y), anti_aliasing=True)
target_atlas = resize(target_atlas, (sz_x, sz_y), anti_aliasing=False, order=0)
target = target.reshape(1,sz_x,sz_y,1)

#target = gaussian_filter(target, sigma=1)
#moving = gaussian_filter(moving, sigma=2.2)

moving       = moving.reshape(1,sz_x, sz_y,1)
moving_atlas = moving_atlas.reshape(1,sz_x, sz_y,1)
target       = target.reshape(1,sz_x,sz_y,1)

for iter_i in range(4000):
    #reset_weights(reg_model)
    loss = reg_model.train_on_batch([moving, target], target)
    print('loss = ' + str(loss))
    if iter_i % 100 == 0:
        print(iter_i)
        save_act_figs(vis_model, moving, target)
        def_moving = reg_model.predict([moving, target])
        vec_field = vis_model.predict([moving, target])
        def_moving_atlas = def_model.predict([moving_atlas, vec_field])
        plt.subplot(1, 7, 3)
        plt.axis('off')
        plt.imshow(def_moving[0, :, :, 0], cmap='gray')
        plt.title('Deformed Moving Image')
        plt.subplot(1, 7, 4)
        plt.imshow(def_moving[0, :, :, 0]**1.8, cmap='gray')
        plt.title('Deformed Moving Image')
        plt.subplot(1, 7, 5)
        plt.axis('off')
        plt.imshow(moving[0, :, :, 0], cmap='gray')
        plt.title('Moving Image')
        plt.subplot(1, 7, 6)
        plt.axis('off')
        plt.imshow(target[0, :, :, 0], cmap='gray')
        plt.title('Target Image')
        plt.subplot(1, 7, 7)
        plt.axis('off')

        def_moving_atlas = def_moving_atlas.reshape(384, 384)
        def_seg = np.zeros_like(def_moving_atlas)

        plt.imshow(def_moving_atlas, cmap='gray')
        plt.title('Diff. Image')
        plt.savefig('out.png')
        plt.close()

