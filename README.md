# 2D/3D Medical Imaging Registration via Fully Unsupervised ConvNet

<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>

This is a Keras/Tensorflow implementation of my paper:

<a href="https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14545">Chen, Junyu, et al. "Generating Anthropomorphic Phantoms Using Fully Unsupervised Deformable Image Registration with Convolutional Neural Networks." Medical Physics. Accepted Author Manuscript. doi:10.1002/mp.14545. 2020.</a>

<a href="http://jnm.snmjournals.org/content/61/supplement_1/522.short">Chen, Junyu, et al. "A fully unsupervised approach to create patient-like phantoms via Convolutional neural networks." Journal of Nuclear Medicine 61.supplement 1 (2020): 522-522.</a>

For most updated scripts (TensorFlow 2.X), see "**CNN_MedPhy_TF2x/**" folder. "CNN_MedPhy_TF2x/img.zip" contains an example image pair of moving and fixed images. Extract "pat.pkl" to "CNN_MedPhy_TF2x/", then you should be able to run "CNN_MedPhy_TF2x/register3D_example.py" without changing code. It took about 16 GB of GPU memory for this 3D registration.

We treat CNN as an optimization tool that iteratively minimizes the loss function via reparametrization in each
iteration. This means that the algorithm is fully unsupervised and thus **no prior training is required**. The registration loss function is defined as:

<img src="https://github.com/junyuchen245/Fully_unsupervised_CNN_registration/blob/master/CNNReg_arxiv/loss.png" width="300"/>

, where I_d and I_f are, respectively, the deformed and the fixed image, L_sim represents the loss function for image similarity, and R represents the regularization applied on the deformation field. 

## The effects of different loss functions:
<img src="https://github.com/junyuchen245/Fully_Unsupervised_CNN_Registration_Keras/blob/master/sample_imgs/loss_compare.PNG" width="500"/>

## The effects of different regularizations:
<img src="https://github.com/junyuchen245/Fully_Unsupervised_CNN_Registration_Keras/blob/master/sample_imgs/reg_compare.PNG" width="850"/>

## Sample results for XCAT phantom to and patient CT registration:
<img src="https://github.com/junyuchen245/Fully_Unsupervised_CNN_Registration_Keras/blob/master/sample_imgs/reg_results.PNG" width="800"/>

## Some deformed phantom and SPECT simulations:
<img src="https://github.com/junyuchen245/Fully_Unsupervised_CNN_Registration_Keras/blob/master/sample_imgs/SPECT_sim.PNG" width="1200"/>


 If you find this code is useful in your research, please consider to cite:

    @article{chen2020phantoms,
    author = {Chen, Junyu and Li, Ye and Du, Yong and Frey, Eric C.},
    title = {Generating Anthropomorphic Phantoms Using Fully Unsupervised Deformable Image Registration with Convolutional Neural Networks},
    journal = {Medical Physics},
    volume = {n/a},
    number = {n/a},
    pages = {},
    doi = {10.1002/mp.14545},
    url = {https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.14545},
    eprint = {https://aapm.onlinelibrary.wiley.com/doi/pdf/10.1002/mp.14545},
    }

 
 
### <a href="https://junyuchen245.github.io"> About Me</a>
