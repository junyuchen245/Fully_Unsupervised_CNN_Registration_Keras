# Medical Imaging Registration via Fully Unsupervised CNN

This is a Keras/Tensorflow implementation of my paper:

<a href="https://arxiv.org/abs/1912.02942">Chen, Junyu, et al. "Generating Anthropomorphic Phantoms Using Fully Unsupervised Deformable Image Registration with Convolutional Neural Networks." Medical Physics. Accepted Author Manuscript. doi:10.1002/mp.14545. 2020.</a>

We treat CNN as an optimization tool that iteratively minimizes the loss function via reparametrization in each
iteration. This means that the algorithm is fully unsupervised and thus **no prior training is required**. For now, the loss function is simply the image similarity measure (i.e., alpha = 0):

<img src="https://github.com/junyuchen245/Fully_unsupervised_CNN_registration/blob/master/CNNReg_arxiv/loss.png" width="300"/>

, where I_d and I_f are the deformed and the fixed image, respectively. Since alpha = 0, the deformation field is not smooth. In our application, a smooth field is not very important, but different regularizations and more loss functions will be introduced in the final paper.

#### Some example data can be found here: <a href="https://drive.google.com/open?id=1cle8nV8g-xxt_SfaJxD-zMSnuXiZoygT"> required data</a>.

This program does 2D registration, the 3D version will be uploaded in the future.

## Sample results for XCAT phantom to and patient CT registration:
<img src="https://github.com/junyuchen245/Fully_unsupervised_CNN_registration/blob/master/CNNReg_arxiv/Sample_out.png" width="1400"/>


                                        Deformed Moving    Moving (XCAT)       Fixed (CT)     Deformed Labels
 If you find this code is useful in your research, please consider to cite:

    @article{doi:10.1002/mp.14545,
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

 
 
### <a href="https://junyuchen245.github.io"> About Myself</a>
