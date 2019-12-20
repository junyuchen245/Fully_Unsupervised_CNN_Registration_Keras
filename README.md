# Medical Imaging Registration via Fully Unsupervised CNN

This is a simple implementation of my paper:

<a href="https://arxiv.org/abs/1912.02942">Chen, Junyu, et al. "Generating Patient-like Phantoms Using Fully Unsupervised Deformable Image Registration with Convolutional Neural Networks." arXiv preprint arXiv:1912.02942, 2019.</a>

We treat CNN as an optimization tool that iteratively minimizes the loss function via reparametrization in each
iteration. For now, the loss function is simply the image similarity measure (i.e., alpha = 0):
<img src="https://github.com/junyuchen245/Fully_unsupervised_CNN_registration/blob/master/CNNReg_arxiv/loss.png" width="300"/>

So the deformation field is not smooth. In our application, a smooth field is not very important, but different regularizations and more loss functions will be introduced in the final paper.

#### Some example data can be found here: <a href="https://drive.google.com/open?id=1cle8nV8g-xxt_SfaJxD-zMSnuXiZoygT"> required data</a>.


## Sample results for XCAT phantom to and patient CT registration:
<img src="https://github.com/junyuchen245/Fully_unsupervised_CNN_registration/blob/master/CNNReg_arxiv/Sample_out.png" width="1400"/>

### <a href="https://junyuchen245.github.io"> About Myself</a>
