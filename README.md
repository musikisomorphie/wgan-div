# wgan-div


Code for reproducing experiments in [Wasserstein Divergence for GANs](https://arxiv.org/abs/1712.01026).

Our code is built upon [WGAN-GP](https://github.com/igul222/improved_wgan_training).

## Prerequisites
For the running enviromnment please check https://github.com/igul222/improved_wgan_training

For the training data such as Cifar10, CelebA, LSUN etc download them on the official website accordingly

For FID evaluation download the Inception model from http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz

For the precalculated statistics download them from http://bioinf.jku.at/research/ttur/

## Run the model
Specify DATA_DIR, INCEPTION_DIR and STAT_FILE in wgan-div.py, then run

python wgan-div.py

The default training image size is 64*64.
