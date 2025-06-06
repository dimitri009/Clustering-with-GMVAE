## GMVAE: Gaussian Mixture Variational Autoencoder

This repository contains a PyTorch implementation of a Gaussian Mixture Variational Autoencoder (GMVAE) for unsupervised clustering.

The code is originally from : [@jariasf](https://github.com/jariasf/GMVAE)

I have added some minor changes.

### Overview

![GMVAE](fig/GMVAE.jpg)
[image credits](https://ieeexplore.ieee.org/abstract/document/8412085). 

The code builds on the foundational work of [Kingma et al. (2014)](https://arxiv.org/pdf/1406.5298) — specifically the M2 model for semi-supervised learning — and integrates key ideas from the Gumbel-Softmax reparameterization trick introduced in [Jang et al. (2016)](https://www.google.com/url?q=https%3A%2F%2Farxiv.org%2Fpdf%2F1611.01144) to allow differentiable sampling of discrete latent variables.

Instead of marginalizing out the categorical variable as done in M2, they sample it using Gumbel-Softmax, and further extend the generative model to use a mixture of Gaussians in the latent space. This allows the GMVAE to:

* Learn cluster-friendly latent representations,

* Encourage interpretable structure in the latent space,

* And perform effective unsupervised clustering on datasets such as MNIST.

### Install requirements

* Scikit-learn:

`pip install -U scikit-learn`

* Pytorch:

`pip install torch torchvision` (without CUDA) or 

`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126` (with CUDA 12.6)

* Pandas and Matplot:

`pip install pandas` , `pip install matplotlib `

### Experiment

#### Dataset

The dataset used for the experiments is the [mnist dataset](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html):

![MNIST](fig/mnist.png)

#### Training per 50 epochs

The ARI and NMI scores during the training:

![ARI_NMI](fig/ari_nmi_50.png)

The ARI, NMI, ACC scores on the test set: 

| NMI   | ARI | ACC |
|-------|-----|------------|
| 72.84 |67.34|78.28|

Visualisation of the feature latent space

![TSNE](fig/TSE_50.png)

Image Reconstruction

![IMG_REC](fig/image_rec_50.png)

Image generation

![IMG_GEN](fig/img_ran_50.png)

#### Training per 100 epochs