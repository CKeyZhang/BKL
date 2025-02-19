# BKL

## Abstract
Knowledge distillation is widely used in the model compression of deep neural networks.
As a metric for quantifying the output probability distribution of models, Kullback-Leibler (KL) divergence is commonly employed in mainstream knowledge distillation methods.
However, we reveal two issues that limit the training effectiveness of the original KL divergence: 1. The phenomenon of category loss response bias caused by asymmetry; 2. The significant suppression of loss response for background classes.
To tackle these issues, we propose the Balanced Kullback-Leibler Divergence (BKL), which reconstructs the KL divergence to modulate its functional properties in a flexible and controllable manner.
In BKL, we propose a probability mapping that imparts symmetry and linear variation to its functional properties within a new probability space.
We validated the effectiveness of BKL on tasks across various modalities separately.
Extensive experimental results on demonstrates that BKL  outperforms baseline methods and enhances the consistency of outputs between the teacher and student models.

## Installation
We have validated the effectiveness of our method on the CIFAR100 dataset and the GLUE benchmark. For experimental setup and environment configuration, please refer to  and  respectively.
