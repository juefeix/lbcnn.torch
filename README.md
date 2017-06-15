# LBCNN
Torch implementation of CVPR'17 - Local Binary Convolutional Neural Networks http://xujuefei.com/lbcnn.html


(this repo is under construction)


### Abstract
We propose **local binary convolution (LBC)**, an efficient alternative to convolutional layers in standard convolutional neural networks (CNN). The design principles of LBC are motivated by local binary patterns (LBP). The LBC layer comprises of a set of fixed sparse pre-defined binary convolutional filters that are not updated during the training process, a non-linear activation function and a set of learnable linear weights. The linear weights combine the activated filter responses to approximate the corresponding activated filter responses of a standard convolutional layer. The LBC layer affords significant parameter savings, 9x to 169x in the number of learnable parameters compared to a standard convolutional layer. Furthermore, the sparse and binary nature of the weights also results in up to 9x to 169x savings in model size compared to a standard convolutional layer. We demonstrate both theoretically and experimentally that our local binary convolution layer is a good approximation of a standard convolutional layer. Empirically, CNNs with LBC layers, called **local binary convolutional neural networks (LBCNN)**, achieves performance parity with regular CNNs on a range of visual datasets (MNIST, SVHN, CIFAR-10, and ImageNet) while enjoying significant computational savings.

***

### Overview
<img src="http://xujuefei.com/lbcnn_image/01_LBP_3_5.png" width="48">
<img src="http://xujuefei.com/lbcnn_image/01_LBP_3_5.png" title="Local Binary Patterns" style="width: 100px;"/>
<img src="http://xujuefei.com/lbcnn_image/02_LBP.png" title="Basic Idea of Local Binary Patterns" style="width: 200px;"/>


We draw inspiration from local binary patterns that have been very successfully used for facial analysis.


<img src="http://xujuefei.com/lbcnn_image/03_LBCNN_CNN.png" title="Local Binary Convolutional Module" style="width: 800px;"/>


Our LBCNN module is designed to approximate a fully learnable dense CNN module.

<img src="http://xujuefei.com/lbcnn_image/04_sparsity_2.png" title="Sparsity: 2" style="width: 260px;"/>
<img src="http://xujuefei.com/lbcnn_image/04_sparsity_4.png" title="Sparsity: 4" style="width: 260px;"/>
<img src="http://xujuefei.com/lbcnn_image/04_sparsity_9.png" title="Sparsity: 9" style="width: 260px;"/>


Binary convolutional kernels with different sparsity levels.

<img src="http://xujuefei.com/lbcnn_image/05_theory.png" title="Main approximation theorem" style="width: 800px;"/>


***

### Contributions

* Convolutional kernels inspired by local binary patterns.
* Convolutional neural network architecture with **non-mutable randomized sparse binary convolutional kernels**.
* Lightweight CNN with massive computational and memory savings.

***

### References

* Felix Juefei-Xu, Vishnu Naresh Boddeti, and Marios Savvides, [**Local Binary Convolutional Neural Networks**](felix_cvpr17_lbcnn.pdf),
* To appear in *IEEE Computer Vision and Pattern Recognition (CVPR), 2017*. (Spotlight Oral Presentation)

* @inproceedings{juefei-xu2017lbcnn,<br>
&nbsp;&nbsp;&nbsp;title={{Local Binary Convolutional Neural Networks}},<br>
&nbsp;&nbsp;&nbsp;author={Felix Juefei-Xu and Vishnu Naresh Boddeti and Marios Savvides},<br>
&nbsp;&nbsp;&nbsp;booktitle={IEEE Computer Vision and Pattern Recognition (CVPR)},<br>
&nbsp;&nbsp;&nbsp;month={July},<br>
&nbsp;&nbsp;&nbsp;year={2017}<br>
}

