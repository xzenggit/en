---
layout: post
title: Notes for CNN for Visual Recongnition
tags: DeepLearning
---


## notes for [CNN for Visual Recongnition](http://cs231n.stanford.edu/syllabus.html)

* [Course Notes](http://cs231n.github.io)

### Loss function

 $L = \frac{1}{N} \sum_{i=1}^N \sum_{j \neq y_i}$ max $(0, f_j(x_i, W) -f_{y_j}(x_i, W)+1)$+$\lambda R(W)$

In commom use:

* L2 regularization $R(W) = \sum_k \sum_l W_{k,l}^2$
* L1 regularization $R(W) = \sum_k \sum_l \vert W_{k,l} \vert$
* Elastic net (L1+L2) $R(W) = \sum_k \sum_l \beta_1 W_{k,l}^2 + \beta_2 \vert W_{k,l} \vert$
* Dropout: randomly set some neurons to zeros in the forward pass; force the network to have a redundant representation; need to scale at test time.

Loss function type:

* Softmax $L_i = -log(\frac{e^{s_{y_j}}}{\sum_j e^{s_j}})$
* SVM (hinge loss) $L_i = \sum_{j \neq y_j} max(0, s_j - s_{y_i} + 1)$

In practice: always use analytic gradient, but check implementation with numerical gradient. This is called gradient check.

### Activation functions

* Sigmoid $\sigma (x) = \frac{1}{(1+e^{-x})}$
    - saturated neurons "kill" the gradients
    - Sigmoid outputs are not zero-centered
    - exp() is a bit compute expensive
* tanh(x)
    - squashes numbers to range [-1, 1]
    - zero centered
    - still kills gradients when saturated
* ReLU ( Rectified Linear Unit) $f(x) = max(0, x)$
    - does not saturate
    - very computationally efficient
    - converges much faster than sigmoid/tanh in practice
    - not zero-centered output
* Leaky ReLU $f(x) = max(\alpha x, x)$
    - does not saturate
    - computationally efficient
    - converges much faster than sigmoid/tanh in practice
    - will not "die"
* Exponential Linear Units (ELU) $f(x) = x $ if $x > 0$; $f(x) = \alpha (exp(x) -1)$ if $x \le 0$
    - all benefits of ReLU
    - does not die
    - closer to zero mean outputs
    - computation requires exp()
* Maxout "neuron" $max(w_1^T x + b_1, w_2^Tx+b_2)$
    - generalize ReLu and Leaky ReLU
    - linear regime, does not saturate, does not die
    - doubles the number of parameters

In practice

* Use ReLU. Be careful with learning rates
* Try out Leaky ReLU/ Maxout/ ELU
* Try out tanh but do not expect much
* Do not use sigmoid

### Weight initilization

* Small random numbers: works okay for small networks, but can lead non-homogeneous distributions of activations across the layers of a network
* [Xavier initilization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization) 

Batch normalization:

* Improves gradient flow through the network
* Allow higher learning rates
* Reduces the strong dependence on initialization
* Acts as a form of regularization

Hyperparameters to play with:

* network architecture
* learning rate, its decay schedule, update type
* regularization 

### Step update

* Gradient descent `x += -learning_rate * dx`
* Momentum update 
    - `v = mu * v - learning_rate * dx # integrate velocity` 
    - `x += v # integrate position`
    - physical interpretation as ball rolling down the loss function + friction (mu coefficient)
    - mu = usually ~0.5, 0.9, or 0.99 (or annealed over time)
* Nesterov momentum update
    - $ v_t = \mu v_{t-1} - \epsilon \nabla f(\theta_{t-1} + \mu \v_{t-1}) $
    - $ \theta _t = \theta_{t-1} + v_t$
    - use variable tansform
        + $ v_t = \mu v_{t-1} - \epsilon \nabla f(\phi)$
        + $\phi_t = \phi_{t-1} - \mu v_{t-1} + (1+\mu) v_t$
    - code
        + `v_pre = v`
        + `v = mu * v -learning_rate * dx`
        + `x += -mu * v_prev + (1 + mu) * v`
* AdaGrad update
    - `cache += dx**2`
    - `x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)`
    - added element-wise scaling of the gradient based on the historical sum of squares in each dimension
* RMSProp update
    - `cache = decay_rate * cache + (1 - decay_rate) * dx**2`
    - `x += - learning_rate * dx / (np.sqrt(cache) + 1e-7)`
* Adam update
    - `m = beta1 * m + (1-beta1) * dx # update first moment`
    - `v = beta2 * v + (1-beta2) * (dx**2) # update second moment`
    - `mb = m/(1-beta1**t) # correct bias`
    - `vb = v/(1-beta2**t) # correct bias`
    - `x += -learning_rate * mb / (np.sqrt(vb) + 1e-7`

Learning rate decay over time:

* step decay: e.g. decay learning rate by half every few epochs
* exponential decay: $\alpha = \alpha_0 e^{-kt}$
* 1/t decay: $\alpha = \alpha_0 / (1+kt)$

Second order optimization methods:

* second order Taylor: $J(\theta) = J(\theta_0) + (\theta-\theta_0)^T \nabla _{\theta}J(\theta_0) + \frac{1}{2} (\theta - \theta_0) ^T H (\theta - \theta_0)$
* solve for critical point: $\hat{\theta} = \theta_0 - H^{-1} \nabla_{\theta} J(\theta_0)$
* Quasi-Newton methods (e.g. BGFS): instead of inverting the Hessian, approximate inverse Hessian with rank 1 updates over time
* L-BFGS (Limited memory BFGS): does not form/store the full inverse Hessian
    - usually works very well in full batch, deterministic code
    - does not transfer very well to mini-batch setting

In practice:

* Adam is a good default choice in most cases
* Try L-BFGS if can afford to do full batch updates

### Convolutional Neural Network (CNN)

ConvNet is a sequence of Convolution Layers, interspersed with activation functions.

$f(x,y)*g(x,y) = \sum_{n_1 = -\infty}^{\infty} \sum_{n2=-\infty}^{\infty} f(n_1, n_2) g(x-n_1, y-n_2)$

The Conv Layer:

* Accepts a volume of size $W_1 \times H_1 \times D_1$
* Requires four hyperparameters:
    - Number of filters $K$,
    - their spatial extent $F$,
    - the stride $S$,
    - the amount of zero padding $P$.
* Produces a volume of size $W_2 \times H_2 \times D_2$ where:
    - $W_2 = (W_1 - F + 2P)/S + 1$
    - $H_2 = (H_1 - F + 2P)/S + 1$ (i.e. width and height are computed equally by symmetry)
    - $D_2 = K$
* With parameter sharing, it introduces $F \cdot F \cdot D_1$ weights per filter, for a total of $(F \cdot F \cdot D_1) \cdot K$ weights and $K$ biases.
* In the output volume, the $d$-th depth slice (of size $W_2 \times H_2$) is the result of performing a valid convolution of the $d$-th filter over the input volume with a stride of $S$, and then offset by $d$-th bias.
* Common settings: $F=3$, $S=1$, $P=1$.

Pooling layer:

* make the representations smaller and more manageable
* operates over each activation map independently

Generally, the pooling layer:

* Accepts a volume of size $W_1 \times H_1 \times D_1$
* Requires three hyperparameters:
    - their spatial extent $F$,
    - the stride $S$,
* Produces a volume of size $W_2 \times H_2 \times D_2$ where:
    - $W_2 = (W_1 - F)/S + 1$
    - $H_2 = (H_1 - F)/S + 1$
    - $D_2 = D_1$
* Introduces zero parameters since it computes a fixed function of the input
* Note that it is not common to use zero-padding for Pooling layers
* Common setting: $F = 2, S=2$; $F=3, S=2$.

How to stack convolutions:

* Replace large convolutions (5x5, 7x7) with stacks of 3x3 convolutions
* 1x1 "bottleneck" convolutions are very efficient
* Can factor NxN convolutions into 1xN and Nx1
* All of the above give fewer parameters, less compute, and more nonlinearity

Convolution Theorem: The convolution of f and g is equal to the elementwise product of their Fourier Transforms: $F(f * g) = F(f)F(g)$. Using the FFT, we can compute the DFT of an N-dimension vector in O(NlogN) time.

Implement convolutions FFT:

* Compute FFT of weights: F(W)
* Compute FFT of image: F(X)
* Compute elementwise product: F(W)F(X)
* Compute inverse FFT: $Y=F^{-1}(F(W)F(X))$

Segmentation:

* Semantic segmentation
    - classify all pixels
    - fully convolutional models, downsample then upsample
    - learnable upsampling: fractionally strided convolution
    - skip connections can help
* Instance segmentation
    - detect instance, generate mask
    - similar pipelines to object detection

Attention:

* Soft attention:
    - easy to implement: produce distribution over input locations, reweight features and feed as input
    - attend to arbitrary input locations using spatial transformer networks
* Hard attention:
    - attend to a single input location
    - cannot use gradient descent
    - need reinforcement learning

Unsupervised learning:

* Autoencoders
    - Traditional: feature learning, reconstruct input, not used much anymore
    - Variational: generate samples, Bayesian meets deep learning
* Generative adversarial networks: generate samples

### Recurrent Neural Networks (RNN)

$ h_t = f_w (h_{t-1}, x_t)$, where $h_t$ is new state, $h_{t-1}$ is old state, $f_w$ is some function with parameters $w$, $x_t$ is input vector at some time step.

![Vanilla RNN](https://c2.staticflickr.com/2/1441/26135570480_1ee2dc68b4.jpg)


### [Software and Packages](http://cs231n.stanford.edu/slides/winter1516_lecture12.pdf)

![Caffe1](https://c2.staticflickr.com/2/1526/26343021171_729f8117d3_z.jpg)

![Caffe2](https://c2.staticflickr.com/2/1483/26343021151_df555451e6_z.jpg)

![Torch](https://c2.staticflickr.com/2/1482/26383282716_8423f19a0b_z.jpg)

![Torch](https://c2.staticflickr.com/2/1525/25806440953_7b129c3b8e_z.jpg)

![Theano](https://c2.staticflickr.com/2/1697/26343021141_c13bbcf647_z.jpg)

![Tensorflow](https://c2.staticflickr.com/2/1565/25806440993_0e74738754_z.jpg)

![OverView](https://c2.staticflickr.com/2/1630/26409316415_696edfaedd_z.jpg)

![Recommendation](https://c2.staticflickr.com/2/1606/26316982582_e7eb64122d_z.jpg)








