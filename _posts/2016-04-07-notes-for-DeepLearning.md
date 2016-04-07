---
layout: post
title: Notes for Deep Learning
tags: DeepLearning MachineLearning Udacity
---


## notes for [Deep Learning at Udacity](https://www.udacity.com/course/deep-learning--ud730)

### 1. Machine Learning to Deep Learning

Logistic classifier: $wx + b = y$, w is weight, x is input, b is bias.

Softmax: $S(y_i) = \frac{e^{y_i}}{\sum_j e^{y_j}}$ turn scores into probabilities.

* If multiply the scores by 10, the probabilities get close to either 0 or 1.
* If divided by 10, the probabilities get close to uniform.

Cross-entropy: $D(S,L) = -\sum_i L_i log(S_i)$, S is socre, L is lable.

Multinomial logistic classification:linear model $wx+b$ -> logit score -> softmax S(y) -> Cross-entropy D(S,L) -> 1-hot labels.

Loss = average cross-entropy: $\frac{1}{N} \sum_i D(S(wx_i +b), L_i)$.

Minimize traning loss using gradient descent method.

Input is typically normalized, weights are typically initilized with random values from Gaussian distribution.

Training, validation, test datasets.

Rule of thumb for validation set size: A change that affects 30 examples in your validation set, one way or another, is usually statistically significant and typically can be trusted.

Use stochastic gradient descent to make the optimization process faster, which is the core of deep learning.

Two methods to help SGD faster:

* Momentum: for SGD, each step we take a very small step in a random direction, but on aggregate, those steps take us toward the minimum of the loss. We can take advantage of the knowledge that we've accumulated from previous steps about where we should be headed. A cheap way to do that is to keep a running average of the gradients, and to use that running average instead of the direction of the current batch of the data. M <- 0.9M + $\Delta \alpha$.
* Learning rate decay: when replacing gradient descent with SGD, we are going to take smaller, noiser steps towards our objective. It is hard to decide how smaller that step should be. However, it is beneficial to make that step smaller and smaller as you train.

Learning rate tuning: plot the loss and step figures. Never trust how quickly you learn, it has often little to do with how well you train.

SGD hyper-parameters: initial learning rate, learning rate decay, momentum, batch size, weight initilization.

When things don't work, always try to lower the learning rate first.

AdaGrad is a modification of SGD which implicitly does momentum and learning rate decay for you. Using AdaGrad often makes learning less sensitive to hyper-parameters. But it often turns to be a litte worse than precisely tuned SDG with momentum.

### 2. Deep Neural Networks

Backward propagation - use chain rule.

You can typically get much more performance with fewer parameters by going deeper rather than wider. A lot of natural phenomena tend to have a hierarchical structure which deep models naturally capture.

Prevent overfitting:

* Early termination: look at the performance under validation set, and stop training as soon as no further improvement.
* Regularization: apply artifical constraints that implicitly reduce the number of free parameters while not making it more difficult to optimize, such as L2 regulization.
* Dropout: Imagine that you have one layer that connects to another layer. The values that go from one layer to the next are often called activiations. Take those activations and randomly for every example you train your network on, set half of them to zero. Basically, you randomly take half of your data through your network and just destroy it, and then randomly again. Learn from reduntant representations. More robust, and prevent overfitting.

### 3. Convolutional Neural Networks

Translation invariance: different places, but same objective.

Weight sharing: when you know that two inputs can contain the same kind of information, then you share the weights and train the weights jointly for those inputs.

CNN is network that share parameters across space.

Patches are sometimes called kernels. Each pancake in stack is called a feature map. Stride is the number of pixels that you shift each time you move the filter.

* Valid padding: do not go past the edge.
* Same padding: go off the edge and pad with zeros.

Pooling: Until now, we've used striding to shift the filters by a few pixel each time and reduce the future map size. This is a very aggresive way to downsample an image. It removes a lot of information. What if instead of skipping one in every two convolutions, we still ran with a very small stride, but then took all the convolutions in a neighborhood and combined them somehow. This is called pooling. The most common one is max pooling. At every point in the future map,look at a small neighborhood around that point and compute the maximum of all the responses around it. Max pooling doesn't add number of parameters, and doesn't risk an increasing overfitting. It simply often yields more accurate models. However, since the convolutions that run below run at a lower stride, the model becomes a lot more expensive to compute, and there are some paramters to set, such as pooling size and stride. Another notable form of pooling is average pooling. Instead of taking the max, just take an average over the window of pixels around a specific location.

1x1 convolutions: The classic convolution setting is  basically a small classifier for a patch of the image, and it's only a linear classifier. But if you add a 1x1 convolution in the middle, then you have a mini neural network running over the patch instead of a linear classifier. This is a very inexpensive way to make your models deeper and have more parameters without completely changing their structure. Actually, they are just matrix multiplies which have relative few parameters and computationaly cheap.

Inception module: Instead of having a single convolution, we put the pooling and 1x1 convolution or nxn convolution together. We can choose parameters in such a way that the total number of parameters in model in very small. Yet the model performs better.

### 4. Deep Models for Text and Sequences

Embeddings: put similar words into a smaller set.

t-SNE: a way of projections that preserves the neighborhood structure of data.

Distance: because of the way embeddings are trained, it's offten better to measure the closeness using a cosine distance instead of L2 and normalized.

Sampled softmax: Instead of treating the softmax as if the label had probability of 1, and every other word has probability of 0, we can sample the words that are not the targets, pick only a handful of them and act as if the other words were not there. It makes things faster at no cost in performance.

Semantic analogy can make word vectors do the math.

Simiar as Covolutional Neural Network (CNN) uses shared parameters across space to extract patterns over an image, Recurrent Neural Network (RNN) (handle word sequence) does the same thing but over time instead of space.

Imagine that you have a sequence of events, at each point in time you want to make a decision about what's happened so far in the sequence. If your sequence is reasonably stationary, you can use the same classifier
at each point in time. That simplifies things a lot already. But since this is a sequence, you also want to take into account the past. One natural thing to do here is to use the state of the previous classifier as a summary of what happened before, recursively.

Backpropagation for RNN will apply all updates to the same parameters (correlated updates), which is very unstable and bad for SGD. The gradients either go exponentially to infinity or zeros very quickly, which is called exploding and vanishing gradient problem, respectively.

Exploding gradients: use gradient clipping (shrink step when gradient norm goes too big) to handle the gradient bounding problem. $\Delta w = \Delta w \frac{\Delta_{max}}{max(\vert \Delta w \vert, \Delta_{max})}$

Vanishing gradients makes model only remeber recent events, which is "memory-loss". This is where Long Short Term Memeory (LSTM) comes to help.

![LSTM](https://c2.staticflickr.com/2/1668/26262894915_8de10a0362_z.jpg)

![LSTM2](https://c2.staticflickr.com/2/1486/25990143400_0774e4c4f2_c.jpg)

LSTM regulization: L2 and Dropout

RNN can be used to generate sequences.

Imagine that you have a model that predicts the next step of your sequence. You can use that to generate sequences. You sample from the predicted distribution and then pick one element based on its probability, and then feed to the next step and go on. Do this repeatedly, you can generate a pretty good sequence. A more sophisticated way is instead of sampling once at each step, sample multiple times. Choose the best sequence out of those by computing the total probability of all the characters that you generated so far. This can prevent the sampling from accidentally making one by choice, and being stuck with the one bad decision forever. A smarter way to do this is called Beam Search, which only keep the mostly likely few candidate sequences at every time step, and simply prune the rest. This works very well in practice.





















