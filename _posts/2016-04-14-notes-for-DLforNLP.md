---
layout: post
title: Notes for Deep Learning for NLP
tags: DeepLearning NLP
---

## Notes for [Deep Learning for NLP](http://cs224d.stanford.edu/syllabus.html)


How to represent meaning in a computer?

* Common answer: use a taxonomy like WordNet that has hypernyms (is-a) relationships and synonym
    - missing nuances, new words
    - subjective
    - require human labor to create and adpat
    - hard to compute accurate word similarity
    - "one-hot" representation
* Distributional similarity based represntations
    - word-document cooccurrence matrix will give general topics
    - window around each word captures both syntactic and semantic information

Problems with simple cooccurrence vectors

* Increase in size with vocabulary
* Very high dimension: require large storage
* Subsequent classification models have sparsity issue
* Models are less robust

Solution: store most of the important information in a fixed, small number of dimensions with the help of dimensionality reduction methods (e.g. apply PCA to cooccurrence matrix)

Problem with PCA:

* Computational cost scales quadratically (bad for millions of words and documents)
* Hard to incorporate new words or documents
* Different learning regime than other DL models

Idea: directly learn low-dimensional word vectors

* Learning representations by back-propagating errors
* Neural probabilistic language models
* [Word2Vec](https://code.google.com/archive/p/word2vec/)

Main idea of Word2Vec:

* Instead of capturing cooccurence counts directly
* Predict surrounding workds of every word
* Faster and can easily incorporate new sentences and documents

Details of Word2Vec:

* Predict surrounding words in a windwo of length m of every word
* Objective function: maximize the log probability of any context word given the current center word 
* $J(\theta) = \frac{1}{T} \sum_{t=1}^T \sum_{-m \le j \le m, j \ne 0} log p(w_{t+j} \vert w_t)$, $\theta$ represents all variables we optimize
* $p(o \vert c) = \frac{exp(u_o^T v_c)}{\sum_{w=1}^W exp(u_w^Tv_c)}$, where o is the outside word id, c is the center word id, u and v and "center" and "outside" vectors of o and c.

Count based vs direct prediction

* Count based: LSA, PCA
    - fast training
    - efficient usage of statistics
    - primarily used to capture word similarity
    - disproportionate importance given to large counts
* Direct prediction: RNN
    - scales with corpus size
    - inefficient usage of statistics
    - generate improved performance on other tasks
    - capture complex patterns beyond word similarity

GloVe: combine them together

* Fast training
* Scalable to huge corpora
* Good performance even with small corpus and vectors
* $J(\theta) =\frac{1}{2} \sum_{i,j=1}^W f(P_{ij})(u_i^T v_j - log P_{ij})^2$

Skip-gram model: train binary logistic regressions for a true pair (center word and word in its context windwo) and a couple of random pairs (the center word with a random word)

Continous bag of words model: predict center word from sum of surrounding word vectors instead of predicting surrounding single words from center word as in skipgram model

* If only have a small training dataset, do not train the word vectors.
* If have a very large dataset, it may work better to train word vectors.

General strategy for successful Neural Nets

* Select network structure appropriate for problem: sigmoid, tanh, ReLu
* Check for implementation bugs with gradient checks
* Parameter initialization
    - Initialize hidden layer biases to 0 and output biases to optimal value if weights were 0
    - Initialize weights ~uniform(-r, r), r inversely proportional to fan-in (previous layer size) and fan-out(next layer size) $\sqrt{6/(fan-in + fan-out)}$
* Optimization: 
    - SGD usually wins over all batch methods on large datasets
    - L-BFGS or Conjugate Gradients win on smaller datasets
* Learning rates
    - Simplest recipe: keep it fixed and use the same for all parameters
    - Collobert scales them by the inverse of square root of the fan-in of each neuron
    - Better results can generally be obtained by allowing learning rates to decrease in O(1/t)
    - Better yet: no hand-set learning rates by using L-BFGS or AdaGrad
* Prevent overfitting
    - simple first step: reduce model size by lowering number of units and layers and other parameters
    - standard L1 or L2 regularization on weights
    - early stopping: use parameter that gave best validation error
    - sparsity constraints on hidden activations
    - dropout

Adagrad:

* Standard SGD, fixed $\alpha$: $\theta_{new} = \theta_{old} - \alpha \bigtriangledown _{\theta} J_{t}(\theta)$
* Instead: adpative learning rates, learning rate is adapting differently for each parameter and parameters get larger updates than frequently occurring parameters. 
* $g_{t,i} = \frac{\partial}{\partial \theta_{t,i}}J_t(\theta)$
* $\theta_{t,i} = \theta_{t-1, i} - \frac{\alpha}{\sqrt{\sum_{\tau=1}^t}}g_{t,i}

Deep learning tricks of the trade (Y. Bengio, 2012): 

* unsupervised pre-training
* SGD and setting learning rates
* main hyper-parameters
    - learning rate schedule and early stopping
    - minibatches
    - parameter initialziation
    - number of hidden units
    - L1 or L2 weight decay
    - sparsity regularization
* how to efficiently search for hyper-parameter configuration:
    - random hyperparameter search

A language model computes a probability for a sequence of words. Probability is usually conditioned on window of n previous words.

RNN: condition the neural network on all previous words and tie the weights at each time step

RNN language model: use the same set of W eights at all time step.

* $h_t = \sigma (W^{(hh)}h_{t-1} + W^{(hx)}x_{t})$
* $\hat{y}_t = softmax(W^{(S)}h_t)$
* $\hat{P}(x_{t+1}=v_j \vert x_t, \ldots, x_1) = \hat{y}_{t,j}$
* $J^{(t)}(\theta) = - \sum_{j=1}^{\vert V \vert} y_{t,j} log \hat{y}_{t,j}$
* $ J = \frac{1}{T} \sum_{t=1}^T J^{(t)}$

Vanishing or exploding gradient problem for RNN: the gradient is a product of Jacobian matrices, each associated with a step in the forward computation. This can become very small or very large qucikly, and the locality assumption of gradient descent breaks down. 

The solution first introduce by Mikolov is to clip gradients to a maximum value. Pesudo code for norm clipping:

* $g <- \frac{\partial \epsilon}{\partial \theta}
* if $\Vert g \Vert \ge$ threshold then
    - $ g <- \frac{threshold}{\Vert g \Vert} g$
* end if

For vanishing gradients: initialization + ReLU

* Initialize W to indentity matrix and $f(z) = rect(z) = max(z, 0)$


Bidirectional RNN: for classification we want to incorporate information from words both preceding and following

Semantic Vector Spaces: vectors representing phrases and setences that do not ignore word order and capture semantics for NLP tasks.

* Standard Recursive Neural Network: paraphrase detection
* Matrix-vector Recursive Neural Network: relation classification
* Recursive Neural Tensor Network: sentiment analysis
* Tree LSTM: phrase similarity

* Recursive nerual nets require a parse to get tree structure
* Recurrent neural nets cannot capture phrases without prefix context and often capture too much of last words in final vector
* CNN: compute vectors for every possible phrases

Model comparison:

* Bag of Vectors: surprisingly good baseline for simple classification problems. Especially if followed by a few layers.
* Window Model: good for single word classification for problems that do not need wide context
* CNNs: good for classification, unclear how to incorporate phrase level annotation (can only take a single label), need zero padding for shorter phrases, hard to interpret, easy to parallelize on GPUs
* Recursive Neural Networks: most linguistically plausible, interpretable, provide most important phrases, need parse trees
* Recurrent Neural Networks: most cognitively plausible (reading from left to right), not usually the highest classification performance but lots of improvements right now with gates.











