---
layout: post
title: Notes for Introduction to Machine Learning
tags: machine learning udacity
---

## Notes for [Introduction for Machine Learning](https://www.udacity.com/course/intro-to-machine-learning--ud120)

### 1. [Naive Bayes](http://scikit-learn.org/stable/modules/naive_bayes.html)

$
P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots x_n \mid y)}
                                 {P(x_1, \dots, x_n)}
$
Using the naive independence assumption that
$ P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y)$,
for all i, this relationship is simplified to
$P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                 {P(x_1, \dots, x_n)}$

Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality.
On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB().fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred, labels_test)
```

### 2. [Support Vector Machine](http://scikit-learn.org/stable/modules/svm.html)

The advantages of support vector machines are:

* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

* If the number of features is much greater than the number of samples, the method is likely to give poor performances.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation.

SVM parameters:

* Kernel: linear, rbf, or others.
* Gamma parameter: defines how much influence a single traning example has. The larger the `gamma` is, the closer other examples must be to be affected.
* C parameter: controls tradeoff between smooth decision boundary and classifying training points correctly. A low `C` makes the decision surface smooth, while a high `C` aims at classifying all training examples correctly.

Naive Bayes is great for text--it’s faster and generally gives better performance than an SVM for this particular problem.

Tuning the parameters can be a lot of work, but just sit tight for now--toward the end of the class we will introduce you to GridCV, a great sklearn tool that can find an optimal parameter tune almost automatically.

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

clf = SVC(kernel="rbf", C=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
```

### 3. [Decision Trees](http://scikit-learn.org/stable/modules/tree.html#)

Entropy: a measure of impurity in a bunch of exmaples, which controls how a Decision Tree decides where to split the data.

Entropy = $- \sum_i (p_i)log_2^{p_i}$

Information gain = entropy(parent) - [weighted average] entropy(children)

Decision tree algorithm : maximize information gain

```python
from sklearn import tree
from sklearn.metrics import accuracy_score

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
```

### 4. Choose your own algorithm

* [K nearest neighbors](http://scikit-learn.org/stable/modules/neighbors.html)
* [Random forest](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [Adaboost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)

### 5. [Regressions](http://scikit-learn.org/stable/modules/linear_model.html)

SSE (sum of squared error) = $\sum (real-predicted)^2$.

R-squared of a regression: "how much of my change in the output is explained by the change in my input?"

```python
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(features_train, target_train)
reg.predict(features_test)
```

### 6. Outliers

Outlier rejection:

* Train
* Remove points with large residual error (~10%)
* Re-train until no large change

### 7. [Clustering](http://scikit-learn.org/stable/modules/clustering.html)

[K-means](http://scikit-learn.org/stable/modules/clustering.html#k-means)

```python
from sklearn.cluster import KMeans
pred = KMeans(n_clusters=2)
pred = pred.fit_predict(finance_features)
```

### 8. [Feature Scaling](http://scikit-learn.org/stable/modules/preprocessing.html)

x_new = (x-x_min) / (x_max - x_min) rearrange the original features to [0, 1].

```python
# with scaling
from sklearn import preprocessing

new_feature = preprocessing.MinMaxScaler().fit(finance_features)
# do the KMeans with new_feature
```

### 9. [Text Learning](http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

Bag of words: count frequency of each word.

* Word order doesn't matter
* Long phrases give different input vectors
* Cannot handle complex phrases, like "Chicago Bulls"

Stopwords: low information, high frequency, like "the", "in", etc.

[NLTK](http://www.nltk.org/): get stop words

```python
from nltk.corpus import stopwords
sw = stopwords.words("english")
```

STEMMER: tranfer similar words into one representative.

```python
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
stemmer.stem("responsiveness")  # give us "respons"
```

TF-IDF: term frequency-inverse document frequency

* TF is like the bag of words
* IDF is the weighting by how often word occurs in corpus

```python
from sklearn.feature_extraction.text import TfidfVectorizer
# vectorize wors with TDIDF way and get rid of stopwords
vectorizer = TfidfVectorizer(stop_words="english")  
tfidf = vectorizer.fit_transform(word_data)
# get names of the vectorized words
vectorizer.get_feature_names()
```

### 10. [Feature Selection](http://scikit-learn.org/stable/modules/feature_selection.html)

Add new features:

* use intuition
* code up the new feature
* visualize results
* repeat a few times and get the best new feature

Remove features:

* Too nosiy
* Cause overfitting
* Strongly related with a feature already present
* Slow down training/testing process

Features $$\ne$$ Information

There are two big univariate feature selection tools in sklearn: SelectPercentile and SelectKBest. The difference is pretty apparent by the names: SelectPercentile selects the X% of features that are most powerful (where X is a parameter) and SelectKBest selects the K features that are most powerful (where K is a parameter).

Bias-variance dilemma:

* high bias: oversimpilfied, high error on training set
* high variance: overfitting, higher error on testing set than training set

Regulation in Regression

* [Lasso](http://scikit-learn.org/stable/modules/linear_model.html#lasso): L1 norm regulation, can set coefficitents to zeros.
* [Ridge](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression): L2 norm regulation, cannot set coefficients to zeros.


### 11. [PCA](http://scikit-learn.org/stable/modules/decomposition.html)

Principal component of a dataset is the direction that has the largest variance because it retains maximum amount of "information" in the original data.

Projection onto direction of maximal variance to minimize information loss.

* systemized way to transform input features into principal components
* use principal components as new features
* PCs are directions in data that maximize variance when you project data onto them
* more variance of data along a PC, higher that PC is ranked
* most variance/most information -> first PC

Select a number of principal components: train on different number of PCs, and see how accuracy responds - cut off when it becomes apparent that adding more PCs doesn't buy you much more discrimination.

When to use PCA:

* latent features driving the patterns in data
* dimensionality reduction: reduce size, visualize high-dimensional data

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
print pca.explained_variance_ratio_
first_pc = pca.components_[0]
sencond_pc = pca.components_[1]

# project data onto new coordinates
transformed_data = pca.transform(data)
for ii, jj in zip(transformed_data, data):
	plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color="r")
	plt.scatter(second_pc[0]*ii[1], second_pc[1]*ii[1], color="c")
	plt.scatter(jj[0], jj[1], color="b") 
```

### 12. [Validation](http://scikit-learn.org/stable/modules/cross_validation.html)

Train test data splitting

```python
from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_spli(data, target, test_size=0.4, random_state=0)
```

Cross validaiton

```python
from sklearn.cross_validaiton import KFold
kf = KFold(len(data), 2) # lenght of record, and number of folds
for tain_indices, test_indices in kf:
	features_train = data[train_indices]
	features_test = data[test_indices]
	# do you model here
```

GridSearchCV is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance.

```python
# Four candidates:[('rbf', 1), ('rbf', 10), ('linear', 1), ('linear', 10)]
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)
print clf.best_params_
```

### 13. Evaluation metrics

accuracy = no. of items labeled correctly / no. of total items

confusion matrix

|                 | Predicted negative | Predicted positive |
| Actual negative |        a           |         b          |
| Actual positive |        c           |         d          |

Accuracy: (a+d)/(a+b+c+d)

Recall: true positive / (false negative + true positive) = d/(c+d)

Precision: true positiove / (ture positive + false positive) = d/(b+d) 

```python
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
```

### 14. Summary

Data/Question -> Features -> Algorithms -> Evaluation -> Data/Question
