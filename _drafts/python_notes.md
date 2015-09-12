---
layout: post
title: Python notes
tags: python notes
---

This is my personal python notes.

There are many many many ... python tutorials on the internet. I found some of them are pretty useful.

* [Google python course (basic)](https://developers.google.com/edu/python/)

* [Codecademy python course (basic)](www.codecademy.com/)

* [Dataquest data science course](https://www.dataquest.io/)

* [Quantitative Economics Python](http://quant-econ.net/py/index.html)

* [Introduction to Python for Econometrics, Statistics and Data Analysis](https://www.kevinsheppard.com/images/0/09/Python_introduction.pdf)

* [Python Scientific Lecture Notes](http://scipy-lectures.github.io/)



# Notes from [Python for data analysis](https://github.com/pydata/pydata-book/blob/master/ch02.ipynb)

## Ch2. Introductory Examples

Use inline function example: `time_zones = [rec['tz'] for rec in records if 'tz' in rec]`

High-performance container datatypes: `Counter` and `defaultdict`. See [here](https://docs.python.org/2/library/collections.html).

Common functions: 
`most_common([n])` return a list of the n most common elements and their counts.
`value_counts` return object containing counts of unique values.
`notnull()` return not null index.
`dropna()` return index without na.
`fillna(n)` fill na with n.
`where` see [here](http://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html).
`argsort` return indices that would sort an array. See [here](http://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html).
`unstack` pivot a level of index labels, returning a DataFrame having a new level of column lables whose inner-most level consists of the pivoted index labels. See [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html).
`pandas.pivot_table` creates a spreadsheet-style pivot table as a DataFrame. See [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.pivot_table.html).
`dataframe.ix` is a primarily label-location based indexer, with integer position fallback. See [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.ix.html).
`dataframe.rename` alter aexes input function.
`dataframe.sort_index` sort dataframe either by labels along either axis or by the values in a colum. See [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_index.html).
`groupby`, `tail`, `head` same meaning as function name.
`pd.read_csv` read csv file.
`numpy.allclose` returns True if two arrays are element-wise equal within a tolerance. See [here](http://docs.scipy.org/doc/numpy/reference/generated/numpy.allclose.html).
`dataframe.div` floating division of dataframe and other, element-wise. See [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.div.html).
`argmax` retunrs the first index of the maximum value.

Plot figures:
`matplotlib.figure.Figure(figsize=(width,height), dpi=None, facecolor=Noe)`. See [here](http://matplotlib.org/api/figure_api.html) for details.
`subplots_adjust` adjust subplots size and white spaces.

## Ch3. Ipython
`function?` shows us the docstring.
`function??` shows us the source code if possible.
`numpy.*load*?` shows us possible functions.
`%run *.py` run the source code `*.py`
`%paste` takes whatever text is in the clipboard and executes it as a single block in the shell.
`%cpaste` is similar, except that it gives you a special prompt for pasting. In this way, you can paste as much code as you want before executing it.
Some shortcuts for ipython can be found [here](http://johnlaudun.org/20131228-ipython-notebook-keyboard-shortcuts/).
[ipython built-in magic commands](https://ipython.org/ipython-doc/dev/interactive/magics.html)
The typical way to launch IPython with matplotlib integration is `ipython --pylab`.

## Ch4. NumPy Basics: Arrays and Vectorized Computation

ndarray is a fast, flexible container for large datasets in Python.
`data.shape`, `data.dtype`
`np.zeros((n,m))` gives a nxm zero matrix.
`np.arange(n)` gives array [0,1,2,3,...,n-1].
Arracy creation functions: `array`,`asarray`,`arange`,`ones`,`ones_like`,`zeros`,`zeros_like`,`empty`,`empty_like`,`eye`,`identity`.
Example: `arr1=np.array([1,2,3],dtype=np.float(64)`
Array scling: remeber that Python array index starts from 0, and does not include the `n` is your use `m:n` format. See [here](https://www.safaribooksonline.com/library/view/python-for-data/9781449323592/ch04.html) for example.
Fancy indexing: you can use a list to do the fancy indexing, such as `arr[[4,1,2]]` will give you the row of 4,1,2, and make them a new array. Or you can use `np.ix_` function to convert two 1D integer arrays to an indexer that selects the square region `arr[np.ix_([1,2,3],[2,3,1])]`.
`np.arange(15).reshape((3,5))` get a 3x5 matrix.
`np.array([[1,2,3],[2,3,1]])` get a 2x3 matrix.
[Universal functions](http://docs.scipy.org/doc/numpy/reference/ufuncs.html).
Array set operations: `unique(x)`, `intersect1d(x,y)`, `union1d(x,y)`, `setdiff1d(x,y)`, `setxor1d(x,y)`.
`np.save` and `np.load` are two workhorse functions for efficiently saving and loading array data on disk.
Commonly used numpy.linalg functions: `diag`, `dot`, `trace`, `det`, `eig`, `inv`, `pinv`, `qr`, `svd`, `slove`(for Ax=b), `lstsq` (for least-square solution to y=Xb).
Partial lsit of numpy.random functions: `seed`, `permutation`, `shuffle`, `rand`, `randint`, `randn`, `binomial`, `normal`, `beta`, `chisquare`, `gamma`, `uniform`.

See the ch04 ipython notebook for [examples](https://github.com/pydata/pydata-book/blob/master/ch04.ipynb).

## Ch.5 Get started with pandas
Series and DataFrame are two important data structures in pandas.
Series is a one-dimensional array-like object containing an array of data and an associated array of data labels, called index. Ex: `Series([4, 2, 3, 1], index = ['a', 'b','c','d'])`.

DataFrame represents a tabular, spreadsheet-like data structurer containing an ordered collection of columns, each of which can be a different value type (numeric, string, boolean, etc.).
`frame.reindex` can reindex the frame.
`apply` can applying a function on 1D arrays to each column (default) or row (axis =1 ).

[Descriptive functions in pandas](http://pandas.pydata.org/pandas-docs/stable/basics.html#descriptive-statistics)

[ipython notebook examples](https://github.com/pydata/pydata-book/blob/master/ch05.ipynb).




# Other notes

`==` is for value equality. Use it when you would like to know if two objects have the same value.
`is` is for reference equality. use it when you would like to know if two references refer to the same object.

`.item()` method can be used to access a dictionary key and value in a loop. For example,
```
animal_types = {"robin": "bird", "pug": "dog", "osprey": "bird"}

# The .items method lets us access a dictionary key and value in a loop.
for key,value in animal_types.items():
print(key)
print(value)
# This is equal to the value
print(animal_types[key])
```

`.iloc` method in `pandas` is similar as indexing method in `numpy`


