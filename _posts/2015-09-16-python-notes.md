---
layout: post
title: Python notes
tags: python notes
---

This is my personal python notes.

There are many many many ... python tutorials on the internet. I found some of them are pretty useful.

1. [Google python course (basic)](https://developers.google.com/edu/python/)
2. [Codecademy python course (basic)](www.codecademy.com/)
3. [Dataquest data science course](https://www.dataquest.io/)
4. [Quantitative Economics Python](http://quant-econ.net/py/index.html)
5. [Introduction to Python for Econometrics, Statistics and Data Analysis](https://www.kevinsheppard.com/images/0/09/Python_introduction.pdf)
6. [Python Scientific Lecture Notes](http://scipy-lectures.github.io/)

### Notes from [Python for data analysis](https://github.com/pydata/pydata-book/blob/master/ch02.ipynb)
### Ch2. Introductory Examples
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

## Ch6. Dadta Loading, Storage, and File Formats

`read_csv` load delimited data from a file, URL, or file-like object. Use comma as default delimiter.

`read_table` load delimited data from a file, URL, or file-like object. Use tab as default delimiter.

`read_fwf` read data in fixed-width column format (that is, no delimiters)

`read_clipboard` version of `read_table` that reads data from the clipboard. Useful for converting tables from webpages.

`read_csv(filename,sep='seperate sign', names=['column names'], index_col=['index column name'], nrows='number of rows')` see [detail](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html).

`read_table(filename, sep='sep sign', names=['**'], index_col=['**'], na_values=[''])` see [detail](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_table.html)

`to_csv(filename, sep='', na_rep='', index='T or F', header='T or F',cols=['column names'])` write data out.

`date_range('1/1/2000',periods=7)` gives seven days dates.

`frame.save('filename')` save a pickle in python, `load('filename')` load back into python.

[ipython notebook examples](https://github.com/pydata/pydata-book/blob/master/ch06.ipynb)

## Ch7. Data Wrangling: clean, transform, merge, reshape.

`merge(df1,df2, how='inner')` connects rows in dataframes based on one or more keys. `how` can be `inner`, `outer`, `right`,`left`. `left_index` and `right_index` can also be used to merge acorrding to index.

`df1.join(df2, how='inner')` has the same effect as above.

Numpy has a `concatenate` function for doing this with raw Numpy arrays.

panda `concat` can do more fancy stuff. See [details](http://pandas.pydata.org/pandas-docs/stable/merging.html).

`df.pivot` transfomr data to that containing one column per disdtinct item value.

`data.duplicated()` show if there are duplicated items.

`data.drop_duplicates()` drops duplicated items.

`map` method on a Series accepts a function or dict-like object containing a mapping.

`fillna` fill in missing data in a general way.

`replace(original_value, new_value)` a generall way of repalcing data.

`data.index` can modify the dataframe index.

`data.rename` can rename the dataframe index and columns.

`data.cut` divide items into bins. See [detail](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html).

`data.qcut` cut data by sample quantiles.

`data.get_dummies` derive a matrix or dataframe containing k columns containing all 1's and 0's.

`split` is often combined with `strip` to trim whitespace. Ex: `new=[x.strip() for x in old.split(',')]`.

[Python built-in string functions](https://docs.python.org/2.6/library/string.html).

[Python regular expression method](https://docs.python.org/2/library/re.html).

[ipython notebook examples](https://github.com/pydata/pydata-book/blob/master/ch07.ipynb)

## Ch8. Plotting and Visualization

`matplotlib` is a primarily 2D plotting package like Matlab. It also has some add-on toolkits, such as `mplot3d` for 3D plots and `basemap` for mapping and projections. 

The common way to use matplotlib is through `ipython --pylab`.

Matplotlib API functions generally are in `matplotlib.pyplot` module, which typically imported by `import matplotlib.pyplot as plt`.

Create a figure object with `fig=plt.figure()`.

Add subplot by `fig.add_subplot(2,2,1)`.

Subplot options include `nrows`, `ncols`, `sharex`, `sharey` et al.

Space between subplots can also be adjusted by `subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)`.

`fig.set_xticks`, `fig.set_xticklabels`, `plt.xlim([])`, `fig.legend(loc='best')`, `fig.text`, `fig.annotate`,

fig.add_patch`, `plt.savefig(filename, dpi=400, bbox_inches='tight')`. `bbox_inches`: the portin of the figure to save. If 'tight' is passed, will attempt to trim the empty space around the figure.

To set the global defualt figure size: `plt.rc('figure', figsize=(n,m))`.

`data.plot(kind='bar', ax=axes[0], color='k', alpha=0.7)`, `ax` is matplotlib subplot object, `alpha` is the plot fill opacity (from 0 to 1).

[pandas plotting examples](http://matplotlib.org/api/pyplot_api.html)

[basemap examples](http://matplotlib.org/basemap/users/examples.html)

[ipython notebook examples](https://github.com/pydata/pydata-book/blob/master/ch08.ipynb)

## Ch9. Data Aggregation and Group Operations

All you need to know is [here](http://pandas.pydata.org/pandas-docs/stable/groupby.html) or [here](https://github.com/pydata/pydata-book/blob/master/ch09.ipynb).

## Ch10. Time Series

`to_datetime` parases many different kinds of date representations.

`strftime` method format a datetime object to strings. See [here](http://strftime.org) for formats.

`strptime` can convert strings to datetime objects.

`date_range(start=, end=, periods=, freq=, normalzie=)` see [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.date_range.html).

Time series can also be modified with offsets, such as [here](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#dateoffset-objects).

[Time zone conversion](http://pytz.sourceforge.net).

Periods represent time spans, like days, months, quarters, or years.

[pandas rolling(moving) statistics](http://pandas.pydata.org/pandas-docs/stable/computation.html#moving-rolling-statistics-moments).

See [here](http://pandas.pydata.org/pandas-docs/stable/timeseries.html) and [here](https://github.com/pydata/pydata-book/blob/master/ch10.ipynb) for examples.

## Ch11. Financial and Economic Data Application

`resample` converts data to a fixed frequency while `reindex` conforms data to a new index.

`DatetimeIndex` immutable ndarray of datetime64 data, represented internally as int65, and which can be boxed to Timestamp objects that are subclasses of datetime and carry metadata such as frequency information. See [here](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DatetimeIndex.html).

`time` and `between_time` can be used to select time series data.

See [here](https://github.com/pydata/pydata-book/blob/master/ch11.ipynb) for examples.

## Ch12. Advanced NumPy

`ravel` does not produce a copy of the underlying data if it does not have to.

`flatten` behaves like `ravel` except it always returns a copy of the data.

`take` and `put` can take out or insert values to array.

[ipython notebook examples](https://github.com/pydata/pydata-book/blob/master/ch12.ipynb).

[python language essentials](https://github.com/pydata/pydata-book/blob/master/appendix_python.ipynb).





# Other notes

`==` is for value equality. Use it when you would like to know if two objects have the same value.

`is` is for reference equality. use it when you would like to know if two references refer to the same object.

`.item()` method can be used to access a dictionary key and value in a loop. For example,

```python
animal_types = {"robin": "bird", "pug": "dog", "osprey": "bird"}
The .items method lets us access a dictionary key and value in a loop.
for key,value in animal_types.items():
print(key)
print(value)
# This is equal to the value
print(animal_types[key])
```

`.iloc` method in `pandas` is similar as indexing method in `numpy`



## Notes from [Python Data Analytics](http://www.amazon.com/Python-Data-Analytics-Fabio-Nelli/dp/1484209591/ref=sr_1_1?ie=UTF8&qid=1450048533&sr=8-1&keywords=Python+Data+Analytics)

### 1. Python Functions
* map(function, list)
* filter(function, list)
* reduce(function, list)
* lambda
* list comprehension
* [other built-in functions](https://docs.python.org/2/library/functions.html#reduce)



```python
items = [1, 2, 3, 4, 5]
def inc(x):
    return x+1
print list(map(inc, items))                # use map function
print list(map(lambda x: x+1, items))      # use map and lambda functions
print list(filter(lambda x: x < 4, items)) # use of filter
print reduce((lambda x,y: x+y), items)     # use of reduce
```

    [2, 3, 4, 5, 6]
    [2, 3, 4, 5, 6]
    [1, 2, 3]
    15


Pip commands:
* pip install package_name
* pip search package_name
* pip show package_name
* pip unistall package_name

### 2. NumPy


```python
import numpy as np
a = np.array([1, 2, 3])
print a.ndim  # dimension
print a.size  # total number of elments
print a.shape # shape of the array
print np.zeros((3, 3))   # zero array
print np.ones((3, 3))    # one array
```

    1
    3
    (3,)
    [[ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]]
    [[ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 1.  1.  1.]]



```python
np.arange(0, 10)     # similar as range(0, 10)
```




    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])




```python
np.arange(0, 12, 3)  # with interval, does NOT include 12
```




    array([0, 3, 6, 9])




```python
np.arange(0, 12, 3).reshape(2, 2)   # reshape the array
```




    array([[0, 3],
           [6, 9]])




```python
np.linspace(0, 10, 5)    # include 10
```




    array([  0. ,   2.5,   5. ,   7.5,  10. ])




```python
np.random.random((3, 3))
```




    array([[ 0.0226314 ,  0.05591402,  0.30557851],
           [ 0.61526516,  0.06592523,  0.94704285],
           [ 0.98933642,  0.74626897,  0.19137706]])




```python
A = np.arange(0, 9).reshape(3, 3)
B = np.ones((3, 3))
print A
print B
print A * B         # elementwise multiply
print np.dot(A, B)  # matrix multiply
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    [[ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 1.  1.  1.]]
    [[ 0.  1.  2.]
     [ 3.  4.  5.]
     [ 6.  7.  8.]]
    [[  3.   3.   3.]
     [ 12.  12.  12.]
     [ 21.  21.  21.]]



```python
# Indexing
a = np.arange(10, 16)
print a
```

    [10 11 12 13 14 15]



```python
print a[1:5:2]   # from index 1 to index 5 (exlcude), every 2 element
print a[:5:2]
print a[:5:]
```

    [11 13]
    [10 12 14]
    [10 11 12 13 14]



```python
A = np.arange(10, 19).reshape(3, 3)
print A
print A[0, :]
print A[:, 0]
print A[0:2, 0:2]
print A[[0, 2], 0:2]
```

    [[10 11 12]
     [13 14 15]
     [16 17 18]]
    [10 11 12]
    [10 13 16]
    [[10 11]
     [13 14]]
    [[10 11]
     [16 17]]



```python
A.mean(axis=0)   # mean, std, sum et al. along certain axis
```




    array([ 13.,  14.,  15.])




```python
np.apply_along_axis(np.mean, axis=0, arr=A)   # similar as above, here np.mean can be other functions
```




    array([ 13.,  14.,  15.])




```python
A[A < 13]         # selection
```




    array([10, 11, 12])




```python
A.reshape(1, 9)   # reshape the array
```




    array([[10, 11, 12, 13, 14, 15, 16, 17, 18]])




```python
A.ravel()        # turn array into one dimension
```




    array([10, 11, 12, 13, 14, 15, 16, 17, 18])




```python
A.transpose()
```




    array([[10, 13, 16],
           [11, 14, 17],
           [12, 15, 18]])




```python
# combine two arrays
A = np.ones((3, 3))
B = np.zeros((3, 3))
print np.vstack((A, B))   # vertically combine two arrays
print np.hstack((A, B))   # horizontally combine two arrays
```

    [[ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 1.  1.  1.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]
     [ 0.  0.  0.]]
    [[ 1.  1.  1.  0.  0.  0.]
     [ 1.  1.  1.  0.  0.  0.]
     [ 1.  1.  1.  0.  0.  0.]]



```python
# combine multipy 1-d arrays
a = np.array([0, 1, 2])
b = np.array([3, 4, 5])
c = np.array([6, 7, 8])
print np.column_stack((a, b, c))   # stack for each column
print np.row_stack((a, b, c))      # stack for each row
```

    [[0 3 6]
     [1 4 7]
     [2 5 8]]
    [[0 1 2]
     [3 4 5]
     [6 7 8]]



```python
# split arrays
A = np.arange(16).reshape((4, 4))
print A
[B, C] = np.hsplit(A, 2)
print B
print C
[B, C] = np.vsplit(A, 2)
print B
print C
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    [[ 0  1]
     [ 4  5]
     [ 8  9]
     [12 13]]
    [[ 2  3]
     [ 6  7]
     [10 11]
     [14 15]]
    [[0 1 2 3]
     [4 5 6 7]]
    [[ 8  9 10 11]
     [12 13 14 15]]



```python
# A more complex way of splitting
[A1, A2, A3] = np.split(A, [1, 3], axis=1)  # split to 3 parts, 0:1, 1:3, 3:end
print A1
print A2
print A3
```

More array split can be found [here](http://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html).

Two arrays may be subjected to broadcasting when all their dimensions are compatible, i.e., the length of each dimension must be equal between the two array or one of them must be equal to 1. 


```python
A = np.arange(16).reshape(4, 4)
b = np.arange(4)
print A
print b
print A+b
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
    [0 1 2 3]
    [[ 0  2  4  6]
     [ 4  6  8 10]
     [ 8 10 12 14]
     [12 14 16 18]]



```python
# save and load data
np.save('saved_data', data)
np.load('saved_data.npy')
# read data in a text file
np.genfromtxt('data.csv', delimiter=',', names=True)
```

### 3. Pandas

Two primary data structures:
* Series
* DataFrame


```python
import pandas as pd
s = pd.Series([12, -4, 7, 9], index=['a', 'b', 'c', 'd'])
print s
print s.values
print s.index
print s['b']
print s[1]
print s[['b', 'c']]
```

    a    12
    b    -4
    c     7
    d     9
    dtype: int64
    [12 -4  7  9]
    Index([u'a', u'b', u'c', u'd'], dtype='object')
    -4
    -4
    b   -4
    c    7
    dtype: int64



```python
print s.unique()
print s.value_counts()
print s.isin([0, 7])
```

    [12 -4  7  9]
     7     1
     12    1
    -4     1
     9     1
    dtype: int64
    a    False
    b    False
    c     True
    d    False
    dtype: bool



```python
print s.isnull()   # NaN or not
print s.notnull()  # NaN or not
```

    a    False
    b    False
    c    False
    d    False
    dtype: bool
    a    True
    b    True
    c    True
    d    True
    dtype: bool



```python
# Series as dictionaries
mydict = {'red': 2000, 'blue': 1000, 'yellow': 500, 'orange': 1000}
myseries = pd.Series(mydict)
myseries
```




    blue      1000
    orange    1000
    red       2000
    yellow     500
    dtype: int64




```python
# DataFrame
data = {'color' : ['blue','green','yellow','red','white'],
'object' : ['ball','pen','pencil','paper','mug'],
'price' : [1.2,1.0,0.6,0.9,1.7]}
frame = pd.DataFrame(data)
frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>object</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>blue</td>
      <td>ball</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>green</td>
      <td>pen</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yellow</td>
      <td>pencil</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>red</td>
      <td>paper</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>white</td>
      <td>mug</td>
      <td>1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2 = pd.DataFrame(data, columns=['object','price'])
frame2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>object</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ball</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>pen</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>pencil</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>paper</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mug</td>
      <td>1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2 = pd.DataFrame(data, index=['one','two','three','four','five'])
frame2
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>object</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>blue</td>
      <td>ball</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>two</th>
      <td>green</td>
      <td>pen</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>three</th>
      <td>yellow</td>
      <td>pencil</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>four</th>
      <td>red</td>
      <td>paper</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>five</th>
      <td>white</td>
      <td>mug</td>
      <td>1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
print frame2.index       # index names
print frame2.columns     # column names
print frame2.values      # values
```

    Index([u'one', u'two', u'three', u'four', u'five'], dtype='object')
    Index([u'color', u'object', u'price'], dtype='object')
    [['blue' 'ball' 1.2]
     ['green' 'pen' 1.0]
     ['yellow' 'pencil' 0.6]
     ['red' 'paper' 0.9]
     ['white' 'mug' 1.7]]



```python
# select elements
print frame.price
print frame['price']
print frame.ix[:, 2]
```

    0    1.2
    1    1.0
    2    0.6
    3    0.9
    4    1.7
    Name: price, dtype: float64
    0    1.2
    1    1.0
    2    0.6
    3    0.9
    4    1.7
    Name: price, dtype: float64
    0    1.2
    1    1.0
    2    0.6
    3    0.9
    4    1.7
    Name: price, dtype: float64



```python
frame.index.name = 'id'
frame.columns.name = 'item'
frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>item</th>
      <th>color</th>
      <th>object</th>
      <th>price</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>blue</td>
      <td>ball</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>green</td>
      <td>pen</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>yellow</td>
      <td>pencil</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>red</td>
      <td>paper</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>white</td>
      <td>mug</td>
      <td>1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
import numpy as np
frame['new'] = np.arange(0, 5)     # add new column
print frame
del frame['new']    # delete column
print frame
```

    item   color  object  price  new
    id                              
    0       blue    ball    1.2    0
    1      green     pen    1.0    1
    2     yellow  pencil    0.6    2
    3        red   paper    0.9    3
    4      white     mug    1.7    4
    item   color  object  price
    id                         
    0       blue    ball    1.2
    1      green     pen    1.0
    2     yellow  pencil    0.6
    3        red   paper    0.9
    4      white     mug    1.7



```python
frame.T   # transpose
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>id</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
    <tr>
      <th>item</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>color</th>
      <td>blue</td>
      <td>green</td>
      <td>yellow</td>
      <td>red</td>
      <td>white</td>
    </tr>
    <tr>
      <th>object</th>
      <td>ball</td>
      <td>pen</td>
      <td>pencil</td>
      <td>paper</td>
      <td>mug</td>
    </tr>
    <tr>
      <th>price</th>
      <td>1.2</td>
      <td>1</td>
      <td>0.6</td>
      <td>0.9</td>
      <td>1.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
ser = pd.Series([5,0,3,8,4], index=['red','blue','yellow','white','green'])
print ser
print ser.index
print ser.idxmin()    # index with the lowest value
print ser.idxmax()    # index with the largest value
```

    red       5
    blue      0
    yellow    3
    white     8
    green     4
    dtype: int64
    Index([u'red', u'blue', u'yellow', u'white', u'green'], dtype='object')
    blue
    white



```python
ser.index.is_unique        # check if the index is unique or not
```




    True




```python
# Reindexing
ser = pd.Series([2,5,7,4], index=['one','two','three','four'])
print ser
ser.reindex(['three','four','five','one'])
```

    one      2
    two      5
    three    7
    four     4
    dtype: int64





    three     7
    four      4
    five    NaN
    one       2
    dtype: float64




```python
# fill the series
ser3 = pd.Series([1,5,6,3],index=[0,3,5,6])
print ser3
print ser3.reindex(range(6), method='ffill')   # fill forward
print ser3.reindex(range(6), method='bfill')   # fill backward
```

    0    1
    3    5
    5    6
    6    3
    dtype: int64
    0    1
    1    1
    2    1
    3    5
    4    5
    5    6
    dtype: int64
    0    1
    1    5
    2    5
    3    5
    4    6
    5    6
    dtype: int64



```python
# drop 
ser = pd.Series(np.arange(4.), index=['red','blue','yellow','white'])
print ser
print ser.drop(['blue', 'yellow'])
```

    red       0
    blue      1
    yellow    2
    white     3
    dtype: float64
    red      0
    white    3
    dtype: float64



```python
frame1 = pd.DataFrame(np.arange(16).reshape((4,4)),
... index=['red','blue','yellow','white'],
... columns=['ball','pen','pencil','paper'])

frame2 = pd.DataFrame(np.arange(12).reshape((4,3)),
... index=['blue','green','white','yellow'],
... columns=['mug','pen','ball'])
print frame1
print frame2
print frame1 + frame2
print frame1.add(frame2)   # operations is done by index
```

            ball  pen  pencil  paper
    red        0    1       2      3
    blue       4    5       6      7
    yellow     8    9      10     11
    white     12   13      14     15
            mug  pen  ball
    blue      0    1     2
    green     3    4     5
    white     6    7     8
    yellow    9   10    11
            ball  mug  paper  pen  pencil
    blue       6  NaN    NaN    6     NaN
    green    NaN  NaN    NaN  NaN     NaN
    red      NaN  NaN    NaN  NaN     NaN
    white     20  NaN    NaN   20     NaN
    yellow    19  NaN    NaN   19     NaN
            ball  mug  paper  pen  pencil
    blue       6  NaN    NaN    6     NaN
    green    NaN  NaN    NaN  NaN     NaN
    red      NaN  NaN    NaN  NaN     NaN
    white     20  NaN    NaN   20     NaN
    yellow    19  NaN    NaN   19     NaN



```python
frame1.sort_index(ascending=False)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ball</th>
      <th>pen</th>
      <th>pencil</th>
      <th>paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>yellow</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>white</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
    <tr>
      <th>red</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>blue</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame1.sort_index(by=['pen', 'pencil'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ball</th>
      <th>pen</th>
      <th>pencil</th>
      <th>paper</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>red</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
    <tr>
      <th>blue</th>
      <td>4</td>
      <td>5</td>
      <td>6</td>
      <td>7</td>
    </tr>
    <tr>
      <th>yellow</th>
      <td>8</td>
      <td>9</td>
      <td>10</td>
      <td>11</td>
    </tr>
    <tr>
      <th>white</th>
      <td>12</td>
      <td>13</td>
      <td>14</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>




```python
# frame.corr()
# frame.cov()
# frame.corrwith(frame2)
# frame.dropna()
# frame.fillna(0)
# frame.stack()
# frame.unstack()
```

**Reading and writting data using Pandas**

* Readers: read_csv, read_excel, read_hdf, read_sql, read_json, read_html, read_stata, read_clipboard, read_pickle, read_msgpack, read_gbq
* Writers: to_csv, to_excel, to_hdf, to_sql, to_json, to_html, to_stata, to_clipboard, to_pickle, to_msgpack, to_gbq

** Data Manipulation **


```python
import numpy as np
import pandas as pd

frame1 = pd.DataFrame( {'id':['ball','pencil','pen','mug','ashtray'],\
        'price': [12.33,11.44,33.21,13.23,33.62]})
frame2 = pd.DataFrame( {'id':['pencil','pencil','ball','pen'],\
        'color': ['white','red','red','black']})
print frame1
print frame2
print pd.merge(frame1, frame2, on='id')  # merge by default perform inner join
print pd.merge(frame1, frame2, on='id', how='outer')  # outer join
# There are also right and left join. 
# To make merge of mulitple keys, you simply just add a list to 'on' option.
# You can also do merge based on index (just set right_index=True, or left_index=True).
# You can also do df1.join(df2).
```

            id  price
    0     ball  12.33
    1   pencil  11.44
    2      pen  33.21
    3      mug  13.23
    4  ashtray  33.62
       color      id
    0  white  pencil
    1    red  pencil
    2    red    ball
    3  black     pen
           id  price  color
    0    ball  12.33    red
    1  pencil  11.44  white
    2  pencil  11.44    red
    3     pen  33.21  black
            id  price  color
    0     ball  12.33    red
    1   pencil  11.44  white
    2   pencil  11.44    red
    3      pen  33.21  black
    4      mug  13.23    NaN
    5  ashtray  33.62    NaN



```python
# combine data frames
pd.concat([df1, df2], axis=1, join='inner')
pd.concat([df1, df2], keys=[1, 2])  # set keys for df1(1) and df2(2)
```


```python
# if values are different for same index
ser1 = pd.Series(np.random.rand(5),index=[1,2,3,4,5])
ser2 = pd.Series(np.random.rand(4),index=[2,4,5,6])
print ser1
print ser2
print ser1.combine_first(ser2)
print ser2.combine_first(ser1)
```

    1    0.611084
    2    0.707381
    3    0.603744
    4    0.487561
    5    0.799834
    dtype: float64
    2    0.367192
    4    0.429333
    5    0.884948
    6    0.998197
    dtype: float64
    1    0.611084
    2    0.707381
    3    0.603744
    4    0.487561
    5    0.799834
    6    0.998197
    dtype: float64
    1    0.611084
    2    0.367192
    3    0.603744
    4    0.429333
    5    0.884948
    6    0.998197
    dtype: float64



```python
# pivoting
frame1 = pd.DataFrame(np.arange(9).reshape(3,3),\
                      index=['white','black','red'],\
                      columns=['ball','pen','pencil'])
frame1
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ball</th>
      <th>pen</th>
      <th>pencil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>white</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>black</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>red</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame1.stack()
```




    white  ball      0
           pen       1
           pencil    2
    black  ball      3
           pen       4
           pencil    5
    red    ball      6
           pen       7
           pencil    8
    dtype: int64




```python
(frame1.stack()).unstack()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ball</th>
      <th>pen</th>
      <th>pencil</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>white</th>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>black</th>
      <td>3</td>
      <td>4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>red</th>
      <td>6</td>
      <td>7</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
longframe = pd.DataFrame({ 'color':['white','white','white',\
                                    'red','red','red',\
                                    'black','black','black'],\
                          'item':['ball','pen','mug',\
                                  'ball','pen','mug',\
                                  'ball','pen','mug'],\
                          'value': np.random.rand(9)})
longframe
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>item</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>ball</td>
      <td>0.182524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>white</td>
      <td>pen</td>
      <td>0.348580</td>
    </tr>
    <tr>
      <th>2</th>
      <td>white</td>
      <td>mug</td>
      <td>0.087091</td>
    </tr>
    <tr>
      <th>3</th>
      <td>red</td>
      <td>ball</td>
      <td>0.180152</td>
    </tr>
    <tr>
      <th>4</th>
      <td>red</td>
      <td>pen</td>
      <td>0.674948</td>
    </tr>
    <tr>
      <th>5</th>
      <td>red</td>
      <td>mug</td>
      <td>0.263541</td>
    </tr>
    <tr>
      <th>6</th>
      <td>black</td>
      <td>ball</td>
      <td>0.838841</td>
    </tr>
    <tr>
      <th>7</th>
      <td>black</td>
      <td>pen</td>
      <td>0.395436</td>
    </tr>
    <tr>
      <th>8</th>
      <td>black</td>
      <td>mug</td>
      <td>0.019370</td>
    </tr>
  </tbody>
</table>
</div>




```python
wideframe = longframe.pivot('color', 'item')
wideframe
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">value</th>
    </tr>
    <tr>
      <th>item</th>
      <th>ball</th>
      <th>mug</th>
      <th>pen</th>
    </tr>
    <tr>
      <th>color</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>black</th>
      <td>0.838841</td>
      <td>0.019370</td>
      <td>0.395436</td>
    </tr>
    <tr>
      <th>red</th>
      <td>0.180152</td>
      <td>0.263541</td>
      <td>0.674948</td>
    </tr>
    <tr>
      <th>white</th>
      <td>0.182524</td>
      <td>0.087091</td>
      <td>0.348580</td>
    </tr>
  </tbody>
</table>
</div>




```python
# remove values from data frame
frame1 = pd.DataFrame(np.arange(9).reshape(3,3),\
                      index=['white','black','red'],\
                      columns=['ball','pen','pencil'])
print frame1
del frame1['ball']
print frame1
print frame1.drop('white')
```

           ball  pen  pencil
    white     0    1       2
    black     3    4       5
    red       6    7       8
           pen  pencil
    white    1       2
    black    4       5
    red      7       8
           pen  pencil
    black    4       5
    red      7       8



```python
# data transformation
dframe = pd.DataFrame({ 'color': ['white','white','red','red','white'],\
                       'value': [2,1,3,3,2]})
print dframe
print dframe.duplicated()  # check if there's duplicated data
dframe.drop_duplicates()   # drop duplicated data
```

       color  value
    0  white      2
    1  white      1
    2    red      3
    3    red      3
    4  white      2
    0    False
    1    False
    2    False
    3     True
    4     True
    dtype: bool





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>white</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>red</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# replace 
frame = pd.DataFrame({ 'item':['ball','mug','pen','pencil','ashtray'],\
                      'color':['white','rosso','verde','black','yellow'], \
                      'price':[5.56,4.20,1.30,0.56,2.75]})
frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>item</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>ball</td>
      <td>5.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rosso</td>
      <td>mug</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>verde</td>
      <td>pen</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>black</td>
      <td>pencil</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yellow</td>
      <td>ashtray</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
newcolors = {'rosso': 'red',
             'verde': 'green'}
frame.replace(newcolors)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>item</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>ball</td>
      <td>5.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>mug</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>green</td>
      <td>pen</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>black</td>
      <td>pencil</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yellow</td>
      <td>ashtray</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# map
frame = pd.DataFrame({ 'item':['ball','mug','pen','pencil','ashtray'],
                      'color':['white','red','green','black','yellow']})
frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>item</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>ball</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>mug</td>
    </tr>
    <tr>
      <th>2</th>
      <td>green</td>
      <td>pen</td>
    </tr>
    <tr>
      <th>3</th>
      <td>black</td>
      <td>pencil</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yellow</td>
      <td>ashtray</td>
    </tr>
  </tbody>
</table>
</div>




```python
prices = {'ball' : 5.56, 'mug' : 4.20, 'bottle' : 1.30,
         'scissors' : 3.41, 'pen' : 1.30, 'pencil' : 0.56,
         'ashtray' : 2.75}
frame['price'] = frame['item'].map(prices)
frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>item</th>
      <th>price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>ball</td>
      <td>5.56</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>mug</td>
      <td>4.20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>green</td>
      <td>pen</td>
      <td>1.30</td>
    </tr>
    <tr>
      <th>3</th>
      <td>black</td>
      <td>pencil</td>
      <td>0.56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>yellow</td>
      <td>ashtray</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>




```python
# rename index or columns
reindex = {0: 'first',
           1: 'second',
           2: 'third',
           3: 'fourth',
           4: 'fifth'}
recolumn = {'item':'object',
            'price': 'value'}
print frame.rename(index=reindex, columns=recolumn)
print frame
frame.rename(index=reindex, columns=recolumn, inplace=True) # inplace makes sure the changes happen to the original dataframe
print frame
```

             color   object  value
    first    white     ball   5.56
    second     red      mug   4.20
    third    green      pen   1.30
    fourth   black   pencil   0.56
    fifth   yellow  ashtray   2.75
        color     item  price
    0   white     ball   5.56
    1     red      mug   4.20
    2   green      pen   1.30
    3   black   pencil   0.56
    4  yellow  ashtray   2.75
             color   object  value
    first    white     ball   5.56
    second     red      mug   4.20
    third    green      pen   1.30
    fourth   black   pencil   0.56
    fifth   yellow  ashtray   2.75



```python
# discretization and binning

# change to different categories
results = [12,34,67,55,28,90,99]
bins = [0,25,50,75,100]
cat = pd.cut(results, bins)
cat
```




    [(0, 25], (25, 50], (50, 75], (50, 75], (25, 50], (75, 100], (75, 100]]
    Categories (4, object): [(0, 25] < (25, 50] < (50, 75] < (75, 100]]




```python
cat.categories
```




    Index([u'(0, 25]', u'(25, 50]', u'(50, 75]', u'(75, 100]'], dtype='object')




```python
cat.value_counts()
```




    (0, 25]      1
    (25, 50]     2
    (50, 75]     2
    (75, 100]    2
    dtype: int64



Another method is `qcut()`, which can directly divide the sample into quntiles.


```python
# filter outliers
randframe = pd.DataFrame(np.random.randn(1000,3))
print randframe.describe()
randframe[(np.abs(randframe) > (3*randframe.std())).any(1)]
```

                     0            1            2
    count  1000.000000  1000.000000  1000.000000
    mean     -0.009871     0.004314    -0.002460
    std       1.031753     0.964007     0.961792
    min      -3.266127    -2.750076    -2.929541
    25%      -0.714287    -0.671256    -0.705985
    50%      -0.022217     0.027222    -0.027829
    75%       0.694838     0.648539     0.674543
    max       3.395532     3.655741     3.336637





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>381</th>
      <td>0.301428</td>
      <td>2.912618</td>
      <td>-0.495989</td>
    </tr>
    <tr>
      <th>419</th>
      <td>0.992485</td>
      <td>3.655741</td>
      <td>1.672764</td>
    </tr>
    <tr>
      <th>549</th>
      <td>-0.904884</td>
      <td>2.820288</td>
      <td>3.336637</td>
    </tr>
    <tr>
      <th>655</th>
      <td>3.395532</td>
      <td>-0.353160</td>
      <td>-0.007438</td>
    </tr>
    <tr>
      <th>778</th>
      <td>-3.266127</td>
      <td>1.645688</td>
      <td>1.344644</td>
    </tr>
    <tr>
      <th>832</th>
      <td>0.715832</td>
      <td>3.406925</td>
      <td>0.116812</td>
    </tr>
    <tr>
      <th>917</th>
      <td>0.386274</td>
      <td>0.301172</td>
      <td>-2.929541</td>
    </tr>
  </tbody>
</table>
</div>




```python
# permutation
nframe = pd.DataFrame(np.arange(25).reshape(5,5))
new_order = np.random.permutation(3)
print nframe
print new_order
print nframe.take(new_order)
```

        0   1   2   3   4
    0   0   1   2   3   4
    1   5   6   7   8   9
    2  10  11  12  13  14
    3  15  16  17  18  19
    4  20  21  22  23  24
    [0 2 1]
        0   1   2   3   4
    0   0   1   2   3   4
    2  10  11  12  13  14
    1   5   6   7   8   9



```python
# string manipulation
text = '16 Bolton Avenue, Boston'
tokens = [s.strip() for s in text.split(',')]
print tokens
','.join(tokens)
```

    ['16 Bolton Avenue', 'Boston']





    '16 Bolton Avenue,Boston'




```python
text.index('Boston')
```




    18




```python
text.find('Boston')
```




    18




```python
text.count('e')
```




    2




```python
text.replace('Avenue', 'Street')
```




    '16 Bolton Street, Boston'




```python
# Regular Expressions (pattern matching, substitution, splitting)
import re
text = "This is an\t odd \n text!"
re.split('\s+', text)   # \s+ represent one or more spaces
```




    ['This', 'is', 'an', 'odd', 'text!']



More details can be found [here](https://docs.python.org/2/library/re.html)


```python
# Data Aggregation
frame = pd.DataFrame({ 'color': ['white','red','green','red','green'],
                      'object': ['pen','pencil','pencil','ashtray','pen'],
                      'price1' : [5.56,4.20,1.30,0.56,2.75],
                      'price2' : [4.75,4.12,1.60,0.75,3.15]})
frame
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>color</th>
      <th>object</th>
      <th>price1</th>
      <th>price2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>white</td>
      <td>pen</td>
      <td>5.56</td>
      <td>4.75</td>
    </tr>
    <tr>
      <th>1</th>
      <td>red</td>
      <td>pencil</td>
      <td>4.20</td>
      <td>4.12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>green</td>
      <td>pencil</td>
      <td>1.30</td>
      <td>1.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>red</td>
      <td>ashtray</td>
      <td>0.56</td>
      <td>0.75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>green</td>
      <td>pen</td>
      <td>2.75</td>
      <td>3.15</td>
    </tr>
  </tbody>
</table>
</div>




```python
group = frame['price1'].groupby(frame['color'])
print group.groups
print group.mean()
```

    {'white': [0], 'green': [2, 4], 'red': [1, 3]}
    color
    green    2.025
    red      2.380
    white    5.560
    Name: price1, dtype: float64


More advanced methods can be found [here](http://pandas.pydata.org/pandas-docs/stable/groupby.html).
