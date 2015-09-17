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

## Notes from [Python for data analysis](https://github.com/pydata/pydata-book/blob/master/ch02.ipynb)

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
```
animal_types = {"robin": "bird", "pug": "dog", "osprey": "bird"}

The .items method lets us access a dictionary key and value in a loop.

for key,value in animal_types.items():

print(key)

print(value)

# This is equal to the value

print(animal_types[key])
```

`.iloc` method in `pandas` is similar as indexing method in `numpy`


