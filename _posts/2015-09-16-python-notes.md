---
layout: post
title: Python notes
tags: Python
---

This is my personal python notes.

There are many many many ... python tutorials on the internet. I found some of them are pretty useful.

1. [Google python course (basic)](https://developers.google.com/edu/python/)
2. [Codecademy python course (basic)](www.codecademy.com/)
3. [Dataquest data science course](https://www.dataquest.io/)
4. [Quantitative Economics Python](http://quant-econ.net/py/index.html)
5. [Introduction to Python for Econometrics, Statistics and Data Analysis](https://www.kevinsheppard.com/images/0/09/Python_introduction.pdf)
6. [Python Scientific Lecture Notes](http://www.scipy-lectures.org/)


## Environment setting of different python versions

As we know, Python has different version (2.x vs 3.x). How to use two different versions while not mess them up? Using Conda, we can do this. 


** Install a different version of Python **

```python
# Create a 'python3' environment and install Python3
conda create --name python3 python=3

# Activate 'python3' environment
source active python3

# Deactive 'python3' environment
source deactive python3

# Check all environments under Conda
conda info --envs

# Install packages under current environment
conda install package_name
# Or
conda install --name environment_name package_name

# Update certain package
conda update package_name
```


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

Please see [Github](https://github.com/xzenggit/self_learning/blob/master/Python/Python%20Notes.ipynb)


## Learn Python3 the Hard Way


```python
# Ex1.
print("Hello world!")
```

    Hello world!



```python
# Ex2.
# Add Comments for coding
print("Alway add comments to your code!")
```

    Alway add comments to your code!



```python
# Ex3. Numbers and Math
print("test:", 25+30/6)
print("Is it greater?", 5>-2)
```

    test: 30.0
    Is it greater? True



```python
# Ex4. Variables and names
# This is a variable
cars = 100
print("There are", cars, "cars available.")
```

    There are 100 cars available.



```python
# Ex5. More variables and printing
# f means format
my_name = "Tom"
my_age = 18
print(f"My name is {my_name}")
print(f"My age is {my_age}")
```

    My name is Tom
    My age is 18



```python
# Ex6. Strings and text
# Another kind of formatting using the .format() syntax
hilarious = False
joke_evaluation = "Isn't that joke so funny?! {}"
print(joke_evaluation.format(hilarious))
```

    Isn't that joke so funny?! False



```python
# Ex7. More printing
a = "A"
b = "B"
print("*"*10)
print(a+b)
print("*"*10)
# By default, the end is "Enter".
print(a, end=" ")
print(b)
print("*"*10)
print(a)
print(b)
```

    **********
    AB
    **********
    A B
    **********
    A
    B



```python
# Ex8. Printting, printing
formatter = "{} {} {}"
print(formatter.format(1, 2, 3))
```

    1 2 3



```python
# Ex9. Printting, printing, priting
days = "Mon Tue Wed"
months = "Jan\nFeb\nMar"

print(days)
print(months)

print("""
With the three double quotes.
We'll be able to type as much as we like.
Even three lines if we want, or 4, or 5.
""")
```

    Mon Tue Wed
    Jan
    Feb
    Mar
    
    With the three double quotes.
    We'll be able to type as much as we like.
    Even three lines if we want, or 4, or 5.
    



```python
# Ex10. What was that?
# \t: tab; \n:enter
tabby_cat = "\tI'm tabbed in."
print(tabby_cat)
```

        I'm tabbed in.



```python
# Ex11/12. Asking questions
print("How old are you?", end=" ")
age = input()
print(f"You are {age} old.")
```

    How old are you? You are 10 old.



```python
# Ex13. Parameters, unpacking, variables.
# Save the following in ex13.py
# %load ex13.py
from sys import argv
script, first, second, third = argv

print("The script is called:", script)
print("Your first variable is:", first)
print("Your second variable is:", second)
print("Your third variable is:", third)
```


```python
%run ex13.py first 2nd 3rd
```

    The script is called: ex13.py
    Your first variable is: first
    Your second variable is: 2nd
    Your third variable is: 3rd



```python
# Ex15. Read files
filename='ex13.py'
txt = open(filename)
print(f"Here is your file {filename}.")
print(txt.read())
```

    Here is your file ex13.py.
    from sys import argv
    script, first, second, third = argv
    
    print("The script is called:", script)
    print("Your first variable is:", first)
    print("Your second variable is:", second)
    print("Your third variable is:", third)


Ex16. Reading and writing files
A few commands
* close: close file
* read: read the file content
* readline: read just one line of a text file
* truncate: empty the file.
* write('stuf'): write something into the file.
* seek(0): move the read/write location to the beginning of the file.


```python
# Ex18. Names, variables, code, functions
# *args here means to take all the arguments to the function and put them
# in a args list.
def print_two(*args):
    arg1, arg2 = args
    print(f"arg1: {arg1}, arg2: {arg2}")
    
def print_two_again(arg1, arg2):
    print(f"arg1: {arg1}, arg2: {arg2}")

def print_one(arg1):
    print(f"arg1: {arg1}")
    
def print_none():
    print("I got nothing.")

print_two("Zed", "Shaw")
print_two_again("Zed", "Shaw")
print_one("First!")
print_none()
```

    arg1: Zed, arg2: Shaw
    arg1: Zed, arg2: Shaw
    arg1: First!
    I got nothing.



```python
# Ex21. Functions can return something
def add(a, b):
    print(f"Adding {a} + {b}")
    return a + b
age = add(30, 5)
print(age)
```

    Adding 30 + 5
    35



```python
# Ex30. Else and if
cars = 40
trucks = 15

if cars > trucks:
    print("Cars are more than trucks.")
else:
    print("Trucks are more than cars.")
```

    Cars are more than trucks.



```python
# Ex32. Loops and lists
the_count = [1, 2, 3, 4, 5] 

for number in the_count:
    print(f"This is count {number}")
```

    This is count 1
    This is count 2
    This is count 3
    This is count 4
    This is count 5



```python
# Ex33. While loops
i = 0
numbers = []

while i < 6:
    print(f"At the top i is {i}")
    numbers.append(i)
    i = i + 1
print(numbers)
```

    At the top i is 0
    At the top i is 1
    At the top i is 2
    At the top i is 3
    At the top i is 4
    At the top i is 5
    [0, 1, 2, 3, 4, 5]



```python
# Ex34. Assessing elements of lists
a = [1, 2, 3, 4]
a[0]
```




    1




```python
# Ex38. Doing things to lists.
class Thing(object):
    def test(object, message):
        print(message)

a = Thing()
a.test("hello")
```

    hello



```python
# Ex39. Dictionaries.
stuff = {'name': 'Zed', 
         'age': 39, 
         'height': 6*12+2}
print(stuff['name'])
```

    Zed



```python
# Ex40. Modules, Classes and Objects

# The following is save as module mystuff.py
def apple():
    print("I am apples.")
    
# Then we can use the module as follow.
import mystuff
mystuff.apple()
```


```python
# Class is similar as modules
class MyStuff(object):
    
    def __init__(self):
        self.tangerine = "***"
        
    def apple(self):
        print("I am apples.")
        
thing = MyStuff()
thing.apple()
print(thing.tangerine)
```

    I am apples.
    ***



```python
class Song(object):
    
    def __init__(self, lyrics):
        self.lyrics = lyrics
        
    def sing_me_a_song(self):
        for line in self.lyrics:
            print(line)

happy_bday = Song(["Happy birthday to you", 
                  "I don't want to get sued", 
                  "So I'll stop right there"])
happy_bday.sing_me_a_song()
```

    Happy birthday to your
    I don't want to get sued
    So I'll stop right there



```python
# Ex42. Objects and Class
class Animal(object):
    pass

class Dog(Animal):
    
    def __init__(self, name):
        self.name = name

class Cat(Animal):
    
    def __init__(self, name):
        self.name = name
        
class Person(object):
    
    def __init__(self, name):
        self.name = name
        self.pet = None
        
class Employee(Person):
    
    def __init__(self, name, salary):
        super(Employee, self).__init__(name)
        self.salary = salary
# Super function is used to return a proxy object that delegates method 
# calls to a parent or sibling class of type. This is useful for 
# acessing iniherited methods that have been overridden in a class.
# The search order is same as that used by getattr() except that the
# type itself is skipped.
class Fish(object):
    pass

class Salmon(Fish):
    pass

rover = Dog("Rover")
satan = Cat("Satan")
mary = Person("Mary")
mary.pet = satan

frank = Employee("Frank", 12000)
frank.pet = rover
flipper = Fish()
crouse = Salmon()
```


```python
# Ex44. Inheritance.
# Inheritance is used to indicate that one class will get most
# or all of its features from a parent class.
# Three ways that the parent and child classes can interact:
# 1. Actions on the child imply an action on the parent.
# 2. Actions on the child override the action on the parent.
# 3. Actions on the child alter the action on the parent.

class Parent(object):
    
    def implicit(self):
        print("Parent implicit()")
        
class Child(Parent):
    pass

dad = Parent()
son = Child()

dad.implicit()
son.implicit()
# All subclass automatically get parent's features.
```

    Parent implicit()
    Parent implicit()



```python
class Parent(object):
    
    def override(self):
        print("Parent override()")
        
class Child(Parent):
    
    def override(self):
        print("Child override()")

dad = Parent()
son = Child()

dad.override()
son.override()
# If same function is defined in subclass, parent function will be overrided.
```

    Parent override()
    Child override()



```python
class Parent(object):
    
    def altered(self):
        print("Parent altered()")
        
class Child(Parent):
    
    def altered(self):
        print("Child, before Parent altered()")
        super(Child, self).altered()
        print("Child, After Parent altered()")
        
dad = Parent()
son = Child()

dad.altered()
son.altered()

# super() function here can get the Parent version altered().
```

    Parent altered()
    Child, before Parent altered()
    Parent altered()
    Child, After Parent altered()



```python
# Use other classes and modules rather than rely on implicit inheritance.
class Other(object):
    
    def override(self):
        print("Other overrider()")
        
    def implicit(self):
        print("Other implicit()")
        
    def altered(self):
        print("Other altered()")
        
class Child(object):
    
    def __init__(self):
        self.other = Other()
        
    def implicit(self):
        self.other.implicit()
    
    def override(self):
        print("Child override()")
        
    def altered(self):
        print("Child, before Other altered()")
        self.other.altered()
        print("Child, after Other altered()")
        
son = Child()

son.implicit()
son.override()
son.altered()
```

    Other implicit()
    Child override()
    Child, before Other altered()
    Other altered()
    Child, after Other altered()


* Avoid multiple inheritance at all costs, as it’s too complex to be reliable.
* Use composition to package code into modules that are used in many different unrelated places
and situations.
* Use inheritance only when there are clearly related reusable pieces of code that fit under a single common concept or if you have to because of something you’re using.
* Use vritual environment when start a new project.


** Creat the Skeleton Project Directory**

```bash
$ mkdir projects
$ cd projects/
$ mkdir skeleton
$ cd skeleton
$ mkdir bin NAME tests docs
# skeleton: put project basis
# NAME: the name of your project's main module.
$ touch NAME/__init__.py
$ touch tests/__init__.py
```

To create a setup.py file, so we can install our project.
```python
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
    
config = {
    'desription': 'My Project',
    'author': 'My Name',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'My email.',
    'version': '0.1'
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'projectname'
}
setup(**config)
```

We also need a simple skeleton file for tests named tests/NAME_tests.py:

```python
from nose.tools import *
import NAME

def setup():
    print("SETUP!")
    
def teardown():
    print("TEAR DOWN!")
    
def test_basic():
    print("I RAN!")
```
