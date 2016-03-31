---
layout: post
title: R notes
tags: R notes
---
This is my personal note for learning R. A lot of contents come from online resources, such as Coursera and Datacamp.

## Basic R, R markdown, and git

### Basic R

If you know the basic programming skills, some cheat sheets will be quite useful for you. Here's what I found:

* [Quandl basic R cheat sheet](https://s3.amazonaws.com/quandl-static-content/Documents/Quandl+-+R+Cheat+Sheet.pdf)

* [R refercen card by Tom Short](https://cran.r-project.org/doc/contrib/Short-refcard.pdf)

* [R Reference card 2.0 by  Matt Boggott](https://cran.r-project.org/doc/contrib/Baggott-refcard-v2.pdf)

* [Google's R style guide](http://google-styleguide.googlecode.com/svn/trunk/Rguide.xml)

* [Rstudio cheat sheets](https://www.rstudio.com/resources/cheatsheets/).

### R markdown

Rmarkdown is very useful tool for putting notes together. More references can be found at [Rstudio rmarkdown reference](https://www.rstudio.com/wp-content/uploads/2015/03/rmarkdown-reference.pdf) and the [official cheat sheet](https://www.rstudio.com/wp-content/uploads/2015/02/rmarkdown-cheatsheet.pdf).

Github also has very useful markdown [tutorial](https://help.github.com/articles/markdown-basics/).

### git

Create a new repository on the command line:

```r
echo "# notes" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/xzenggit/notes.git
git push -u origin master
```

or push an existing repository from the command line

```r
git remote add origin https://github.com/xzenggit/notes.git
git push -u origin master
```

Github's guide for using git can be found  [here](https://help.github.com/categories/using-git/).

Some guides including github pages and other interesting stuff can also be found [here](https://guides.github.com/).

## Data manipulation (dplyr and data.table related)

The `dplyr` package provides very powerful functions for data manipulation:

* `filter()` and `slice()`

* `arrange()`

* `select()` and `rename()`

* `distinct()`

* `mutate()` and `transmute()`

* `summarise()`

* `sample_n()` and `sample_farc()`

Here is a good [introduction](https://cran.rstudio.com/web/packages/dplyr/vignettes/introduction.html) about `dplyr`.

For `data.table`, Datacamp has a good [tutorial](http://blog.datacamp.com/data-table-r-tutorial/) and [cheat-sheet](https://s3.amazonaws.com/assets.datacamp.com/img/blog/data+table+cheat+sheet.pdf), which are pretty straight.

## Exploratory data analysis (ggplot2 related)

The basic principle of exploratory data analysis is to better understand the data. So you can use whatever tools you like to analyze the data, and get preliminary understanding of the data structure and distribution etc. 

Typlical, people plot distribution of interesting variables, or cluster difference variables, depending on the problem.

ggplot is a great graphic tool to explore the data. Please see:

* [ggplot2 tutoiral](http://www.ceb-institute.org/bbs/wp-content/uploads/2011/09/handout_ggplot2.pdf)

* [ggplot2 cheat sheet](https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf)

## Regression and statistical inference

Regression and statistical inference are more complicated topics. Here are some resources:

* [Coruse notes for Data Science Specialization in Coursera](http://sux13.github.io/DataScienceSpCourseNotes/)

* [R functions for regression analysis](https://cran.r-project.org/doc/contrib/Ricci-refcard-regression.pdf)

## Machine learing and big data

Machine learning and big data are really the future of data science. There are tons of tutorials and courses about this. Here's what I find:

* [10 popular machine learning algorithms](http://vitalflux.com/cheat-sheet-10-machine-learning-algorithms-r-commands/)

* [20 shor tutorials all data scientists should read](http://www.datasciencecentral.com/profiles/blogs/17-short-tutorials-all-data-scientists-should-read-and-practice)

* [R data mining](http://www.rdatamining.com/home)

* [Machine learning cheat sheets](http://designimag.com/best-machine-learning-cheat-sheets/)

* [Dzone big data and machine learning referece card](https://kaggle2.blob.core.windows.net/forum-message-attachments/8888/DZone%20introduction%20to%20ML.pdf?sv=2012-02-12&se=2015-08-27T19%3A31%3A49Z&sr=b&sp=r&sig=DO52btrK6hhocw6G1s6s3nQwokmJaYSlByktI16R8N0%3D)

* [Dzone big data guide 2014](https://dzone.com/storage/assets/20827-dzone_bigdataresearchguide_4%20(2).pdf)


