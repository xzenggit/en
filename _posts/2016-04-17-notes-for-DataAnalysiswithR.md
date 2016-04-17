---
layout: post
title: Notes for Data Analysis with R at Udacity
tags: R Udacity
---

## Notes for [Data Analysis with R at Udacity](https://www.udacity.com/course/data-analysis-with-r--ud651)

### R basics

* [Google R style guide](https://google.github.io/styleguide/Rguide.xml)
* [R cookbook](http://www.cookbook-r.com)

```r
# load data
data(mtcars)
# strcuture of dataframe
str(mtcars)
# dimension of the data
dim(mtcars)
# row names 
row.names(mtcars)
# access indiviual variable
mtcars$mpg
# get names
names(mtcars)
```

```r
# get current directory
getwd()
# set new directory
setwd()
```

```r
# Setting levels of ordered factors solution
reddit$age.range <- ordered(reddit$age.range, 
                             levels = c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 or Above'))

# Alternate Solution
reddit$age.range <- factor(reddit$age.range, 
                             levels = c('Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65 or Above'), ordered = T)
```

### Explore one variable

```r
pf <- read.csv('pseudo_facebook.tsv', sep = '\t')
names(pf)

# ggplot
ggplot(aes(x = dob_day), data = pf) + 
  geom_histogram(binwidth = 1) + 
  scale_x_continuous(breaks = 1:31)

# use facet_wrap
ggplot(data = pf, aes(x = dob_day)) + 
  geom_histogram(binwidth = 1) + 
  scale_x_continuous(breaks = 1:31) + 
  facet_wrap(~dob_month)

ggplot(aes(x = friend_count), data = subset(pf, !is.na(gender)), binwidth = 25) +   
  geom_histogram() +
  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50)) + 
  facet_wrap(~gender)

# make a table
table(pf$gender)
# use by to summary
by(pf$friend_count, pf$gender, summary)

ggplot(aes(x = tenure / 365), data = pf) + 
  geom_histogram(color = 'black', fill = '#F79420') + 
  scale_x_continuous(breaks = seq(1, 7, 1), limits = c(0, 7)) + 
  xlab('Number of years using Facebook') + 
  ylab('Number of users in sample')

# Tranforming data soultion
library(gridExtra)

p1 <- qplot(x = friend_count, data = pf)
p2 <- qplot(x = log10(friend_count), data = pf)
p3 <- qplot(x = sqrt(friend_count), data = pf)

grid.arrange(p1, p2, p3, ncol = 1)

# alernative method
p1 <- ggplot(aex(x = friend_count), data = pf) +
  geom_histogram()
p2 <- p1 + scale_x_log10()
p3 <- p1 + scale_x_sqrt()

grid.arrange(p1, p2, p3, ncol = 1)
```

```r
# Frequency Polygons
ggplot(aes(x = friend_count, y = ..count../sum(..count..)), data = subset(pf, !is.na(gender))) + 
  geom_freqpoly(aes(color = gender), binwidth=10) + 
  scale_x_continuous(limits = c(0, 1000), breaks = seq(0, 1000, 50)) + 
  xlab('Friend Count') + 
  ylab('Percentage of users with that friend count')

# Box plot
# box plot
qplot(x = gender, y = friend_count, data = subset(pf, !is.na(gender)),
      geom = 'boxplot') + 
  coord_cartesian(ylim = c(0, 1000))

# scatter plots
qplot(x = age, y = friend_count, data = pf)
# alternative scatter plots
ggplot(aes(x = age, y = friend_count), data = pf) + 
  geom_point() + 
  xlim(13, 90)
# alpha is transparency
# jitter to avoid overploting
ggplot(aes(x = age, y = friend_count), data = pf) + 
  geom_jitter(alpha = 1/20, position = position_jitter(h = 0)) + 
  xlim(13, 90) + 
  coor_trans(y = 'sqrt')
```


* [Facets](http://www.cookbook-r.com/Graphs/Facets_(ggplot2)/)

```r
facet_wrap(formula)
facet_wrap(~variable)
facet_grid(formula)
facet_grid(vertical~horizontal)
```

### Explore two variables

```r
# two variables
library(dplyr)
age_groups <- group_by(pf, age)
pf.fc_by_age <- summarise(age_groups,
          friend_count_mean = mean(friend_count),
          firend_count_median = median(friend_count),
          n = n())
pf.fc_by_age <- arrange(pf.fc_by_age)
head(pf.fc_by_age)

# alternative way
pf.fc_by_age <- pf %.%
  group_by(age) %.%
  summarise(friend_count_mean = mean(friend_count),
            friend_count_median = median(friend_count),
            n = n()) %.%
  arrange(age)

# overlayering summaries with Raw data
ggplot(aes( x = age, y = friend_count), data = pf) + 
  coord_cartesian(xlim = c(13, 70), ylim = c(0, 1000)) + 
  geom_point(alpha = 0.05,
             position = position_jitter(h = 0),
             color = 'orange') + 
  coord_trans(y = 'sqrt') +
  geom_line(stat = 'summary', fun.y = mean) + 
  geom_line(stat = 'summary', fun.y = quantitle, fun.args = list(probs = 0.1), 
            linetype = 2, color = 'blue') + 
  geom_line(stat = 'summary', fun.y = quantitle, fun.args = list(probs = 0.5), 
            linetype = 2, color = 'blue') + 
  geom_line(stat = 'summary', fun.y = quantitle, fun.args = list(probs = 0.9), 
            linetype = 2, color = 'blue')
```

```r
# correlation
cor.test(pf$age, pf$friend_count, method = 'pearson')
# alternative way
with(pf, cor.test(age, friend_count, method = 'pearson'))
# subset
with(subset(pf, age <= 70), cor.test(age, friend_count, method = 'pearson'))
```

```r
# create scatterplots, strong correlations
ggplot(aes(x = www_like_recieved, y = like_received), data = pf) + 
  geom_point() + 
  xlim(0, quantile(pf$www_likes_received, 0.95)) + 
  ylim(0, quantile(pf$likes_received, 0.95)) + 
  geom_smooth(method = 'lm', color = 'red')

# age with months means
pf.fc_by_age_months <- pf %.%
  group_by(age_with_months) %.%
  summarise(friend_count_mean = mean(friend_count),
            friend_count_median = median(friend_count),
            n = n()) %.%
  arrange(age_with_months)

# plot together
p1 <- ggplot(aes(x = age, y = friend_count_mean), 
       data = subset(pf.fc_by_age_months, age_with_months < 71)) + 
  geom_line() + 
  geom_smooth()

p2 <- ggplot(aes(x = age_with_months, y = friend_count_mean), 
       data = subset(pf.fc_by_age_months, age_with_months < 71)) + 
  geom_line() + 
  geom_smooth()

p3 <- ggplot(aes(x = round(age / 5) * 5, y = friend_count_mean), 
             data = subset(pf.fc_by_age_months, age_with_months < 71)) + 
  geom_line(stat = 'summary', fun.y = mean)
grid.arrange(p2, p1, p3, ncol = 1)
```

```r
# third qualitative variable
ggplot(aes(x = gender, y = age), data = subset(pf, !is.na(gender))) + 
  geom_boxplot() + 
  stat_summary(fun.y = mean, geom = 'point', shape = 4)

ggplot(aes(x = age, y = friend_count), data = subset(pf, !is.na(gender))) + 
  geom_line(aes(color = gender), stat = 'summary', fun.y = median)


pf.fc_by_age_gender <- pf %.%
  filter(!is.na(gender)) %.%
  group_by(age, gender) %.%
  summarise(mean_friend_count = mean(friend_count),
            median_friend_count = median(friend_count), 
            n = n()) %.%
  ungroup() %.%
  arrange(age)
```

```r
# cut a variable
pf$year_joined <- floor(2014 - pf$tenure/365)
pf$year_joined.bucket <- cut(pf$year_joined, 
                             c(2004, 2009, 2011, 2012, 2014))


# Friend rate
with(subset(pf, tenure >= 1), summary(friend_count / tenure))
```

### Explor many variables

```r
# load data and see its structure
yo <- read.csv('yogurt.csv')
str(yo)

# change the id from an int to a factor
yo$id <- factor(yo$id)
str(yo)

# histogram 
qplot(data = yo, x = price, fill = I('#F79420'))

# numer of purchurses
summary(yo)
length(unique(yo$price))
table(yo$price)

# add new variable all.purchases
yo <- tansform(yo, all.purchases = strawberry + blueberry + pina.colada + plain + mixed.berry)

# scatter plot
ggplot(aex(x = time, y = price), data = yo) + 
  geom_jitter(alpha = 1/4, shape = 21, fill = I('#F79420'))

# look at samples of households
set.seed(4230)
sample.ids <- sample(levels(yo$id), 16)

ggplot(aes(x = time, y = price),
       data = subset(yo, id %in% sample.ids)) + 
  facet_wrap( ~ id) + 
  geom_line() + 
  geom_point(aes(size = all.purchases), pch = 1)

# scatterplot matrices
library(GGally)
theme_set(theme_minimal(20))
set.seed(1836)
pf_subset <- pf[, c(2:15)]
names(pf_subset)
ggpairs(pf_subset[sample.int(nrow(pf_subset), 1000), ])


######################
# even more variables
nci <- read.table('nci.tsv')

# change the colnames to produce a nicer plot
colnames(nci) <- c(1:64)

# creat a heat map
# melt data to long format
nci.long.samp <- melt(as.matrix(nci[1:200, ]))
names(nci.long.samp) <- c('gene', 'case', 'value')
head(nci.long.samp)
# make the heat map
ggplot(aes(y = gene, x = case, fill = value),
       data = nci.long.samp) + 
  geom_title() + 
  scale_fill_gradientn(colors = colorRampPalette(c('blue', 'red'))(100))
```

### References

* [Data Wrangling with dplyr and tidyr](https://www.rstudio.com/wp-content/uploads/2015/02/data-wrangling-cheatsheet.pdf)


