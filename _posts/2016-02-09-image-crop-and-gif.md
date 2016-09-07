---
layout: post
title: Notes Image crop and gif animation
tags: image
---
Script for crop png files:

```shell
#!/bin/bash
# crop white space for images in a directory
for X in *.png
do
    echo $X
    convert -trim $X $X
done
exit
# eof
```

Script for making a gif animation:

```shell
#!/bin/bash
# convert png to jpg, and then make a gif with delay in 1/100 seconds, and cycle(1) or not (0).
mogrify -format jpg *.png
convert -delay 10 -loop 0 *.jpg my_animation.gif
```

Script for [combing multiple pictures into one picture](http://superuser.com/questions/290656/combine-multiple-images-using-imagemagick):

```shell
#!/bin/bash
# Combine horizontally
convert in-1.jpg in-2.jpg +append outh.jpg
# Combine vertically
convert in-1.jpg in-2.jpg -append outv.jpg
```


Note: to use the `convert` command, you need to install [ImageMagic](http://www.imagemagick.org/script/index.php).
