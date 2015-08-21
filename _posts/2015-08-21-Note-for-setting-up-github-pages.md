---
layout: post
title: Notes for setting up github pages
tags: github gh-pages
---

The following is how I setup the github papges.

Basically, I followed the steps given at [github-pages](https://pages.github.com/). For the user papges, you need to create a repo as `username.githup.io`. For the project pages, you need to create a branch called `gh-pages` under you project repo. The procedures are quite clear at its [help pages](https://help.github.com/categories/github-pages-basics/).

There are some templates you can download at http://jekyllthemes.org. After downloading your favorite template, you can uncompress it under your local repo directory (for user pages, it's your master branch; and for project pages, it's your gh-pages branch). 

Then, you need to change the the line `baseurl:` in the `_config.yml` file to `baseurl: http:/your-username.github.io`. Substitute `your-username` with your own github username. You can also change other things as you like. 

As last, you can do `git add .` and `git commit -m "new web"`, and `git push` to your github repo. Go to `your-username.github.io`(for user papges) or `your-username.github.io/project-name` (for project pages), you can find your own website now! The greatest thing is it's free!

How to make your first post? Under the `_posts` directory, create a file named like `2015-08-21-post-title.md` if you use markdown. Then put whatever you want in it following the markdown format, and push it back to your repo. You'll get your first post.

Checkout the jekyll [docs](http://jekyllrb.com/docs/posts/). They have a pretty good instruction on how-to. 
 

