---
layout: post
title: Notes for Docker
tags: Docker
---

### 1. Docker Basics

Docker is an open standard platforms for developing, packaging, and running portable distributed applications.

What makes Docker images unique and different from other virtual machines is that while each virtual machine image runs on a separate guest OS, the Docker images run within the same OS kernel.

[Install Docker with the Official Get Started with Docker](https://docs.docker.com/mac/)

[Docker cheet sheet](https://github.com/wsargent/docker-cheat-sheet)

An instance of an image is called container.

List all images: `$docker images`

List all containers: `$docker ps -a`

Download a pre-built image (ubuntu): `$docker pull ubuntu`

Run a container from a specific image:

`$ sudo docker run -i -t <image_id || repository:tag> /bin/bash`

Start a existed container:

`$ sudo docker start -i <image_id>`

Attach a running container:

`$ sudo docker attach <container_id>`

Exit without shutting down a container:

`[Ctrl-p] + [Ctrl-q]`

Start a new container:

`$ JOB=$(docker run -d ubuntu /bin/sh -c "while true; do echo Hello world; sleep 1; done")`

Stop the container: `$ docker stop $JOB`

Start the container: `$ docker start $JOB`

Restart the container: `$ docker restart $JOB`

Kill a container: `$ docker kill $JOB`

Remove a container:

`$ docker stop $JOB # Container must be stopped to remove it`
`$ docker rm $JOB` or `$ docker rm <container_id>`

Remove an image:

`$ docker rmi <image_id> # container must be removed first`

Search for images command:`$ docker search <image_name>`

Pull image: `$ docker pull <image_name>`

Commit your container to a new named image:

```bash
# run a container
$ docker run -it <image_name> /bin/bash
# make some changes in the container, then exit.
# commit a copy of this container to an image.
$ docker commit -m "changes content" <container_id> <new_image_name>
# -m is for the commit message
```

Build an image:

```bash
# create a Dockerfile like the following:
FROM ubuntu:14.04
MAINTAINER Kate Smith <ksmith@example.com>
RUN apt-get update && apt-get install -y ruby ruby-dev
RUN gem install sinatra

$ docker build - t <user>/<image_name>:<tag> .
```

Docker command flags:

* `-d`: run in background
* `-P`: map any require network
* `-i`: interactive

### 2. Docker for Hadoop

The image I used for launch Hadoop is [Hadoop-docker](https://github.com/sequenceiq/hadoop-docker). Following the instruction on the readme file, you can launch and test the Hadoop platfrom pretty easily.

To use the image: `$ docker run -it sequenceiq/hadoop-docker:2.7.1 /etc/bootstrap.sh -bash`

### 3. Docker for Spark

[Docker-spark](https://github.com/sequenceiq/docker-spark)

To use the image: `$ docker run -d -h sandbox sequenceiq/spark:1.6.0 -d`




















