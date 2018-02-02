FROM bernddoser/docker-devel-cpp:cuda-9.1-devel-cmake-3.10-gtest-1.8.0-doxygen-1.8.14

MAINTAINER Bernd Doser <bernd.doser@h-its.org>

RUN apt-get update \
 && apt-get install -y \
    libboost-all-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
