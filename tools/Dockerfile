FROM nvidia/cuda:9.2-devel-ubuntu18.04

LABEL maintainer="Bernd Doser <bernd.doser@h-its.org>"

ENV TZ 'Europe/Berlin'
ENV DEBIAN_FRONTEND noninteractive

RUN echo $TZ > /etc/timezone && \
  apt-get update && apt-get install -y tzdata && \
  rm /etc/localtime && \
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
  dpkg-reconfigure -f noninteractive tzdata && \
  apt-get clean

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-tk \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
    hash -r pip3 && \
    pip3 install -I \
    conan \
    jupyter \
    matplotlib \
    numpy \
    tensorflow \
    tqdm

RUN conan remote add braintwister https://api.bintray.com/conan/braintwister/conan && \
    conan remote add conan-community https://api.bintray.com/conan/conan-community/conan && \
    conan remote add bincrafters https://api.bintray.com/conan/bincrafters/public-conan

RUN git clone https://github.com/HITS-AIN/PINK.git && \
    cd PINK && git checkout redesign && \
    mkdir build && cd build && \
    cmake .. && \
    make

#RUN cd PINK/build && \
#    make test

ENV PYTHONPATH /PINK/build/lib
ENV JUPYTER_PATH /PINK/build/lib

COPY notebooks /notebooks

# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

WORKDIR /notebooks
EXPOSE 8888

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --notebook-dir=/notebooks --ip 0.0.0.0 --no-browser --allow-root"]
