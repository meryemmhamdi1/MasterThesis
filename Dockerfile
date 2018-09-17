FROM ubuntu:16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
MAINTAINER Athanasios Giannakopoulos

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 && \
    NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub && \
    apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub && \
    echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - && rm cudasign.pub && \
    echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDA_VERSION 9.0.176

ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# nvidia-docker 1.0
LABEL com.nvidia.volumes.needed="nvidia_driver"
LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Added
RUN apt-get update
RUN apt-get -y install python-setuptools python-dev build-essential swig wget
RUN apt-get -y install python-pip

#RUN easy_install pip

RUN pip install --upgrade pip
RUN pip install beautifulsoup4
RUN pip install keras
RUN pip install tensorflow
RUN pip install nltk
RUN pip install cython
RUN pip install tqdm
RUN pip install sklearn
RUN pip install theano
RUN pip install keras
RUN pip install numpy 
RUN pip install pandas
RUN pip install h5py
RUN pip install gensim
RUN pip install networkx
RUN pip install -U spacy
RUN python -m spacy download en
RUN python -m spacy download de
RUN python -m spacy download fr
RUN python -m spacy download it
RUN python -m spacy download pt
RUN python -m spacy download es
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
###########


ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

## user dependencies ##

RUN apt-get update
RUN apt-get -y install git

RUN apt-get -y install build-essential checkinstall
RUN apt-get -y install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
RUN apt -y dist-upgrade

RUN apt-get -y install python2.7 python-pip
RUN apt-get -y install python3

RUN apt-get -y install curl wget

RUN curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
RUN bash Anaconda3-5.0.1-Linux-x86_64.sh -b
RUN rm Anaconda3-5.0.1-Linux-x86_64.sh

RUN apt-get -y install vim nano htop screen iputils-ping

RUN mkdir -p /aimlx

WORKDIR /

ENTRYPOINT ["/bin/bash"]

ENV LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8
