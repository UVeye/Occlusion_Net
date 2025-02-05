ARG CUDA="10.0"
ARG CUDNN="7"

#FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04

# Base Image: CUDA 10.0, cuDNN 7, Ubuntu 16.04
FROM sulfurheron/nvidia-cuda:10.0-cudnn7-devel-ubuntu16.04-2019-07-29

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# Install system dependencies
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++-5 \
 && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev vim feh wget xterm \
 software-properties-common \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update -y && apt-get install -y \
 cmake ninja-build \
 openssh-server && \
 mkdir /var/run/sshd && \
 echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
 echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config && \
 apt-get clean && rm -rf /var/lib/apt/lists/*

# Set a default root password (Please change in production!)
RUN echo "root:rootpassword" | chpasswd

# Expose SSH port
EXPOSE 22

# Install Miniconda
RUN wget -O /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

# Set Conda environment variables
ENV PATH=/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN conda install -y conda-build && \
    conda create -y --name py36 python=3.6.7 && \
    conda clean -ya

ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib opencv-python-headless==3.4.10.37 tqdm scikit-learn comet_ml shapely pandas

# Install PyTorch 1.1.0 (CUDA 10.0)
RUN conda install -y pytorch=1.1.0 torchvision=0.3.0 cudatoolkit={CUDA} -c pytorch && conda clean -ya

## Install PyTorch via pip (ensures correct CUDA build)
## RUN pip install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
#RUN pip install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu100/stable.html  # This one works

# Install TorchVision (v0.2.2)
RUN git clone --single-branch --branch v0.2.2_branch https://github.com/pytorch/vision.git && \
    cd vision && python setup.py install

# Install pycocotools (for COCO dataset)
RUN git clone https://github.com/cocodataset/cocoapi.git && \
    cd cocoapi/PythonAPI && git checkout aca78bcd6b4345d25405a64fdba1120dfa5da1ab && \
    python setup.py build_ext install

# Install NVIDIA Apex (for mixed-precision training)
RUN git clone https://github.com/NVIDIA/apex.git \
 && cd apex && git checkout 4ff153cd50e4533b21dc1fd97c0ed609e19c4042 \
 && python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection (Mask R-CNN Benchmark)
ARG FORCE_CUDA="1"
ENV FORCE_CUDA=${FORCE_CUDA}
RUN git clone https://github.com/facebookresearch/maskrcnn-benchmark.git \
 && cd maskrcnn-benchmark && git checkout a44d65dcdb9c9098a25dd6b23ca3c36f1b887aba\
 && python setup.py build develop

WORKDIR /code

# Start SSH service on container start
CMD ["/usr/sbin/sshd", "-D"]
