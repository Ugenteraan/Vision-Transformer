FROM nvidia/cuda:12.2.0-base-ubuntu20.04


RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y apt-utils
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 \
  libfontconfig1 libxext6 libgl1-mesa-glx > /dev/null

RUN apt install python3 python3-pip -y
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install numpy
RUN pip3 install Cython scipy pandas seaborn
RUN pip3 install Pillow torchvision
RUN pip3 install "deeplake[enterprise]"
RUN pip3 install einops


COPY . /home

WORKDIR /home