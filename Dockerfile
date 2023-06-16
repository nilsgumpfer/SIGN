# Use pre-installed nvidia runtime
FROM nvcr.io/nvidia/tensorflow:20.06-tf2-py3

# Tell debian that no interactive frontend is available
ARG DEBIAN_FRONTEND=noninteractive

# Run some default installations
RUN apt-get update
RUN apt-get install -y apt-transport-https
RUN apt-get install -y systemd
RUN apt-get install -y nano
RUN apt-get install -y git
RUN apt-get install -y curl
RUN apt-get install -y libcurl4-openssl-dev libssl-dev
RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN apt-get install -y texlive-full
RUN pip3 install opencv-python==4.5.3.56
RUN pip3 install opencv-contrib-python==4.5.5.64
RUN pip3 install opencv-python-headless==4.1.2.30

# Add line to .bashrc to set PYTHONPATH on each login
RUN echo 'export PYTHONPATH='..:.:../..'' >> ~/.bashrc

# Git credential config
RUN git config --global credential.helper store

# After installation, run pip3 install -r requirements.txt manually, but remove the tensorflow entry first if using docker