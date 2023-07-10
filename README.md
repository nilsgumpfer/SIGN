# SIGNed explanations: Unveiling relevant features by reducing bias

This repository contains code to reproduce the experiments published in below journal article:
https://doi.org/10.1016/j.inffus.2023.101883

If you use the code from this repository in your work, please cite:
```bibtex
 @article{Gumpfer2023SIGN,
    title = {SIGNed explanations: Unveiling relevant features by reducing bias},
    author = {Nils Gumpfer and Joshua Prim and Till Keller and Bernhard Seeger and Michael Guckert and Jennifer Hannig},
    journal = {Information Fusion},
    pages = {101883},
    year = {2023},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2023.101883},
    url = {https://www.sciencedirect.com/science/article/pii/S1566253523001999}
}
```

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1566253523001999-ga1_lrg.jpg" title="Graphical Abstract" width="900px"/>

## Environment setup

### Virtual environment

 - To prepare your local environment, setup a virtual python environment in Anaconda or similar, using Python==3.6.9 or any other compatible version. 
 - Install all requirements for a CPU-based environment using ``` pip3 install -r requirements.txt ```. 
 - If you aim to use a GPU, modify ``` requirements.txt ``` and replace the CPU-based version of tensorflow with the GPU-based pendant in advance.

### Docker

 - To run the experiments in a GPU-powered docker environment, setup a docker image using the Dockerfile provided: ``` docker build -t SIGNenvImg:v1.0 . --no-cache ```. 
 - Afterwards, create a new container based oin this image: ``` docker run -it -d -v /path/to/repository:/SIGN --gpus all --name SIGNenv SIGNenvImg:v1.0 bash ```. 
 - To connect to the container, run: ``` docker exex -it SIGNenv bash ```. 
 - Inside the container, navigate to the linked directory: ``` cd /SIGN ``` .


## Datasets
To download the datasets, run ```bash download_datasets.sh```. Prior, make sure to request the required access rights to the databases listed below (see "Terms and conditions").

### ImageNet (ILSVRC2012)
[ImageNet database](https://image-net.org)

[Terms and contitions](https://image-net.org/download.php)

[Paper: International Journal of Computer Vision](https://doi.org/10.1007/s11263-015-0816-y)
```bibtex
 @article{ILSVRC15,
    author={Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    title={{ImageNet Large Scale Visual Recognition Challenge}},
    year={2015},
    journal={International Journal of Computer Vision (IJCV)},
    doi={10.1007/s11263-015-0816-y},
    volume={115},
    number={3},
    pages={211-252}
 }
```

### MIT Places 365
[Places2 database](http://places2.csail.mit.edu)

[Terms and contitions](http://places2.csail.mit.edu/download.html)

[Paper: IEEE Transactions on Pattern Analysis and Machine Intelligence](https://doi.org/10.1109/TPAMI.2017.2723009)
```bibtex
 @article{zhou2017places,
    author={Zhou, Bolei and Lapedriza, Agata and Khosla, Aditya and Oliva, Aude and Torralba, Antonio},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
    title={Places: A 10 Million Image Database for Scene Recognition}, 
    year={2018},
    volume={40},
    number={6},
    pages={1452-1464},
    doi={10.1109/TPAMI.2017.2723009}
 }
```

## VGG 16 Models
To download the pre-trained models, run ``` bash download_models.sh ```

The VGG16 architecture was proposed by Karen Simonyan and Andrew Zisserman in their 2015 paper:
[Paper: 3rd ICLR 2015 San Diego](http://arxiv.org/abs/1409.1556)
```bibtex
@inproceedings{Simonyan2015VGG16,
  author    = {Karen Simonyan and Andrew Zisserman},
  editor    = {Yoshua Bengio and Yann LeCun},
  title     = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
  booktitle = {3rd International Conference on Learning Representations, {ICLR} 2015,
               San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings},
  year      = {2015},
  url       = {http://arxiv.org/abs/1409.1556},
}
```

### VGG16 trained on ImageNet (ILSVRC2012)

VGG16 model and weights obtained from ``` tensorflow.python.keras.applications.vgg16 ```. We exported the model and uploaded it alongside with the model trained on MIT Places 365 for reasons of long-term reproducibility.
Reference:
https://keras.io, https://www.tensorflow.org/

### VGG16 trained on MIT Places 365

VGG16 model and weights obtained from [CSAILVision group's GitHub](https://github.com/CSAILVision/places365):

 - [Model](https://github.com/CSAILVision/places365/blob/master/deploy_vgg16_places365.prototxt)
 - [Weights](http://places2.csail.mit.edu/models_places365/vgg16_places365.caffemodel)

As the model originally had been trained in [Caffe](https://caffe.berkeleyvision.org/), 
models have been converted to [Keras](https://keras.io) H5 format using [MMdnn](https://github.com/microsoft/MMdnn).


## Experiments

To reproduce the experiments from our paper, simply run ``` python3 reproduce_results.py ```. Depending on your GPU-performance, this may take a substantial while. If you are interested only in partial results, feel free to comment out parts of the code. 

## Python package

We released a toolbox containing all XAI methods used in our paper as a stand-alone python package as well. You can install it via ```pip3 install signxai``` (further details: https://pypi.org/project/signxai/1.0.0/ ).