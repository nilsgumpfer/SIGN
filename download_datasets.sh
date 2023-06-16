# Make dataset directories
mkdir -p ./data/datasets/ILSVRC2012val/
mkdir -p ./data/datasets/MITPLACES365val/
mkdir -p ./data/datasets/MNIST/

# Download MNIST dataset
wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
mv mnist.npz ./data/datasets/MNIST/mnist.npz

# Download and untar ImageNet validation dataset
# For access to the ImageNet dataset, please request access via https://www.image-net.org/download.php
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xvf ILSVRC2012_img_val.tar -C ./data/datasets/ILSVRC2012val/
rm ILSVRC2012_img_val.tar

# Download and untar MIT Places 365 validation dataset
# For access to the MIT Places 365 dataset, please request access via http://places2.csail.mit.edu/download.html
wget http://data.csail.mit.edu/places/places365/val_256.tar
tar -xvf val_256.tar
mv ./val_256/* ./data/datasets/MITPLACES365val/
rm -r ./val_256/
rm val_256.tar