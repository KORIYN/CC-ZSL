# Discriminative and Robust Attribute Alignment for Zero-Shot Learning

## Overview
This repository is the official pytorch implementation of Discriminative and Robust Attribute Alignment for Zero-Shot Learning.  

## Requirements


Experiments were done with the following package versions for Python 3.6:

- PyTorch 1.9.0+cu111 with CUDA 11.4
- h5py          3.1.0                   
- matplotlib      3.1.0                    
- numpy                     1.19.5                 
- opencv-python             4.5.4.60                 
- pillow                    8.4.0                 
- python                    3.6.13             
- scikit-image              0.16.2                 
- scikit-learn              0.24.2                   
- scipy                     1.5.4                  
- setproctitle              1.2.2                   
- sklearn                   0.0                      
- tensorboard               1.9.0                 
- tensorflow-gpu            1.9.0                    
- torchvision               0.10.0+cu111             
- tqdm                      4.43.0                



## Data Preparation

 Please download and data into the `./data folder.` We show the details about download links.  


## Train and Test

For different datasets (AWA2/CUB/SUN), you can run the code:

```
sh CUB_GZSL.sh
sh AWA2_GZSL.sh
sh SUN_GZSL.sh

``` 
