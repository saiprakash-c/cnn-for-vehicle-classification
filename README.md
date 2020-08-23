# CNN for vehicle classification

This project is part of ROB535 - Self-driving car course at the University of Michigan Ann Arbor. 

A self-driving car needs to identify the type of vehicles in its vicinity to appropriately plan its motion. We implemented a Convolutional Neural Network to solve this problem. 

The training data consists of 7573 labelled images and the testing data consits of 2631 unlabelled images. The objective is to classify each testing image into one of the 23 categories, which correspond to the type of vehicle present in the image. The data for this can be downloaded at https://www.kaggle.com/c/fall2018-rob535-task1/data

We used VGG19 network as a feature extractor for this task as it is one of the most successful neural networks for classification is VGG19.
It is a simple convolutional neural network, unlike others like GoogleNet,AlexNet etc., but gave an error rate of less
than 10% in ImageNet challenge, which contains 1.2 million images with 1000 categories. 

During the back propagation, we only updated the weights of last fully connected layers of VGG19 network.  

## Prerequisites

Make sure the following libraries are installed 

1. Python 3.x
2. Numpy >=1.15
3. Tensorflow >=1.12
4. scikit-image >=0.14.1


## Installing and running

git clone https://www.github.com/saiprakash-c/cnn-for-vehicle-classification

Download the data from www.kaggle.com/c/fall2018-rob535-task1/data and keep it in the appropriate folders as follow. 

Pretrained model of VGG19 is obtained from [here](https://mega.nz/file/xZ8glS6J#MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs)

### Training

- Keep trainval folder, labels.csv and vgg19.npy (weights of vgg19) in the directory above the main directory
- Run classification.py
- Stores tensorflow checkpoint files for each batch and each epoch

### Testing

- Keep test folder, vgg19.npy in the directory above main directory
- Takes "model_6.ckpt" by default
- Stores submission.csv which can be used to submit in kaggle

## Contact

If you have any questions, contact me at saip@umich.edu

