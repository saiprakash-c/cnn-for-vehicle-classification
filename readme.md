This is made as part of course project for Self Driving Cars at umich. 

## To clone : type

git clone https://www.github.com/saiprakash-c/vehicle_class

## Files to be run

Training - classification.py
- Requires trainval folder, labels.csv and vgg19.npy (weights of vgg19)

- Stores .ckpt files in the same folder for each batch and each epoch

Testing - testing.py 

- Requires test folder, vgg19.npy

- Requires .ckpt file. It takes "model_6.ckpt" by default

- Stores submission2.csv in the same folder which can be used to submit in kaggle

## Packages needed:
Numpy

Tensorflow

scikit-image

# Download
The data for this can be downloaded at https://www.kaggle.com/c/fall2018-rob535-task1/data

Define the paths for trainval, test and labels.csv or
Keep trainval,test folders and labels.csv just outside the folder which contains the python files.

Download vgg19.npy weights from https://mega.nz/#!xZ8glS6J!MAnE91ND_WyfZ_8mvkuSa2YcA7q-1ehfSm-Q1fxOvvs and keep it in the folder outside of the main pthon files.

