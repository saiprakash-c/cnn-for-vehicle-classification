from matplotlib.image import imread
import glob
import os
import csv
from collections import defaultdict
from skimage import transform
import numpy as np



columns = defaultdict(list)  # each value in each column is appended to a list

with open('/home/sai/PycharmProjects/AV/labels.csv') as f:
    reader = csv.DictReader(f)  # read rows into a dictionary format
    for row in reader:  # read a row as {column1: value1, column2: value2,...}
        for (k, v) in row.items():  # go over each column name and value
            columns[k].append(v)

path_train = '/home/sai/PycharmProjects/AV/deploy/trainval/*/*.jpg'
files_train = glob.glob(path_train)
print(type(files_train))
n_train = len(files_train)

path_test = '/home/sai/PycharmProjects/AV/deploy/test/*/*.jpg'
files_test = glob.glob(path_test)
n_test = len(files_test)

batch_size = 500


image_train = np.zeros((n_train,224,224,3))
labels = np.zeros(n_train)
image_test = np.zeros((n_test,224,224,3))



for i in range(5):
    img = transform.resize(imread(files_train[i]),(224,224))
    image_train[i,:,:,:] = img

    l2 = list(files_train[i].split('/')[0:-1])
    l3 = list(os.path.split(os.path.abspath(files_train[i])))
    l4 = list(l3[1].split('_')[0:-1])
    image_name = str(l2[-1] + '/' + l4[0])
    index_label = columns["guid/image"].index(image_name)
    list_label = columns["label"]
    labels[i] = list_label[index_label]

print(image_train.shape)
print(labels)
print(len(columns["guid/image"]))


for i in range(len(files_test)):
    img = transform.resize(imread(files_test[i]),(224,224))
    image_test[i,:,:,:] = img

    l2 = list(files_test[i].split('/')[0:-1])
    l3 = list(os.path.split(os.path.abspath(files_test[i])))
    l4 = list(l3[1].split('_')[0:-1])
    image_name = str(l2[-1] + '/' + l4[0])



