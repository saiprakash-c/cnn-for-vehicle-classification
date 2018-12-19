from matplotlib.image import imread
import glob
import pandas as pd
import os
import csv
from collections import defaultdict


def preprocessing():
    columns = defaultdict(list)  # each value in each column is appended to a list

    with open('/home/sai/PycharmProjects/AV/labels.csv') as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                columns[k].append(v)

    path_train = '/home/sai/PycharmProjects/AV/vehicle_class/deploy/trainval/*/*.jpg'
    files_train = glob.glob(path_train)
    print(len(files_train))

    path_test = '/home/sai/PycharmProjects/AV/vehicle_class/deploy/test/*/*.jpg'
    files_test = glob.glob(path_test)
    print(len(files_test))

    image_train = []
    labels = []
    image_test = []
    for i in range(len(files_train)):
        img = imread(files_train[i])
        image_train.append(img)

        l2 = list(files_train[i].split('/')[0:-1])
        l3 = list(os.path.split(os.path.abspath(files_train[i])))
        l4 = list(l3[1].split('_')[0:-1])
        image_name = str(l2[-1] + '/' + l4[0])
        index_label = columns["guid/image"].index(image_name)
        list_label = columns["label"]

        labels.append(list_label[index_label])

    for i in range(len(files_test)):
        img = imread(files_test[i])
        image_test.append(img)

    return (labels, image_train, image_test)


labels, image_train, image_test = preprocessing()
