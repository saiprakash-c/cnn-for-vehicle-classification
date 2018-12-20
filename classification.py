#!/usr/bin/python3

import time
import random
import os

# supress silly warnings from master branch of tensorflow...
# import sys
# sys.stderr = None

import matplotlib.image as mpimg
import skimage.transform

#import torch
#import torchvision
#from utils import Logger

import tensorflow as tf
from tensorflow import nn, layers
import numpy as np
from matplotlib.image import imread
import glob
import csv
from collections import defaultdict
from skimage import transform


# Set random seem for reproducibility
manualSeed = 999
np.random.seed(manualSeed)
#torch.manual_seed(manualSeed)
#tf.set_random_seed(manualSeed)

# make dictionary of lists of all bounding boxes
#bbox_file = "WIDER_train/wider_face_split/wider_face_train_bbx_gt.txt"

#min_image_size = 64
image_size = (224, 224, 3)

#making a dictionary for vgg19 values
data_dict = np.load('vgg19.npy', encoding='latin1').item()

columns = defaultdict(list)

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


#image_train = np.zeros((n_train,224,224,3))
labels = np.zeros(n_train)
#image_test = np.zeros((n_test,224,224,3))

for i in range(len(files_train)):
    #img = transform.resize(imread(files_train[i]),(224,224))
    #image_train[i,:,:,:] = img

    l2 = list(files_train[i].split('/')[0:-1])
    l3 = list(os.path.split(os.path.abspath(files_train[i])))
    l4 = list(l3[1].split('_')[0:-1])
    image_name = str(l2[-1] + '/' + l4[0])
    index_label = columns["guid/image"].index(image_name)
    list_label = columns["label"]
    labels[i] = list_label[index_label]


def batch_generator(e_images,e_labels,which_batch):

    if (which_batch+1)*batch_size > len(e_images):
        n_images = len(e_images) - which_batch*batch_size
    else :
        n_images = batch_size

    b_images = np.zeros((n_images,224,224,3))

    for j in range(n_images) :
        b_images[j,:,:,:] = skimage.transform.resize(imread(e_images[(which_batch-1)*batch_size + j]))

    b_labels = e_labels[(which_batch-1)*batch_size : (which_batch-1)*batch_size + n_images]

    return b_images, b_labels


batch_size = 10
num_batches = int(n_train / batch_size) +1  # approximately

def noise(size):
    return np.random.normal(size=size)

def discriminator(x):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        #vgg19_convolutional layers
        with tf.variable_scope("conv1"):
            conv1_1 = conv_layer(x, "conv1_1")
            conv1_2 = conv_layer(conv1_1, "conv1_2")
            pool1 = max_pool(conv1_2, "pool1")


        with tf.variable_scope("conv2"):
            conv2_1 = conv_layer(pool1, "conv2_1")
            conv2_2 = conv_layer(conv2_1, "conv2_2")
            pool2 = max_pool(conv2_2, "pool2")

        with tf.variable_scope("conv3"):
            conv3_1 = conv_layer(pool2, "conv3_1")
            conv3_2 = conv_layer(conv3_1, "conv3_2")
            conv3_3 = conv_layer(conv3_2, "conv3_3")
            conv3_4 = conv_layer(conv3_3, "conv3_4")
            pool3 = max_pool(conv3_4, "pool3")

        with tf.variable_scope("conv4"):
            conv4_1 = conv_layer(pool3, "conv4_1")
            conv4_2 = conv_layer(conv4_1, "conv4_2")
            conv4_3 = conv_layer(conv4_2, "conv4_3")
            conv4_4 = conv_layer(conv4_3, "conv4_4")
            pool4 = max_pool(conv4_4, "pool4")

        with tf.variable_scope("conv5"):
            conv5_1 = conv_layer(pool4, "conv5_1")
            conv5_2 = conv_layer(conv5_1, "conv5_2")
            conv5_3 = conv_layer(conv5_2, "conv5_3")
            conv5_4 = conv_layer(conv5_3, "conv5_4")
            #pool5 = max_pool(conv5_4, 'pool5')

        #fc6 = fc_layer(pool5, "fc6")
        #assert fc6.get_shape().as_list()[1:] == [4096]
        #relu6 = tf.nn.relu(fc6)

        #fc7 = fc_layer(relu6, "fc7")
        #relu7 = tf.nn.relu(fc7)
        with tf.variable_scope("linear"):
            linear = layers.flatten(conv5_4)
            y_ = layers.dense(linear, 4, use_bias=False, kernel_initializer=tf.initializers.random_normal(0.0, 0.1))

    return y_

def fc_layer(bottom, name):
    with tf.variable_scope(name):
        shape = bottom.get_shape().as_list()
        dim = 1
        for d in shape[1:]:
            dim *= d
        x = tf.reshape(bottom, [-1, dim])

        weights = get_fc_weight(name)
        biases = get_bias(name)

        # Fully connected layer. Note that the '+' operation automatically
        # broadcasts the biases.
        fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

        return fc

def avg_pool(bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def max_pool(bottom, name):
    return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

def conv_layer(bottom, name):
    with tf.variable_scope(name):
        filt = get_conv_filter(name)

        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

        conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, conv_biases)

        relu = tf.nn.relu(bias)
        return relu

def get_conv_filter(name):
    return tf.constant(data_dict[name][0], name="filter")

def get_fc_weight(self, name):
    return tf.constant(data_dict[name][0], name="weights")

def get_bias(name):
    return tf.constant(data_dict[name][1], name="biases")



# real input (full size)
X = tf.placeholder(tf.float32, shape=(None, ) + image_size)
# real labels (face vs non-face)
X_labels = tf.placeholder(tf.uint8, shape=(None, 1))
y = tf.one_hot(X_labels,4)

# Discriminator, has two outputs [face (1.0) vs nonface (0.0), real (1.0) vs generated (0.0)]

y_ = discriminator(X)

#node for accuracy
correct_prediction = tf.equal(tf.argmax[y_,1],tf.argmax[y,1])
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


#node for loss
D_loss = tf.reduce_mean(
    nn.softmax_cross_entropy_with_logits(
        logits=y_, labels = y))

# Obtain trainable variables for both networks
train_vars = tf.trainable_variables()

D_vars = [var for var in train_vars if 'discriminator' in var.name]

print("Discriminator parameter count: {}".format(np.sum([np.product(v.get_shape()) for v in D_vars])))

learning_rate = tf.placeholder(tf.float32, shape=[])
D_opt = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=D_vars)

#num_test_samples = 25
#_, (test_batch, test_small_images, test_labels) = next(batch_generator(num_test_samples))

"""Logging"""

"""
# Create logger instance
logger = Logger(model_name='Log')

# Write out ground truth examples, and "dumb" upscaled examples
logger.log_images(
    test_batch, num_test_samples,
    -100, 0, num_batches
)

"""

# Total number of epochs to train
num_epochs = 10

# Start interactive session
session = tf.InteractiveSession()
# Init Variables
tf.global_variables_initializer().run()

saver = tf.train.Saver()

# Initial SR training by itself
batch_start_time = time.time()

for epoch in range(2):
    lr = 1e-4

    if epoch > 0:
        saver.save(session, "./model_{}.ckpt".format(epoch))

    #randomize the files
    d_indices = range(n_train)
    random.shuffle(d_indices)
    e_images = files_train[d_indices]
    e_labels = labels[d_indices]

    for i in range(num_batches):
        #get the b_images and b_labels
        n_batch = i+1
        b_images,b_labels = batch_generator(e_images,e_labels,n_batch)

        #Train Discriminator
        feed_dict = {X: b_images, X_labels: b_labels, learning_rate: lr}
        _,Loss, Accuracy = session.run([D_opt, D_loss, accuracy], feed_dict=feed_dict)

        # Display Progress every few batches
        if n_batch % 2 == 0:
            #print the time taken by the bathces
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time

            print("Batches took {:.3f} ms".format(elapsed * 1000))
            print("epoch:",epoch/num_epochs,"n_batches:",n_batch/num_batches, "Loss:", Loss,
                  "Accuracy:" ,Accuracy)


            """
            test_images = session.run(out, feed_dict={X:test_images})
            test_images = (test_images + 1.0) * 0.5

            logger.log_images(
                test_images, num_test_samples,
                epoch-10, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                -1, D_error, -1, [-1], [-1]
            )
            
            """

saver.save(session, "./model_full.ckpt")






"""
real_face_sum = 0
fake_face_sum = 0
total_sum = 0

batch_start_time = time.time()
for epoch in range(num_epochs):
    if epoch > 0:
        saver.save(session, "./model_{}.ckpt".format(epoch))

    lr = 1e-4 if epoch < 5 else 1e-5

    batch_gen = batch_generator(batch_size)
    for n_batch, (real_images, small_images, real_labels) in batch_gen:
        # 1. Train Discriminator
        feed_dict = {X: real_images, X_labels: real_labels, Z: small_images, learning_rate: lr}
        _, d_error, d_pred_real, d_pred_fake, d_real_face, d_fake_face = session.run([D_opt, D_loss, D_real, D_fake, D_real_face, D_fake_face], feed_dict=feed_dict)

        real_face_sum += np.sum(d_real_face.round() == real_labels)
        fake_face_sum += np.sum(d_fake_face.round() == real_labels)
        total_sum += len(real_images)

        # 2. Train Generator
        feed_dict = {X: real_images, X_labels: real_labels, Z: small_images, learning_rate: lr}
        _, g_error = session.run([G_opt, G_loss], feed_dict=feed_dict)

        # Display Progress every few batches
        if n_batch % 2 == 0:
            now_time = time.time()
            elapsed = now_time - batch_start_time
            batch_start_time = now_time
            print("Batches took {:.3f} ms".format(elapsed * 1000))

            test_images = session.run(G_sample2, feed_dict={Z: test_small_images})
            test_images = (test_images + 1.0) * 0.5

            logger.log_images(
                test_images, num_test_samples,
                epoch, n_batch, num_batches
            )
            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, 0, d_pred_real, d_pred_fake
            )

            print("Real accuracy ({}/{}) {:.2f}%    Fake accuracy ({}/{}) {:.2f}%".format(real_face_sum, total_sum, real_face_sum / total_sum * 100,
                                                                                          fake_face_sum, total_sum, fake_face_sum / total_sum * 100,))
saver.save(session, "./model_full.ckpt")
"""