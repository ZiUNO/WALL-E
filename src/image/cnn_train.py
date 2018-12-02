# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/11/26 15:57
"""

import tensorflow as tf

from utils import data_input

batch_size = 128
attribute_holder = tf.placeholder(tf.float32, [batch_size, 30])
image_holder = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
label_holder = tf.placeholder(tf.string, [batch_size])


def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var


image_train, labels_train, attributes_train = data_input.train_inputs(
    r'G:\AI\zero\DatasetA_train_20180813\DatasetA_train_20180813', batch_size=batch_size)

weight1 = variable_with_weight_loss(shape=[5, 5, 3, 128], stddev=5e-2, wl=0.0)
kernel1 = tf.nn.conv2d(image_holder, weight1, [1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[128]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

reshape = tf.reshape(pool1, [batch_size, -1])
dim = reshape.get_shape()[1].value
weight2 = variable_with_weight_loss(shape=[dim, 768], stddev=0.04, wl=0.004)
bias2 = tf.Variable(tf.constant(0.1, shape=[768]))
local2 = tf.nn.relu(tf.matmul(reshape, weight2) + bias2)

weight3 = variable_with_weight_loss(shape=[768, 384], stddev=0.04, wl=0.004)
bias3 = tf.Variable(tf.constant(0.1, shape=[384]))
local3 = tf.nn.relu(tf.matmul(local2, weight3) + bias3)

weight4 = variable_with_weight_loss(shape=[384, 230], stddev=1/384.0, wl=0.0)
bias4 = tf.Variable(tf.constant(0.0, shape=[230]))
logits = tf.add(tf.matmul(local3, weight4), bias4)
