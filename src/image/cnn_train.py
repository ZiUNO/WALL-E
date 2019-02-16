# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/11/26 15:57
"""
import os
import time

import tensorflow as tf

from utils import jpg_handle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

size_12 = size_21 = 64
size_23 = size_32 = 128
size_3 = 64
size_attri_12 = size_attri_21 = 768
size_attri_2 = 384
size_label_12 = size_label_21 = 320
size_label_2 = 640

max_step = 1000
batch_size = 32
attribute_holder = tf.placeholder(tf.float32, [batch_size, 30])
image_holder = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
label_holder = tf.placeholder(tf.int32, [batch_size, 230])

with tf.name_scope('Inputs'):
    train_iterator = jpg_handle.get_iterator(
        r'G:\AI\zero\DatasetA_train_20180813\DatasetA_train_20180813', type='train', batch_size=batch_size)
    image_train, attributes_train, labels_train = train_iterator.get_next()
    # test_iterator = jpg_handle.get_iterator(r'G:\AI\zero\DatasetA_test_20180813\DatasetA_test_20180813', type='test',
    #                                         batch_size=batch_size)
    # image_test, attributes_test, labels_test = test_iterator.get_next()
# attribute_list = jpg_handle.get_attribute_list(
#     r'G:\AI\zero\DatasetA_train_20180813\DatasetA_train_20180813\attribute_list.txt')
# label_list = jpg_handle.get_label_list(r'G:\AI\zero\DatasetA_train_20180813\DatasetA_train_20180813\label_list.txt')

"""
Tensorflow network structure
"""
with tf.name_scope('Conv1'):
    kernel_size_1 = 5
    weight_1 = tf.Variable(tf.truncated_normal([kernel_size_1, kernel_size_1, 3, size_12], stddev=5e-2))
    kernel_1 = tf.nn.conv2d(image_holder, weight_1, [1, 1, 1, 1], padding='SAME')
    bias_1 = tf.Variable(tf.constant(0.0, shape=[size_12]))
    conv_1 = tf.nn.relu(tf.nn.bias_add(kernel_1, bias_1))
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    norm_1 = tf.nn.lrn(pool_1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

with tf.name_scope('Conv2'):
    kernel_size_2 = 5
    weight_2 = tf.Variable(tf.truncated_normal([kernel_size_2, kernel_size_2, size_21, size_23], stddev=5e-2))
    kernel_2 = tf.nn.conv2d(norm_1, weight_2, [1, 1, 1, 1], padding='SAME')
    bias_2 = tf.Variable(tf.constant(0.1, shape=[size_23]))
    conv_2 = tf.nn.relu(tf.nn.bias_add(kernel_2, bias_2))
    norm_2 = tf.nn.lrn(conv_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool_2 = tf.nn.max_pool(norm_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope('Conv3'):
    kernel_size_3 = 5
    weight_3 = tf.Variable(tf.truncated_normal([kernel_size_3, kernel_size_3, size_32, size_3], stddev=5e-2))
    kernel_3 = tf.nn.conv2d(pool_2, weight_3, [1, 1, 1, 1], padding='SAME')
    bias_3 = tf.Variable(tf.constant(0.0, shape=[size_3]))
    conv_3 = tf.nn.relu(tf.nn.bias_add(kernel_3, bias_3))
    norm_3 = tf.nn.lrn(conv_3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    pool_3 = tf.nn.avg_pool(norm_3, ksize=[1, 3, 3, 1], strides=[1,2, 2, 1], padding='SAME')

with tf.name_scope('FCA1'):
    reshape = tf.reshape(pool_3, [batch_size, -1])
    dim = reshape.get_shape()[1].value
    weight_attri_1 = tf.Variable(tf.truncated_normal([dim, size_attri_12], stddev=0.04))
    bias2_attri = tf.Variable(tf.constant(0.1, shape=[size_attri_12]))
    local_attri_1 = tf.nn.relu(tf.matmul(reshape, weight_attri_1) + bias2_attri)

with tf.name_scope('FCA2'):
    weight_attri_2 = tf.Variable(tf.truncated_normal([size_attri_21, size_attri_2], stddev=0.04))
    bias_attri_2 = tf.Variable(tf.constant(0.1, shape=[size_attri_2]))
    local_attri_2 = tf.nn.relu(tf.matmul(local_attri_1, weight_attri_2) + bias_attri_2)

with tf.name_scope('LogitsA'):
    weight_attri_logits = tf.Variable(tf.truncated_normal(shape=[size_attri_2, 30], stddev=1 / float(size_attri_2)))
    bias4_attri = tf.Variable(tf.constant(0.0, shape=[30]))
    logits_attri = tf.add(tf.matmul(local_attri_2, weight_attri_logits), bias4_attri)

with tf.name_scope('FCL1'):
    weight_label_1 = tf.Variable(tf.truncated_normal([30, size_label_12], stddev=1 / 30.0))
    bias_label_1 = tf.Variable(tf.constant(0.0, shape=[size_label_12]))
    # local_label_1 = tf.nn.relu(tf.add(tf.matmul(logits_attri, weight_label_1), bias_label_1))
    # 更改为属性的holder进行测试
    local_label_1 = tf.nn.relu(tf.add(tf.matmul(attribute_holder, weight_label_1), bias_label_1))

with tf.name_scope('FCL2'):
    weight_label_2 = tf.Variable(tf.truncated_normal([size_label_21, size_label_2], stddev=1 / 30.0))
    bias_label_2 = tf.Variable(tf.constant(0.0, shape=[size_label_2]))
    local_label_2 = tf.nn.relu(tf.add(tf.matmul(local_label_1, weight_label_2), bias_label_2))

with tf.name_scope('LogitL'):
    weight_logits_label = tf.Variable(tf.truncated_normal([size_label_2, 230], stddev=1 / float(size_label_12)))
    bias_logits_label = tf.Variable(tf.constant(0.0, shape=[230]))
    logits_label = tf.add(tf.matmul(local_label_2, weight_logits_label), bias_logits_label)

with tf.name_scope('MeanA'):
    cross_entropy_attribute = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_attri, labels=attribute_holder)
    loss_attribute = tf.reduce_mean(cross_entropy_attribute, name='loss_attribute')
    loss_attribute_w = tf.multiply(tf.nn.l2_loss(loss_attribute), 0.3, name='w_attribute')
    tf.add_to_collection('loss', loss_attribute_w)

with tf.name_scope('MeanL'):
    cross_entropy_label = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits_label, labels=label_holder)
    loss_label = tf.reduce_mean(cross_entropy_label, name='loss_label')
    tf.add_to_collection('loss', loss_label)

with tf.name_scope('Loss'):
    loss = tf.add_n(tf.get_collection('loss'))
    tf.summary.scalar('loss', loss)

train_op = tf.train.AdamOptimizer(0.05).minimize(loss)
with tf.name_scope('Prediction'):
    correct_prediction = tf.equal(tf.argmax(label_holder, 1), tf.argmax(logits_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# top_k_op = tf.nn.in_top_k(logits_label, label_holder, 1)

sess = tf.InteractiveSession()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logs', sess.graph)
tf.global_variables_initializer().run()
sess.run(train_iterator.initializer)
# sess.run(test_iterator.initializer)

for step in range(max_step):
    image_batch, attribute_batch, label_batch = sess.run([image_train, attributes_train, labels_train])
    if step % 10 == 0:
        accu = accuracy.eval(
            {image_holder: image_batch, attribute_holder: attribute_batch, label_holder: label_batch})
        print('accuracy before = %.3f, ' % accu, end='')
    start_time = time.time()
    _, loss_value, loss_a_value, loss_l_value = sess.run([train_op, loss, loss_attribute, loss_label],
                                                         feed_dict={image_holder: image_batch,
                                                                    attribute_holder: attribute_batch,
                                                                    label_holder: label_batch})
    duration = time.time() - start_time
    if step % 10 == 0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = 'step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch), loss_a = %.2f, loss_l = %.2f'
        result = sess.run(merged,
                          {image_holder: image_batch, attribute_holder: attribute_batch, label_holder: label_batch})
        writer.add_summary(result, step)
        accu = accuracy.eval(
            {image_holder: image_batch, attribute_holder: attribute_batch, label_holder: label_batch})
        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch, loss_a_value, loss_l_value),
              ', accuracy after = %.3f' % accu)
        # print(format_str % (step, loss_value, examples_per_sec, sec_per_batch, loss_a_value, loss_l_value))
writer.close()
# for i in range(1000):
#     image_batch, attribute_batch, label_batch = image_train, attributes_train, labels_train
#     train_op.run({image_holder: image_batch, attribute_holder: attribute_batch, label_holder: label_batch})
#     print(i)
