# -*- coding: utf-8 -*-
"""
* @Author: ziuno
* @Software: PyCharm
* @Time: 2018/11/27 11:33
"""

import os
import numpy as np

import tensorflow as tf


def get_attribute_list(attribute_list_filename):
    """
    获取属性列表（与attribute_per_class中属性顺序相同）（30）
    :param attribute_list_filename: 属性列表文件名
    :return: 属性列表（含顺序）
    """
    with open(attribute_list_filename, 'r') as f:
        attribute_list = f.readlines()
    attribute_list = [attribute.strip().split('\t')[1] for attribute in attribute_list]
    return attribute_list


def get_label_list(label_list_filename):
    """
    获取标签列表（与label顺序相同）（230）
    :param label_list_filename:
    :return: 标签列表（含顺序）（按ZJL排序）
    """
    with open(label_list_filename, 'r') as f:
        label_list = f.readlines()
    label_list = [label.strip().split('\t') for label in label_list]
    label_list = sorted(label_list, key=(lambda x: x[0]))
    label_list = [label[1] for label in label_list]
    return label_list


def __read_data(data_dir, type):
    """
    读取文件夹内相应的训练集或测试集图片、属性和标签，生成相应格式数据
    :param data_dir: 数据文件夹
    :param type: 数据集的类型（train或test）
    :return: 图片集， 属性集， 标签集（三者均为概率格式）
    """
    if type not in ['train', 'test']:
        raise ValueError('wrong type')
    image_dir_name = data_dir + os.sep + type
    attribute_per_class_filename = data_dir + os.sep + 'attributes_per_class.txt'
    train_filename = data_dir + os.sep + 'train.txt'
    label_list_filename = data_dir + os.sep + 'label_list.txt'
    with open(attribute_per_class_filename, 'r') as f:
        attribute_per_class = f.readlines()
    attribute_per_class = [attribute.strip().split('\t') for attribute in attribute_per_class]
    attribute_per_class = dict([(attribute[0], attribute[1:]) for attribute in attribute_per_class])
    with open(label_list_filename, 'r') as f:
        label_list = f.readlines()
    label_list = [label.strip().split('\t') for label in label_list]
    label_list = [label[0] for label in label_list]  # label_list ZJL格式
    label_list = sorted(label_list)
    with open(train_filename, 'r') as f:
        train = f.readlines()
    train = [(train[index].strip().split('\t')) for index in range(len(train))]
    train = sorted(train, key=(lambda x: x[0]))
    label = [item[1] for item in train]
    for index in range(len(label)):
        label_of_index = label[index]
        label_ = np.zeros(230, dtype=np.int32)
        index_of_label = label_list.index(label_of_index)
        label_[index_of_label] = 1
        label[index] = list(label_)  # label: 所有图片对应的230中的概率（相应的为1）（含顺序）
    attribute = [[float(tmp) for tmp in attribute_per_class[item[1]]] for item in train]  # attribute: 所有图片对应的30中属性概率
    image_names = os.listdir(image_dir_name)
    image_names = sorted(image_names)
    image_names = [image_dir_name + os.sep + image_name for image_name in image_names]  # image_name: 所有图片名称（含顺序）（绝对路径）
    return image_names, attribute, label


def __map_fn(image, attribute, label):
    """
    tf.data.Dataset.map函数专用
    :param image: 图片名集合（tensor）
    :param attribute: 属性集（tensor）（不处理）
    :param label: 标签集（tensor）（不处理）
    :return: 处理后的三种格式的数据
    """
    image_string = tf.read_file(image)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])
    return image_resized, attribute, label


def __read_images(images_names, attribute, label):
    """
    读取图片以及相应的属性集和标签集
    :param images_names: 图片名集合
    :param attribute: 属性集
    :param label: 标签集
    :return: 数据集
    """
    dataset = tf.data.Dataset.from_tensor_slices((images_names, attribute, label))
    dataset = dataset.map(__map_fn)
    return dataset


def train_inputs(data_dir, batch_size):
    """
    训练数据输入
    :param data_dir:数据集路径（内含list等）
    :param batch_size: 每批数据大小
    :return: image_train, attribute_train, label_train
    """
    image_names, attribute, label = __read_data(data_dir, type='train')
    image_names = [os.path.join(tmp) for tmp in image_names]
    read_input = __read_images(image_names, attribute, label)
    read_input.batch(batch_size=batch_size)
    read_input.shuffle(buffer_size=1000)
    iterator = read_input.make_initializable_iterator()
    return iterator.get_next()

def test_inputs(data_dir):
    image_names, attribute, label = __read_data(data_dir, 'test')
    image_names = [os.path.join(tmp) for tmp in image_names]
    read_input = __read_images(image_names, attribute, label)
    read_input.shuffle(buffer_size=500)
    iterator  = read_input.make_initializable_iterator()
    return iterator.get_next()

batch_size = 128
image_train, attribute_train, label_train = train_inputs(r'G:\AI\zero\DatasetA_train_20180813\DatasetA_train_20180813',
                                                         batch_size=batch_size)
print(image_train, attribute_train, label_train)
input("success")
