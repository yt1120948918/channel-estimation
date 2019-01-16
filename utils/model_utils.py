import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from tensorflow import logging


def conv_weight_init(dimension, name, stride=None, mode="Xavier"):
    # dimension维度是[filter_size_height, filter_size_weight, in_channels, out_channels]
    # stride维度是[1, stride, stride, 1]
    if mode == "Xavier":
        fan_in = dimension[2] * dimension[0] * dimension[1]
        fan_out = dimension[3] * (dimension[0] / stride[1]) * (dimension[1] / stride[2])
        low = -np.sqrt(6. / (fan_in + fan_out))
        high = np.sqrt(6. / (fan_in + fan_out))
        return tf.Variable(tf.random_uniform(dimension, minval=low, maxval=high, dtype=tf.float32), name=name)
    elif mode == "normal":
        return tf.Variable(tf.random_normal(dimension), name=name)


def linear_weight_init(dimension, name, mode="Xavier"):
    # dimension的维度是[input_size, output_size]
    if mode == "Xavier":
        low = -np.sqrt(6. / (dimension[0] + dimension[1]))
        high = np.sqrt(6. / (dimension[0] + dimension[1]))
        return tf.Variable(tf.random_uniform(dimension, minval=low, maxval=high, dtype=tf.float32), name=name)
    elif mode == "normal":
        return tf.Variable(tf.random_normal(dimension), name=name)


def bias_init(dimension, name):
    # dimension的维度是[out_channels]
    return tf.Variable(tf.random_normal(dimension), name=name)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def avg_pool(inputs, filter_size, strides, padding, name):
    with tf.variable_scope(name):
        # inputs -- [None, height, weight, channels]
        # filter_size -- [1, height, weight, 1]
        # strides -- [1, weight, height, 1]
        # padding -- "VALID"或者"SAME"
        return tf.nn.avg_pool(inputs, filter_size, strides, padding, name=name + "_avg_pooling")


def conv_layer(inputs, input_dim, output_dim, filter_size,
               strides, padding, normal_type, is_training, name):
    with tf.variable_scope(name):
        fils = conv_weight_init([filter_size[0], filter_size[1], input_dim, output_dim],
                                name="weights", stride=strides, mode="Xavier")
        biases = bias_init([output_dim], name="biases")

        # weight normalization作用于模型参数权值，因此是加在卷积计算前面
        # weight normalization的实现代码参考了WGAN-GP的源代码
        if normal_type == "weight normalization":
            norm_values = tf.sqrt(tf.reduce_sum(tf.square(fils), axis=[0, 1, 2]))
            with tf.variable_scope("weight_norm"):
                norms = tf.sqrt(tf.reduce_sum(tf.square(fils), reduction_indices=[0, 1, 2]))
                fils = fils * (norm_values / norms)

        conv = tf.nn.conv2d(inputs, fils, strides, padding=padding)
        result = tf.nn.bias_add(conv, biases)

        # 在WGAN-GP论文中指出不要用batch normalization
        # 可以用其他normalization方法如layer normalization、weight normalization、instance normalization代替
        # 论文中推荐使用layer normalization
        if normal_type == "layer normalization":
            result = slim.layer_norm(result,
                                     reuse=True,
                                     scope="layer_norm")
        elif normal_type == "instance normalization":
            result = slim.instance_norm(result,
                                        reuse=True,
                                        scope="instance_norm")
        elif normal_type == "batch normalization":
            result = slim.batch_norm(result,
                                     center=True,
                                     scale=True,
                                     is_training=is_training,
                                     scope="batch_norm")
        elif not normal_type:
            return result

        return result


def deconv_layer(inputs, output_size, input_dim, output_dim, filter_size,
                 stride, padding, normal_type, is_training, name):
    with tf.variable_scope(name):
        # 反卷积的kernel的维度是[height, width, output_channel, input_channel]
        fils = conv_weight_init([filter_size[0], filter_size[1], output_dim, input_dim],
                                name="weights", stride=stride, mode="Xavier")
        biases = bias_init([output_dim], name="biases")

        # weight normalization的实现代码参考了WGAN-GP的源代码
        if normal_type == "weight normalization":
            norm_values = tf.sqrt(tf.reduce_sum(tf.square(fils), axis=[0, 1, 3]))
            with tf.variable_scope("weight_norm"):
                norms = tf.sqrt(tf.reduce_sum(tf.square(fils), reduction_indices=[0, 1, 3]))
                fils = fils * tf.expand_dims(norm_values / norms, 1)

        deconv = tf.nn.conv2d_transpose(inputs,
                                        fils,
                                        output_size,
                                        strides=stride,
                                        padding=padding)
        result = tf.nn.bias_add(deconv, biases)

        if normal_type == "layer normalization":
            result = slim.layer_norm(result,
                                     reuse=True,
                                     scope="layer_norm")
        elif normal_type == "instance normalization":
            result = slim.instance_norm(result,
                                        reuse=True,
                                        scope="instance_norm")
        elif normal_type == "batch normalization":
            result = slim.batch_norm(result,
                                     center=True,
                                     scale=True,
                                     is_training=is_training,
                                     scope="batch_norm")
        elif not normal_type:
            return result

        return result


def linear(inputs, output_size, name):
    shape = inputs.get_shape().as_list()

    with tf.variable_scope(name):
        matrix = linear_weight_init([shape[1], output_size], name=name + "_weights")
        bias = bias_init([output_size], name=name + "_biases")

        result = tf.matmul(inputs, matrix) + bias

        return result


def recover_model(session, meta_filename, cpkt_filename):
    logging.info("从文件 %s 中恢复模型图结构", meta_filename)
    saver = tf.train.import_meta_graph(meta_filename)
    saver.restore(session, cpkt_filename)


def save_model(saver, session, save_path):
    logging.info("保存模型信息至文件夹 %s 中", save_path)
    saver.save(session, save_path)
