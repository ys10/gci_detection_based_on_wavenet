# coding=utf-8
import tensorflow as tf
from model.layers import one_multiply_one_convolution, causal_convolution, residual_block


class WaveNetModel(object):

    def __init__(self, batch_size, seq_length, input_channels, layer_num, class_num, filter_num,
                 dilation_rates=(1, 2, 4, 8)):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.input_channels = input_channels
        self.layer_num = layer_num
        self.class_num = class_num
        self.filter_num = filter_num
        self.dilation_rates = dilation_rates
        self.dilated_filter_kernel = 2
        # self.inputs = tf.placeholder(dtype=tf.int8, shape=(self.batch_size, self.seq_length, self.input_channels),
        #                              name="input")
        # self.conditions = tf.placeholder(dtype=tf.int8, shape=(self.batch_size, self.seq_length, self.input_channels),
        #                                  name="condition")
        # self.labels = tf.placeholder(dtype=tf.int8, shape=(self.batch_size, self.seq_length, self.input_channels),
        #                              name="label")

    def forward(self, inputs, conditions):
        with tf.name_scope("network"):
            layer_inputs = causal_convolution(inputs, self.filter_num)
            skip_connections = []
            for i in range(len(self.dilation_rates)):
                layer_inputs, c = residual_block(layer_inputs, conditions, self.class_num,
                                                 self.dilated_filter_kernel, i, self.dilation_rates[i])
                skip_connections.append(c)
            with tf.name_scope("skip_connections"):
                outputs = sum(skip_connections)
                outputs = tf.nn.relu(outputs)
                outputs = one_multiply_one_convolution(outputs, self.class_num)
                outputs = tf.nn.relu(outputs)
                outputs = one_multiply_one_convolution(outputs, self.class_num)
                outputs = tf.nn.softmax(outputs)
                outputs = tf.layers.conv1d(outputs, self.class_num, 1, padding='same', activation=tf.nn.softmax)
                return outputs
