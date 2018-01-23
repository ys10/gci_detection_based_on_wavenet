# coding=utf-8
import tensorflow as tf
from model.layers import one_multiply_one_convolution, causal_convolution, residual_block


class WaveNetModel(object):

    def __init__(self, receptive_field, input_channels, layer_num, class_num, filter_num,
                 dilation_rates=(1, 2, 4, 8)):
        self.receptive_field = receptive_field
        self.input_channels = input_channels
        self.layer_num = layer_num
        self.class_num = class_num
        self.filter_num = filter_num
        self.dilation_rates = dilation_rates
        self.dilated_filter_kernel = 2

    def network(self, inputs, conditions):
        with tf.name_scope("network"):
            layer_inputs = causal_convolution(inputs, self.filter_num)
            skip_connections = []
            for layer_id in range(len(self.dilation_rates)):
                layer_inputs, c = residual_block(layer_inputs, conditions, self.class_num,
                                                 self.dilated_filter_kernel, layer_id, self.dilation_rates[layer_id])
                skip_connections.append(c)
            with tf.variable_scope("skip_connections"):
                outputs = sum(skip_connections)
                outputs = tf.nn.relu(outputs)
                outputs = one_multiply_one_convolution(outputs, self.class_num, name="one_multiply_one_convolution_1")
                outputs = tf.nn.relu(outputs)
                outputs = one_multiply_one_convolution(outputs, self.class_num, name="one_multiply_one_convolution_2")
                outputs = tf.nn.softmax(outputs)
                outputs = tf.layers.conv1d(outputs, self.class_num, 1, padding='same', activation=tf.nn.softmax)
                return outputs
