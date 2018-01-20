# coding=utf-8
import tensorflow as tf


def one_multiply_one_convolution(inputs, filter_num, name="one_multiply_one_convolution"):
    with tf.variable_scope(name):
        outputs = tf.layers.conv1d(inputs, filter_num, 1, strides=1, padding='same')
        return outputs


def causal_convolution(inputs, filter_num, name="causal_convolution"):
    with tf.variable_scope(name):
        outputs = tf.layers.conv1d(inputs, filter_num, 2, strides=1, padding='same')
        return outputs


def residual_block(inputs, conditions, filter_num, filter_size, layer_id, dilation_rate):
    with tf.variable_scope("residual_block_" + str(layer_id)):
        in_filters = inputs.shape[2]
        with tf.variable_scope("dilated_causal_convolution"):
            input_conv = tf.layers.conv1d(inputs, filter_num, filter_size, padding='same', dilation_rate=dilation_rate)
            condition_conv = one_multiply_one_convolution(conditions, filter_num)
            filter_conv = tf.tanh(input_conv + condition_conv)
            gate_conv = tf.sigmoid(input_conv + condition_conv)
            conv = filter_conv + gate_conv
            conv = tf.layers.conv1d(conv, filter_num, 1, padding='same')
            # skip connection
            if in_filters != filter_num:
                residual = tf.layers.dense(inputs, filter_num) + conv
            else:
                residual = inputs + conv
            return residual, conv


if __name__ == '__main__':
    batch_size = 2
    seq_length = 2
    input_channels = 2
    filter_num = 4
    dilated_filter_kernel = 2
    class_num = 2
    # x = tf.constant(np.random.rand(batch_size, seq_length, input_channels).astype(np.float32))
    x = tf.constant([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    x = tf.to_float(x)
    # h = tf.constant(np.random.rand(batch_size, seq_length, input_channels).astype(np.float32))
    h = tf.constant([[[1], [1000]], [[1000], [1]]])
    h = tf.to_float(h)
    o = causal_convolution(x, filter_num)

    dilation_rates = [1, 2, 4, 8]
    skip_connections = []
    for i in range(len(dilation_rates)):
        o, c = residual_block(o, h, class_num, dilated_filter_kernel, i, dilation_rates[i])
        skip_connections.append(c)
    output = sum(skip_connections)
    output = tf.layers.conv1d(output, class_num, 1, padding='same', activation=tf.nn.relu)
    output = tf.layers.conv1d(output, class_num, 1, padding='same', activation=tf.nn.softmax)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(output).shape)
        print(sess.run(output))
