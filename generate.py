# coding=utf-8
import time
import tensorflow as tf
from model.wavenet import WaveNetModel
from model.ops import load_model


def main():
    # network structure
    layer_num = 10
    class_num = 16
    filter_num = 4
    dilation_rates = tuple([2**i for i in range(layer_num+1)])
    # data info
    batch_size = 16
    seq_length = 2**layer_num
    input_channels = 1
    # data
    conditions = None
    inputs = None
    # TODO
    # create dnn model
    model = WaveNetModel(batch_size, seq_length, input_channels, layer_num,
                         class_num, filter_num, dilation_rates=dilation_rates)
    # loss function
    outputs = model.forward(inputs, conditions)
    # Saver for storing checkpoints of the model.
    max_checkpoints = 50
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=max_checkpoints)
    restore_path = "/tmp/"
    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # restore variables
        saved_global_step = load_model(saver, sess, restore_path)
        if saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
        # run
        step = None
        last_saved_step = saved_global_step
        for step in range(saved_global_step + 1, max_checkpoints):
            start_time = time.time()
            logits = sess.run([outputs])
            duration = time.time() - start_time
            tf.logging.info('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, logits, duration))
            # save logits
            # TODO


if __name__ == '__main__':
    main()
