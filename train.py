# coding=utf-8
import time
import tensorflow as tf
from model.wavenet import WaveNetModel
from model.ops import optimizer_factory, load_model, save_model

tf_record_file_name = "data/dataset.tfrecords"


def load_data(file_name):
    data_set = tf.data.TFRecordDataset(file_name)
    iterator = data_set.make_one_shot_iterator()
    next_element = iterator.get_next()
    # TODO
    pass

def main():
    # network structure
    layer_num = 10
    class_num = 2
    filter_num = 4
    dilation_rates = tuple([2**i for i in range(layer_num)])
    # data info
    batch_size = 32
    seq_length = 2**layer_num
    input_channels = 1
    # data
    conditions = None
    inputs = None
    labels = None
    # TODO
    # create dnn model
    model = WaveNetModel(batch_size, seq_length, input_channels, layer_num,
                         class_num, filter_num, dilation_rates=dilation_rates)
    # loss function
    outputs = model.forward(inputs, conditions)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
    reduced_loss = tf.reduce_mean(loss)
    # optimizer
    learning_rate = 1e-4
    momentum = 9e-1
    optimizer = optimizer_factory["adam"](learning_rate=learning_rate, momentum=momentum)
    trainable = tf.trainable_variables()
    op = optimizer.minimize(loss, var_list=trainable)
    # Saver for storing checkpoints of the model.
    save_step = 10
    max_checkpoints = 50
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=max_checkpoints)
    restore_path = "/tmp/"
    save_path = "/tmp/"
    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = save_path != restore_path
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # restore variables
        saved_global_step = load_model(saver, sess, restore_path)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1
        # run
        step = None
        last_saved_step = saved_global_step
        for step in range(saved_global_step + 1, max_checkpoints):
            start_time = time.time()
            reduced_loss, logits, _ = sess.run([reduced_loss, outputs, op])
            duration = time.time() - start_time
            tf.logging.info('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, reduced_loss, duration))
            # save model
            if step % save_step == 0:
                save_model(saver, sess, save_path, step)
                last_saved_step = step
        # save model at the last step
        if step > last_saved_step:
            save_model(saver, sess, save_path, step)


if __name__ == '__main__':
    main()
