# coding=utf-8
import time
import tensorflow as tf
from model.wavenet import WaveNetModel
from model.ops import optimizer_factory, load_model, save_model


# Use `tf.parse_single_example()` to extract data from a `tf.Example`
# protocol buffer, and perform any additional per-record pre-processing.
def parser(record):
    # keys_to_features = {
    #     'wave': tf.FixedLenFeature([None], tf.int64),
    #     'labels': tf.FixedLenFeature([None], tf.float32)
    # }
    # parsed = tf.parse_single_example(record, keys_to_features)

    # Perform additional pre-processing on the parsed data.
    # wave = parsed["wave"]
    # labels = parsed["labels"]

    features = tf.parse_single_example(record,
                                       features={
                                           'wave': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.FixedLenFeature([], tf.string),
                                       })
    # decode
    wave = tf.decode_raw(features['wave'], tf.int64)
    labels = tf.decode_raw(features['labels'], tf.float32)

    return wave, labels


def get_data_set_inputs(file_name, buffer_size=1024, epoch=10, batch_size=32):
    data_set = tf.data.TFRecordDataset(file_name)

    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
    # tensor for each example.
    data_set = data_set.map(parser)
    data_set = data_set.shuffle(buffer_size)
    data_set = data_set.repeat(epoch)
    data_set = data_set.batch(batch_size)
    iterator = data_set.make_one_shot_iterator()

    # `features` is a dictionary in which each value is a batch of values for
    # that feature; `labels` is a batch of labels.
    wave, labels = iterator.get_next()

    return wave, labels


def read_and_decode(filename):
    # generate a queue by filename
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'wave': tf.FixedLenFeature([], tf.string),
                                           'labels': tf.FixedLenFeature([], tf.string),
                                       })
    # decode
    wave = tf.to_float(tf.decode_raw(features['wave'], tf.int64))
    labels = tf.decode_raw(features['labels'], tf.float32)
    return wave, labels


def slice_data(seq_length, conditions, inputs, labels):
    batch_size = 4
    # init
    batch_conditions = tf.expand_dims(tf.slice(conditions, [0, 0], [seq_length, 1]), 0)
    batch_inputs = tf.expand_dims(tf.slice(inputs, [0, 0], [seq_length, 2]), 0)
    batch_labels = tf.expand_dims(tf.slice(labels, [0, 0], [seq_length, 2]), 0)
    # concat
    for i in range(1, batch_size):
        condition = tf.expand_dims(tf.slice(conditions, [i, 0], [seq_length, 1]), 0)
        batch_conditions = tf.concat([batch_conditions, condition], 0)
        input = tf.expand_dims(tf.slice(inputs, [i, 0], [seq_length, 2]), 0)
        batch_inputs = tf.concat([batch_inputs, input], 0)
        label = tf.expand_dims(tf.slice(labels, [i, 0], [seq_length, 2]), 0)
        batch_labels = tf.concat([batch_labels, label], 0)
    return batch_conditions, batch_inputs, batch_labels


def main():
    # network structure
    layer_num = 3
    class_num = 2
    filter_num = 4
    dilation_rates = tuple([2**i for i in range(layer_num)])

    # data info
    # batch_size = 2
    seq_length = 2**layer_num
    input_channels = 1

    # data
    tf_record_file_name = "data/dataset.tfrecords"
    # conditions, inputs = get_data_set_inputs(tf_record_file_name, batch_size=batch_size)
    conditions, inputs = read_and_decode(tf_record_file_name)
    labels = inputs[1:]
    # padding
    conditions = tf.pad(conditions, paddings=tf.constant([[seq_length - 1, seq_length - 1]]))
    inputs = tf.pad(inputs, paddings=tf.constant([[seq_length - 1, seq_length - 1]]))
    labels = tf.pad(labels, paddings=tf.constant([[seq_length - 1, seq_length]]))
    # 1d to 2d
    conditions = tf.expand_dims(conditions, 1)
    inputs = tf.expand_dims(inputs, 1)
    labels = tf.expand_dims(labels, 1)
    inputs = tf.concat([inputs, 1-inputs], 1)
    labels = tf.concat([labels, 1-labels], 1)
    # batch
    batch_conditions, batch_inputs, batch_labels = slice_data(seq_length, conditions, inputs, labels)
    # TODO

    # create WaveNet model
    model = WaveNetModel(seq_length, input_channels, layer_num,
                         class_num, filter_num, dilation_rates=dilation_rates)

    # loss function
    batch_outputs = model.forward(batch_inputs, batch_conditions)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=batch_outputs, labels=batch_labels)
    reduced_loss = tf.reduce_mean(loss)

    # optimizer
    learning_rate = 1e-4
    momentum = 9e-1
    optimizer = optimizer_factory["adam"](learning_rate=learning_rate, momentum=momentum)
    trainable = tf.trainable_variables()
    op = optimizer.minimize(loss, var_list=trainable)

    # Saver for storing checkpoints of the model.
    save_step = 10
    max_checkpoints = 51
    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=max_checkpoints)
    restore_path = "/tmp/"
    save_path = "/tmp/"

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = save_path != restore_path

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    tf.logging.set_verbosity(tf.logging.DEBUG)

    with tf.Session(config=config) as sess:
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        init = tf.global_variables_initializer()
        sess.run(init)

        saved_global_step = load_model(saver, sess, restore_path)
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

        # run
        step = 0
        last_saved_step = saved_global_step
        tf.logging.info("Training start !")

        try:
            for step in range(saved_global_step + 1, max_checkpoints):
                tf.logging.debug("Global step: " + str(step))
                start_time = time.time()
                cost, logits, _ = sess.run([reduced_loss, batch_outputs, op])
                duration = time.time() - start_time
                tf.logging.info("step {:d} - loss = {:.3f}, ({:.3f} sec/step)".format(step, cost, duration))
                # save model
                if step % save_step == 0:
                    save_model(saver, sess, save_path, step)
                    last_saved_step = step

            # save model at the last step
            if step > last_saved_step:
                save_model(saver, sess, save_path, step)

        except Exception as e:
            # Report exceptions to the coordinator
            coord.request_stop(e)
        coord.request_stop()
        # Terminate as usual.  It is innocuous to request stop twice.
        coord.join(threads)


if __name__ == "__main__":
    main()
