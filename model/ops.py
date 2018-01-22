# coding=utf-8
import os
import sys
import tensorflow as tf


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate,
                                  epsilon=1e-4)


def create_sgd_optimizer(learning_rate, momentum):
    return tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                      momentum=momentum)


def create_rms_prop_optimizer(learning_rate, momentum):
    return tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                     momentum=momentum,
                                     epsilon=1e-5)


optimizer_factory = {"adam": create_adam_optimizer,
                     "sgd": create_sgd_optimizer,
                     "rms_prop": create_rms_prop_optimizer}


def mu_law_encode(audio, quantization_channels):
    """
    Quantize waveform amplitudes.
    :param audio:
    :param quantization_channels:
    :return:
    """
    with tf.name_scope("encode"):
        mu = tf.to_float(quantization_channels - 1)
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
    """
    Recover waveform from quantized values.
    :param output:
    :param quantization_channels:
    :return:
    """
    with tf.name_scope("decode"):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.to_float(output) / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude


def load_model(saver, sess, logdir):
    tf.logging.info("Trying to restore saved checkpoints from {} ...".format(logdir))
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        tf.logging.info("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        tf.logging.info("  Global step was: {}".format(global_step))
        tf.logging.info("  Restoring...")
        saver.restore(sess, ckpt.model_checkpoint_path)
        tf.logging.info(" Done.")
        return global_step
    else:
        tf.logging.warning(" No checkpoint found.")
        return None


def save_model(saver, sess, logdir, step):
    model_name = "model.ckpt"
    checkpoint_path = os.path.join(logdir, model_name)
    tf.logging.info("Storing checkpoint to {} ...".format(logdir))
    sys.stdout.flush()
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    tf.logging.info(" Done.")
