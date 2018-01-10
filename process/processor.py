# coding=utf-8
import tensorflow as tf
import numpy as np
from process.ops import file_names, read_wave_data, read_marks_data, make_mask, mask_wave, mask_marks_1d

marks_path = "data/marks/"
wave_path = "data/wave/"
marks_extension = ".marks"
wave_extension = ".wav"
data_path = "data/dataset.tfrecords"


def dtype_feature_function(ndarray):
    """match appropriate tf.train.Feature class with dtype of ndarray or bytes. """
    assert isinstance(ndarray, np.ndarray)
    dtype_ = ndarray.dtype
    if dtype_ == np.float64 or dtype_ == np.float32:
        return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
    elif dtype_ == np.int64:
        return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
    else:
        raise ValueError("The input should be numpy ndarray. Instaed got {}".format(ndarray.dtype))


def main():
    with tf.python_io.TFRecordWriter(data_path) as writer:
        keys = file_names(marks_path)
        for key in keys:
            print("***************************************")
            print("key:" + key)
            # wave data.
            rate, wave_data = read_wave_data(wave_path + key + wave_extension)
            wave_data = wave_data.astype(np.int64)
            wave_length = len(wave_data)
            print("wave_length:" + str(wave_length))
            # marks data.
            marks_data = read_marks_data(marks_path + key + marks_extension, rate, wave_length)
            print("number of marks:" + str(len(marks_data)))
            # mask
            mask = make_mask(marks_data, wave_length)
            # mask wave & marks data
            masked_wave = mask_wave(wave_data, mask)
            masked_marks = mask_marks_1d(marks_data, mask, wave_length)
            assert len(masked_marks) == len(masked_wave)
            print("number of sub sequences:" + str(len(masked_wave)))
            # write to TFRecords file.
            for i in range(len(masked_marks)):
                wave = masked_wave[i]
                labels = masked_marks[i]
                example = tf.train.Example(features=tf.train.Features(feature={
                    "wave": dtype_feature_function(wave)(wave),
                    "labels": dtype_feature_function(labels)(labels)
                }))
                writer.write(example.SerializeToString())


if __name__ == '__main__':
    main()
