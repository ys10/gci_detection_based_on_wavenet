# coding=utf-8

from scipy.io import wavfile
import numpy as np
import os


def read_marks_data(path, rate, wave_length):
    """
    Read marks file.
    :param path:
        marks file path(containing time of gci).
    :param rate:
        sampling rate.
    :param wave_length:
        wave length.
    :return:
        an list containing the index(time * rate) of gci.
    """
    marks = list()
    with open(path) as mark_file:
        while 1:
            lines = mark_file.readlines(10000)
            if not lines:
                break
            for line in lines:
                marks.append(round(float(line) * rate))
    if marks[-1] == wave_length:
        return marks[:-2]
    return marks


def read_wave_data(path):
    """
    Read wav file.
    :param path:
        wav file path.
    :return:
        sampling rate, waveform data.
    """
    rate, data = wavfile.read(path)
    return rate, data[:]


def file_names(file_dir):
    """
    List all file names(without extension) in target directory.
    :param file_dir:
        target directory.
    :return:
        a list containing file names.
    """
    file_names_list = list()
    for _, _, files in os.walk(file_dir):
        for file in files:
            file_names_list.append(file.split(".")[0])
    return file_names_list


def full_file_names(file_dir):
    """
    List all full file names(with extension) in target directory.
    :param file_dir:
        target directory.
    :return:
        a list containing full file names.
    """
    for _, _, files in os.walk(file_dir):
        return files


def make_mask(marks, wave_length, mask_range=16):
    """
    Make mask based on mask range & marks.
    :param marks:
        a numpy array containing index(time * sampling rate) of marks.
    :param wave_length:
        wav file length.
    :param mask_range:
        a range around marks index should be masked.
    :return:
        a list of mask tuples, like:
        [
        (1, 99),
        (105, 118),
        ...,
        (177, 192),
        ]
    """
    tuples = list()
    # create mask tuples
    for index in marks:
        if index == wave_length:
            break
        begin = index - mask_range
        begin = begin if begin >= 0 else 0
        end = index + mask_range
        end = end if end < wave_length else wave_length - 1
        tuples.append((begin, end))
    # merge adjacent mask tuples
    mask = list()
    i = 0
    while i < len(tuples):
        begin = tuples[i][0]
        end = tuples[i][1]
        j = i+1
        while j < len(tuples):
            if tuples[j][0] <= end:
                end = tuples[j][1]
                j += 1
            else:
                break
        mask.append((begin, end))
        i = j
    return mask


def mask_wave(wave, mask):
    """
    Mask wave data.
    :param wave:
        wave data, an numpy array.
    :param mask:
        a list of mask tuples, see make_mask function.
    :return:
        masked wave data, a list of numpy arrays.
    """
    masked_wave = list()
    for (begin, end) in mask:
        masked_wave.append(wave[begin:end])
        pass
    return masked_wave


def mask_marks_1d(marks, mask, wave_length):
    """
    Mask marks data.
    :param marks:
        marks data, a list.
    :param wave_length:
        wave data length.
    :param mask:
        a list of mask tuples, see make_mask function.
    :return:
        masked marks data, a list of numpy arrays.
    """
    # init labels
    labels = np.zeros(wave_length, dtype=np.float32)
    # label labels with marks
    for index in marks:
        labels[index] = 1
    # mask labels
    masked_labels = list()
    for (begin, end) in mask:
        masked_labels.append(labels[begin:end])
        pass
    return masked_labels


def mask_marks_2d(marks, mask, wave_length):
    """
    Mask marks data.
    :param marks:
        marks data, a list.
    :param wave_length:
        wave data length.
    :param mask:
        a list of mask tuples, see make_mask function.
    :return:
        masked marks data, a list of numpy arrays.
    """
    # init labels
    one_vector = np.ones(wave_length, dtype=np.float32)
    zero_vector = np.zeros(wave_length, dtype=np.float32)
    labels = np.array([one_vector, zero_vector]).transpose()
    # label labels with marks
    for index in marks:
        labels[index][0] = 0
        labels[index][1] = 1
    # mask labels
    masked_labels = list()
    for (begin, end) in mask:
        masked_labels.append(labels[begin:end])
        pass
    return masked_labels
