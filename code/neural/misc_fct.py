#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Miscellaneous functions.
Date: 25/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf


def find_indices(labels):
    """ Returns the indices for the examples of each class given
    the labels. The label of the first class must be 0.
    Input:  labels -> tf.tensor[int] | np.array[int], list of labels 
                associated with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: indices -> list[np.array[int]], list of indices for each class. 
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        labels = np.argmax(labels, axis=1)

    nb_classes = np.max(labels)+1
    indices = [None]*nb_classes

    for i in range(nb_classes):
        # Find indices for each class
        indices[i] = np.where(labels == i)[0]

    return indices


def find_examples(data, labels):
    """ Returns the examples for each class given a data set
    and the associated labels. The label of the first class must be 0.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int] | np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: examples -> list[tf.tensor[float32]], list of examples.
                For each element in the list:
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        labels = np.argmax(labels, axis=1)

    indices = find_indices(labels)
    nb_classes = np.max(labels)+1
    examples = [None]*nb_classes

    for i in range(nb_classes):
        # Extract examples for each class from the data
        examples[i] = tf.gather(data, indices[i], axis=0)

    return examples


def find_class_nb(labels):
    """ Finds the highest class number. The first class must be 0.
    Input:  labels -> tf.tensor[int] | np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: class_nb -> int, number of classes.
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        return np.max(np.argmax(labels, axis=1))+1

    else:
        return np.max(labels)+1


def average_examples(data, labels):
    """ Calculates the average examples for each class given
    the train data and labels. The label of the first class must be 0.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int] | np.array[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label | label.
    Output: average_data -> tf.tensor[float32], average examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    """

    if tf.is_tensor(labels):
        # If labels are one hot
        labels = np.argmax(labels, axis=1)

    nb_classes = np.max(labels)+1
    averages = np.zeros((nb_classes, data[0].shape[0], data[0].shape[1], data[0].shape[2]), dtype=np.float32)
    examples = find_examples(data, labels)

    for i in range(nb_classes):
        # Calculate the mean for each class
        averages[i] = np.mean(examples[i], axis=0) 

    return tf.stack(averages)


def display(acc_list, display_labels, times):
    """ Displays the test accuracies and the times.
    Input:  acc_list -> list[float32], list of test accuracies for
                each strategy.
            display_labels -> list[str], list of labels associated with each
                strategy.
            times -> np.array[float32], list of times associated with each
                strategy.
    Output:
    """

    try:
        assert(len(acc_list) == len(display_labels) == len(times))

    except AssertionError:
        print("Error in function display: the inputs must all have the same dimension.")
        exit(1)

    # Get longest label for justification
    if len(max(display_labels, key=len)) > len("strategy"):
        max_label_len = len(max(display_labels, key=len))

    else:
        max_label_len = len("strategy")

    print("\nStrategy".ljust(max_label_len, " "), "| Test accuracies | Times")

    for i in range(len(acc_list)):
        label_str = display_labels[i].ljust(max_label_len, " ") + " |" 
        acc_str = str(np.round(acc_list[i], 4)).ljust(len("test accuracies"), " ") + " |"
        
        print(label_str, acc_str, "%.2f" % times[i], "s")
