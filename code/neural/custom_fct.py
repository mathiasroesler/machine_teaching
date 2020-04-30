#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from sklearn import svm


def find_indices(labels):
    """ Returns the indices for the positive and negative examples given
    the labels. The positive labels must be 1 and the negative ones 0.
    Input:  labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: positive_indices -> np.array[int], list of positive labels indices.
            negative_indices -> np.array[int], list of negative labels indices.
    """
    
    # The data is one_hot
    positive_indices = np.nonzero(labels[:, 0] == 0)[0]
    negative_indices = np.nonzero(labels[:, 0] == 1)[0]

    return positive_indices, negative_indices


def find_examples(data, labels):
    """ Returns the positive and the negative examples given a data set
    and the associated labels.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: positive_examples -> tf.tensor[float32], list of
                the positive examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            negative_examples -> tf.tensor[float32], list of
                the negative examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
    """

    positive_indices, negative_indices = find_indices(labels)

    # Extract positive and negative examples for the cnn model
    positive_examples = tf.gather(data, positive_indices, axis=0)
    negative_examples = tf.gather(data, negative_indices, axis=0)

    return positive_examples, negative_examples


def average_examples(data, labels):
    """ Calculates the average positive and negative examples given
    the train data and labels.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: positive_average -> tf.tensor[float32], average positive example.
            negative_average -> tf.tensor[float32], average negative example.
    """

    positive_examples, negative_examples = find_examples(data, labels)

    return tf.constant(np.mean(positive_examples, axis=0)), tf.constant(np.mean(negative_examples, axis=0))


def sort_examples(data, labels):
    """ Extracts the positive and negative examples of the data using the labels and
    returns the sorted indices from closest to furthest to the positive and negative 
    average example respectively.
    Input:  data -> tf.tensor[float32], list of examples.
                First dimension, number of examples.
                Second and third dimensions, image. 
                Fourth dimension, color channel. 
            labels -> tf.tensor[int], list of labels associated
                with the data.
                First dimension, number of examples.
                Second dimension, one hot label.
    Output: positive_sorted_indices -> np.array[int], sorted positive indices.
            negative_sorted_indices -> np.array[int], sorted negative indices.
    """

    positive_examples, negative_examples = find_examples(data, labels)  # Find positive and negative examples
    positive_average, negative_average = average_examples(data, labels) # Estimate the positive and negative average

    # Distance between averages and examples of the same label
    positive_dist = tf.squeeze(tf.norm(positive_examples-positive_average, axis=(1, 2)))
    negative_dist = tf.squeeze(tf.norm(negative_examples-negative_average, axis=(1, 2)))

    return np.argsort(positive_dist, kind='heapsort'), np.argsort(negative_dist, kind='heapsort')


def estimate_average(full_list, cur_list, mt_list, iterations):
    """ Estimates the average accuracies for curriculum learning and 
    machine teaching after a certain number of iterations.
    Input:  full_list -> list[np.array[float]], list of accuracies at each iteration for
            fully trained model.
            cur_list -> list[np.array[float]], list of accuracies at each iteration for
            curriculum learning.
            mt_list -> list[np.array[float]], list of accuracies at each iteration for
            machine teaching.
            iterations -> int, number of iterations to average on.
    Output: full_avg -> np.array[float], list of averaged accuracies for the fully trained
            model.
            cur_avg -> np.array[float], list of averaged accuracies for curriculum learning.
            mt_avg -> np.array[float], list of averaged accuracies for machine teaching.
    """

    assert(len(cur_list) == len(mt_list) == len(full_list))

    max_ite = len(full_list)

    for i in range(max_ite):
        assert(len(full_list[i]) == len(cur_list[i]) == len(mt_list[i]))

    return np.mean(full_list, axis=0), np.mean(cur_list, axis=0), np.mean(mt_list, axis=0)
