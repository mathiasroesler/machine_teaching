#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for selection of misclassified examples
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import tensorflow as tf
from numpy.random import default_rng 
from custom_fct import *


def select_examples(missed_indices, thresholds, weights):
    """ Selects the indices of the examples to be added to the teaching set where
    the weight of the example is greater than its threshold.
    Input:  missed_indices -> np.array[int], list of indices of missclassified
                examples.
            thresholes -> np.array[int], threshold for each example.
            weights -> np.array[int], weight of each example.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    added_indices = np.array([], dtype=np.intc)

    while np.sum(weights[missed_indices]) < 1:
        weights[missed_indices] = 2*weights[missed_indices]
        added_indices = np.nonzero(weights[missed_indices] >= thresholds[missed_indices])[0]

    return added_indices


def select_rndm_examples(missed_indices, max_nb):
    """ Selects randomly the indices of the examples to be added to the teaching set.
    Input:  missed_indices -> np.array[int], list of indices of missclassified
                examples.
            max_nb -> int, maximal number of examples to add.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    rng = default_rng() # Set random seed

    if len(missed_indices) > max_nb:
        added_indices = rng.choice(missed_indices, max_nb, replace=False)

    else:
        added_indices = missed_indices

    return added_indices


def select_min_avg_dist(model_type, missed_indices, max_nb, train_data, train_labels, positive_average, negative_average):
    """ Selects the max_nb//2 positive examples nearest to the negative average and vice-versa. 
    Input:  model_type -> str, {'svm', 'cnn'} model used for the student.
            missed_indices -> np.array[int], list of indices of missclassified
                examples.
            max_nb -> int, maximal number of examples to add.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
            positive_average -> np.array[int] or tf.tensor, average positive example.
            negative_average -> np.array[int] or tf.tensor, average negative example.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    if model_type == 'cnn':
    # For cnn student model
        # Extract misclassified examples and labels 
        missed_data = tf.gather(train_data, missed_indices, axis=0)
        missed_labels = tf.gather(train_labels, missed_indices, axis=0)

        positive_indices, negative_indices = find_indices(model_type, missed_labels)

        # Extract positive and negative missclassified examples
        positive_examples = tf.gather(missed_data, positive_indices, axis=0)
        negative_examples = tf.gather(missed_data, negative_indices, axis=0)

        if max_nb//2 < positive_indices.shape[0]:
            positive_dist = tf.norm(positive_examples-negative_average, axis=(1, 2))
            positive_indices = np.argsort(positive_dist, axis=0)[:max_nb//2]

        if max_nb//2 < negative_indices.shape[0]:
            negative_dist = tf.norm(negative_examples-positive_average, axis=(1, 2))
            negative_indices = np.argsort(negative_dist, axis=0)[:max_nb//2]

    else:
    # For svm student model
        # Extract misclassified examples and labels 
        missed_data = train_data[missed_indices]
        missed_labels = train_labels[missed_indices]

        positive_indices, negative_indices = find_indices(model_type, missed_labels)

        # Extract positive and negative missclassified examples
        positive_examples = missed_data[positive_indices]
        negative_examples = missed_data[negative_indices]
    
        if max_nb//2 < positive_indices.shape[0]:
            positive_dist = np.linalg.norm(positive_examples-negative_average, axis=1)
            positive_indices = np.argsort(positive_dist, axis=0)[:max_nb//2]

        if max_nb//2 < negative_indices.shape[0]:
            negative_dist = np.linalg.norm(negative_examples-positive_average, axis=1)
            negative_indices = np.argsort(negative_dist, axis=0)[:max_nb//2]

    try:
        added_indices = np.concatenate((np.squeeze(positive_indices), np.squeeze(negative_indices)), axis=0)

    except ValueError:
        
        if isinstance(positive_indices, np.ndarray):
            # If negative_indices is a scalar
            added_indices = np.concatenate((np.squeeze(positive_indices), [negative_indices]), axis=0)

        else:
            # If positive_indices is a scalar
            added_indices = np.concatenate(([positive_indices], np.squeeze(negative_indices)), axis=0)

    return added_indices
