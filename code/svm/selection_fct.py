#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for selection of misclassified examples
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
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


def select_min_avg_dist(missed_indices, max_nb, train_data, train_labels, positive_average, negative_average):
    """ Selects the max_nb//2 positive examples nearest to the negative average and vice-versa. 
    Input:  missed_indices -> np.array[int], list of indices of missclassified
                examples.
            max_nb -> int, maximal number of examples to add.
            train_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
            positive_average -> np.array[int], average positive example.
            negative_average -> np.array[int], average negative example.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    # Extract misclassified examples and labels 
    missed_data = train_data[missed_indices]
    missed_labels = train_labels[missed_indices]

    positive_indices, negative_indices = find_indices(missed_labels)
    positive_examples, negative_examples = find_examples(missed_data, missed_labels)

    if max_nb//2 < positive_indices.shape[0]:
        positive_dist = np.linalg.norm(positive_examples-negative_average, axis=1)
        positive_indices = positive_indices[np.argsort(positive_dist, axis=0)[:max_nb//2]]

    if max_nb//2 < negative_indices.shape[0]:
        negative_dist = np.linalg.norm(negative_examples-positive_average, axis=1)
        negative_indices = negative_indices[np.argsort(negative_dist, axis=0)[:max_nb//2]]

    try:
        added_indices = np.concatenate((np.squeeze(missed_indices[positive_indices]), np.squeeze(missed_indices[negative_indices])), axis=0)

    except ValueError:
        
        if len(positive_indices) != 1: 
            # If negative_indices is a scalar
            added_indices = np.concatenate((np.squeeze(positive_indices), negative_indices), axis=0)

        else:
            # If positive_indices is a scalar
            added_indices = np.concatenate((positive_indices, np.squeeze(negative_indices)), axis=0)

    return added_indices


def select_curriculum_examples(max_nb, train_data, train_labels, ite, overlap=0):
    """ Selects the max_nb//2 positive and max_nb//2 negative examples  
    Input:  max_nb -> int, maximal number of examples to add.
            train_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the train data.
            ite -> int, 
            overlap -> int, number of examples to reuse.
    Output: added_indices -> np.array[int], list of indices of examples to be 
                added to the teaching set.
    """

    if overlap >= max_nb:
        # Check for inconsitency
        print("Error in function select_curriculum_examples: overlap is greater than number of selected examples.")
        return None

    positive_indices, negative_indices = find_indices(train_labels)
    positive_examples, negative_examples = find_examples(train_data, train_labels)  # Find positive and negative examples
    positive_average, negative_average = average_examples(train_data, train_labels) # Estimate the positive and negative average

    positive_dist = np.linalg.norm(positive_examples-positive_average, axis=1)
    negative_dist = np.linalg.norm(negative_examples-negative_average, axis=1)

    positive_sorted_indices = np.argsort(positive_dist, kind='heapsort')
    negative_sorted_indices = np.argsort(negative_dist, kind='heapsort')

    index_block = range((ite*max_nb//2)-overlap, ((ite+1)*max_nb//2)-overlap) # Block of indices to select examples from

    if index_block[0] < 0:
        # If the first index is negative shift all indices
        index_block = index_block + index_block[0]

    if index_block[-1] > len(positive_sorted_indices):
        # If the number of desired positive examples is to great
        selected_positive_indices = positive_sorted_indices[index_block[0]:]

    else:
        selected_positive_indices = positive_sorted_indices[index_block]

    if index_block[-1] > len(negative_sorted_indices):
        # If the number of desired negative examples is to great
        selected_negative_indices = negative_sorted_indices[index_block[0]:]

    else:
        selected_negative_indices = negative_sorted_indices[index_block]
        #selected_negative_indices = negative_sorted_indices[index_block[0]*4:index_block[-1]*4:4]

    #breakpoint()

    return np.concatenate((positive_indices[selected_positive_indices], negative_indices[selected_negative_indices]))
