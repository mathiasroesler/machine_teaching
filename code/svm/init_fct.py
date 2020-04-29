#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the initialization of the teacher.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
from numpy.random import default_rng 
from sklearn import svm
from custom_fct import *


def student_model(max_iter=5000, dual=False):
    """ Returns the student model used. The default is svm model. 
    Input:  max_iter -> int, maximum number of iteration for the SVM if no convergence
                is found.
            dual -> bool, selects dual or primal optimization problem for the SVM.
    Output: model -> SVM model
    """

    return svm.LinearSVC(dual=dual, max_iter=max_iter) # SVM model


def teacher_initialization(model, data, labels, positive_index, negative_index):
    """ Initializes the student model and the teaching set.
    Input:  model -> svm model, student model.
            data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], list of labels associated with
                the  data.
            positive_index -> int, index of the positive example to initialize the
                teacher set.
            negative_index -> int, index of the negative example to initialize the
                teacher set.
    Output: model -> svm model, student model fitted with the data.
            teaching_data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            teaching_labels -> np.array[int], labels associated with the teaching data.
    """

    positive_example = data[positive_index] 
    negative_example = data[negative_index]

    # Add labels to teaching labels
    teaching_labels = np.concatenate(([labels[positive_index]], [labels[negative_index]]), axis=0)

    teaching_data = np.concatenate(([positive_example], [negative_example]), axis=0)
    model.fit(teaching_data, teaching_labels.ravel())

    return model, teaching_data, teaching_labels


def rndm_init(labels):
    """ Randomly selects an index from the positive and the negative
    index pool.
    Input:  labels -> np.array[int], list of labels associated with
                the  data.
    Output: positive_index -> int, selected positive example index.
            negative_index -> int, selected negative example index.
    """

    positive_indices, negative_indices = find_indices(labels)

    rng = default_rng() # Set seed 

    # Find a random positive example 
    positive_index = rng.choice(positive_indices)
    
    # Find a random negative example
    negative_index = rng.choice(negative_indices)

    return positive_index, negative_index


def min_avg_init(data, labels, positive_average, negative_average):
    """ Selects the index of the positive example closest to the negative average and the 
    index of the negative example closest to the positive average.
    Input:  data -> np.array[np.array[int]], list of examples.
                First dimension number of examples.
                Second dimension features.
            labels -> np.array[int], list of labels associated with
                the  data.
            positive_average -> np.array[int], average positive example.
            negative_average -> np.array[int], average negative example.
    Output: positive_index -> int, selected positive example index.
            negative_index -> int, selected negative example index.
    """

    positive_indices, negative_indices = find_indices(labels)
    positive_examples, negative_examples = find_examples(data, labels)

    positive_dist = np.linalg.norm(positive_examples-negative_average, axis=1)
    negative_dist = np.linalg.norm(negative_examples-positive_average, axis=1)

    return positive_indices[np.argmin(positive_dist)], negative_indices[np.argmin(negative_dist)]
