#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the teacher algorithm.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
from numpy.random import default_rng 
from sklearn import svm

def create_teacher_set(data, lam_coef):
    """ Produces the optimal teaching set given the data and
    a lambda coefficiant. If the teacher cannot converge in 
    ite_max_nb then it returns None.
    Input:  data -> np.array[np.array[int]], list of features with
                the last element being the label greater than 0.
                First dimension number of examples.
                Second dimension features.
            lam_coef -> int, coefficiant for the exponential distribution
                for the thresholds.
    Output: teaching_set -> np.array[np.array[int]], each row is the features 
                for an example with the last element being the label.
                First dimension number of examples.
                Second dimension features.
            accuracy -> np.array[int], accuracy of the model at each iteration.
            example_nb -> np.array[int], number of examples in teaching set at
                each iteration.
    """

    rng = default_rng() # Set seed 

    # Variables
    full_data = data
    nb_examples = data.shape[0]
    weights = np.ones(shape=(nb_examples))/nb_examples # Weights for each example
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples)) # Threshold for each example
    accuracy = np.array([0], dtype=np.intc) # List of accuracy at each iteration, starts with 0
    example_nb = np.array([0], dtype=np.intc) # List of number of examples at each iteration, starts with 0
    model = svm.LinearSVC() # SVM model

    for i in range(1, len(data)):
        # Seek two examples of different classes
        if data[i][-1] != data[i-1][-1]:
            # If the two examples belong to different classes
            model.fit(data[i-1:i+1, :-1], data[i-1:i+1, -1]) # Intialize learner

            # Add examples to the teaching set
            teaching_set = np.vstack((data[i-1], data[i]))
            
            # Test the accuracy of the model
            accuracy = np.append(accuracy, [model.score(full_data[:, :-1], full_data[:, -1])], axis=0)
            example_nb = np.append(example_nb, [len(teaching_set)], axis=0)
            
            # Remove examples from data, weights and threshold
            data = np.delete(data, [i-1, i], axis=0)
            weights = np.delete(weights, [i-1, i], axis=0)
            thresholds = np.delete(thresholds, [i-1, i], axis=0)
            break

    while len(teaching_set) != nb_examples:
        # Exit if all of the examples are in the teaching set
        missed_indices = np.array([], dtype=np.intc) # List of the indices of the missclassified examples
        added_indices = np.array([], dtype=np.intc)  # List of the indices of added examples to the teaching set

        for i in range(len(data)):
            example = data[i]
            if model.predict(example[:-1].reshape(1, -1)) != example[-1]:
                missed_indices = np.append(missed_indices, [i], axis=0)
                
        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        while np.sum(weights[missed_indices]) < 1:
            for index in missed_indices:
                # Double the value of each weight
                weights[index] = 2*weights[index] 

                if weights[index] >= thresholds[index] and index not in added_indices:
                    # If the weight exceeds the thresholds add example to teaching set and has not been added
                    added_indices = np.append(added_indices, [index], axis=0)
                    teaching_set = np.vstack((teaching_set, data[index]))

        # Fit the model with the new teaching set
        model.fit(teaching_set[:, :-1], teaching_set[:, -1])
        
        # Test model accuracy
        accuracy = np.append(accuracy, [model.score(full_data[:, :-1], full_data[:, -1])], axis=0)
        example_nb = np.append(example_nb, [len(teaching_set)], axis=0)

        # Remove data, weights and thresholds of examples in the teaching set
        data = np.delete(data, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)

    return teaching_set, accuracy, example_nb
