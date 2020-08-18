#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the functions for the teacher algorithm.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng 
from sklearn import svm

def create_teacher_set(train_set, test_set, lam_coef, set_limit):
    """ Finds a teaching set.
    
    If the teacher cannot converge in ite_max_nb then it returns None.
    Input:  train_set -> np.array[np.array[int]], list of features with
                the last element the label.
                First dimension number of examples.
                Second dimension features.
            test_set -> np.array[np.array[int]], list of features with
                the last element being the label. 
                First dimension number of examples.
                Second dimension features.
            lam_coef -> int, coefficiant for the exponential 
                distribution for the thresholds.
            set_limit -> int, maximum number of examples to be put in 
                the teaching set.
    Output: teaching_set -> np.array[np.array[int]], list of features
                with the last element label.
                First dimension number of examples.
                Second dimension features.
            accuracy -> np.array[int], accuracy of the model at
                each iteration.
            example_nb -> np.array[int], number of examples in
                teaching set at each iteration.

    """
    rng = default_rng() # Set seed 

    # Variables
    ite = 0
    nb_examples = train_set.shape[0]
    weights = np.ones(shape=(nb_examples))/nb_examples 
    thresholds = rng.exponential(1/lam_coef, size=(nb_examples))
    accuracy = np.array([0], dtype=np.intc) 
    example_nb = np.array([0], dtype=np.intc)
    model = svm.LinearSVC() # SVM model

    # Initialize learner with two examples
    for i in range(1, len(train_set)):
        if train_set[i][-1] != train_set[i-1][-1]:
            model.fit(train_set[i-1:i+1, :-1], train_set[i-1:i+1, -1])

            # Add examples to the teaching set
            teaching_set = np.vstack((train_set[i-1], train_set[i]))
            
            # Test the accuracy of the model
            accuracy = np.append(accuracy, [model.score(test_set[:, :-1],
                test_set[:, -1])], axis=0)
            example_nb = np.append(example_nb, [len(teaching_set)], axis=0)
             
            # Remove examples from train_set, weights and threshold
            train_set = np.delete(train_set, [i-1, i], axis=0)
            weights = np.delete(weights, [i-1, i], axis=0)
            thresholds = np.delete(thresholds, [i-1, i], axis=0)
            
            break

    while len(teaching_set) != nb_examples and len(teaching_set) < set_limit:
        missed_indices = np.array([], dtype=np.intc) 
        added_indices = np.array([], dtype=np.intc)
        weights = np.ones(shape=(nb_examples))/nb_examples 

        # Find all the missed examples indices
        missed_indices = np.where(model.predict(train_set[:, :-1]) !=
                train_set[:, -1])[0]

        if missed_indices.size == 0:
            # All examples are placed correctly
            break

        if len(missed_indices) > np.power(2, ite):
            added_indices = rng.choice(missed_indices, np.power(2, ite),
                    replace=False)

        else:
            added_indices = missed_indices

        teaching_set = np.vstack((teaching_set, train_set[added_indices]))

        # Fit the model with the new teaching set
        model.fit(teaching_set[:, :-1], teaching_set[:, -1])
        
        # Test model accuracy
        accuracy = np.append(accuracy, [model.score(test_set[:, :-1],
            test_set[:, -1])], axis=0)
        example_nb = np.append(example_nb, [len(teaching_set)], axis=0)

        # Remove the selected examples, weights and thresholds
        train_set = np.delete(train_set, added_indices, axis=0)
        weights = np.delete(weights, added_indices, axis=0)
        thresholds = np.delete(thresholds, added_indices, axis=0)
        ite += 1

    print("\nIteration number:", ite)

    return teaching_set, accuracy, example_nb
