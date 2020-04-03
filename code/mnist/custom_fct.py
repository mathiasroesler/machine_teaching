#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import matplotlib.pyplot as plt
from init_fct import *
from sklearn import svm


def train_student_model(model_type, train_data, train_labels, test_data, test_labels, max_iter=5000, batch_size=32, epochs=10):
    """ Trains a student model fitted with the train set and
    tested with the test set. Returns None if an error occured.
    Input:  model_type -> str, {'svm', 'cnn'} model type for the student.
            train_data -> np.array[np.array[int]] or tf.tensor, list of examples
                First dimension number of examples.
                Second dimension features.
            train_labels -> np.array[int], list of labels associated with the
                train examples.
            test_data -> np.array[np.array[int]] or tf.tensor, list of examples
                First dimension number of examples.
                Second dimension features.
            test_labels -> np.array[int], list of labels associated with the 
                test examples.
            max_iter -> int, maximum iterations for the model fitting.
            batch_size -> int, number of examples used in a batch for the neural
                network.
            epochs -> int, number of epochs for the training of the neural network.
    Output: test_score -> float, score obtained with the test set. 
    """

    model = student_model(model_type, max_iter=max_iter) # Declare student model
    
    if model == None:
        print("Error in function train_svm_model: the model was not created.")
        return None

    print("\nSet length", len(train_data))

    if model_type == 'cnn':
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
        test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)
        print("\nTest score", test_score[1])
        return test_score[1]

    else: 
        model.fit(train_data, train_labels) # Train model with data
        test_score = model.score(test_data, test_labels)    # Test score for fully trained model
        print("\nTest score", test_score)
        return test_score


def plot_data(full_train_score, accuracy, example_nb, missed_len):
    """ Plots the data.
    Input:  full_train_score -> float, accuracy for the fully trained model.
            accuracy -> np.array[int], accuracy at each iteration for
                the optimally trained model.
            example_nb -> np.array[int], number of examples at each 
                iteration for the optimally trained model.
            missed_len -> np.array[int], number of misclassified examples at
                each iteration.
    Output:
    """

    plt.plot(example_nb, accuracy,'bo-', label="Optimally trained model") # Plot for the optimal trained model 
    plt.plot(example_nb, [full_train_score for i in range(len(example_nb))], 'ro-', label="Fully trained model") # Plot for the fully trained model

    plt.xlabel("Teaching set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()
    
    plt.figure()
    plt.plot(range(len(missed_len)), missed_len, 'ko-', label="Misclassified examples")

    plt.xlabel("Iteration number")
    plt.ylabel("Number of examples")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()
    

def find_indices(model_type, labels):
    """ Returns the indices for the positive and negative examples given
    the labels. The positive labels must be 1 and the negative ones 0.
    Input:  model_type -> str, {svm, cnn} student model type.
            labels -> np.array[int], labels for a given data set.
    Output: positive_indices -> np.array[int], list of positive labels indices.
            negative_indices -> np.array[int], list of negative labels indices.
    """
    
    if model_type == 'cnn':
    # The data is one_hot
        positive_indices = np.nonzero(labels[:, 0] == 0)[0]
        negative_indices = np.nonzero(labels[:, 0] == 1)[0]

    else:
        positive_indices = np.nonzero(labels == 1)[0]
        negative_indices = np.nonzero(labels == 0)[0]

    return positive_indices, negative_indices

