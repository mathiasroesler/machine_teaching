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
from mt_fct import *
from sklearn import svm


def train_student_model(model_type, train_data, train_labels, test_data, test_labels, max_iter=5000, batch_size=32, epochs=20):
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

    if model_type == 'cnn':
        model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)
        test_score = model.evaluate(test_data, test_labels, batch_size=batch_size)

    else: 
        model.fit(train_data, train_labels) # Train model with data
        test_score = model.score(test_data, test_labels)    # Test score for fully trained model


    print("\nSet length", len(train_data))
    print("\nTest score", test_score)

    return test_score


def plot_data(full_train_score, accuracy, example_nb):
    """ Plots the data.
    Input:  full_train_score -> float, accuracy for the fully trained model.
            accuracy -> np.array[int], accuracy at each iteration for
                the optimally trained model.
            example_nb -> np.array[int], number of examples at each 
                iteration for the optimally trained model.
    Output:
    """

    plt.plot(example_nb, accuracy,'bo-', label="Optimally trained model") # Plot for the optimal trained model 
    plt.plot(example_nb, [full_train_score for i in range(len(example_nb))], 'ro-', label="Fully trained model") # Plot for the fully trained model

    plt.xlabel("Teaching set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()


