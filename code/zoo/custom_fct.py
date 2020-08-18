#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for the SVM model.
Date: 18/8/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm


def create_svm_model(train_set, test_set, max_iter=5000):
    """ Creates an SVM model. 
    
    The model is fitted with the train set and is tested with the test
    set, the convergence of the model is halted after max_iter
    iterations.
    Input:  train_set -> np.array[np.array[int]], list of examples with
                the last element the label.
                First dimension number of examples.
                Second dimension features.
            test_set -> np.array[np.array[int]], list of examples with
                the last element the label.
                First dimension number of examples.
                Second dimension features.
            max_iter -> int, maximum iterations for the model
                fitting, default value 5000.
    Output: full_train_score -> float, score obtained with the train 
                set. 

    """
    model = svm.LinearSVC(dual=False, max_iter=max_iter) # Declare SVM model
    model.fit(train_set[:, :-1], train_set[:, -1]) # Train model with data
    full_test_score = model.score(test_set[:, :-1], test_set[:, -1])
    print("\nSet length", len(train_set))
    print("\nTest score", full_test_score)

    return full_test_score


def plot_data(full_train_score, accuracy, example_nb):
    """ Plots the data.

    Input:  full_train_score -> float, accuracy for the fully trained
                model.
            accuracy -> np.array[int], accuracy at each iteration for
                the optimally trained model.
            example_nb -> np.array[int], number of examples at each 
                iteration for the optimally trained model.
    Output:

    """
    plt.plot(example_nb, accuracy,'bo-', label="Optimally trained model")
    plt.plot(example_nb, [full_train_score for i in range(len(example_nb))],
            'ro-', label="Fully trained model")

    plt.xlabel("Teaching set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()


def custom_input():
    """ Function to take input from the user.

    The user inputs two classes and the features used for comparison.
    Input: 
    Output: first_class -> int, first class to use.
            second_class -> int, second class to use.
            used_features -> np.array[int], list of features.

    """
    first_class = 0
    second_class = 0
    used_features = np.array([], dtype=np.intc)

    # User selects the classes
    print("1: mammals, 2: birds, 3: reptiles, 4: fish, 5: amphibians,\
            6: insects, 7: mollusks\n")

    print("Select first class between 1 and 7: ")
    while first_class < 1 or first_class > 8:
        try:
            first_class = int(input())
        except:
            print("Please enter an integer")

    print("Select second class between 1 and 7: ")
    while second_class < 1 or second_class > 8:
        try:
            second_class = int(input())
        except:
            print("Please enter an integer")

    # User selects the features
    print("\n1: hair, 2: feathers, 3: eggs, 4: milk, 5: airborne, 6: aquatic,\
        7: predator, 8: toothed, 9: backbone, 10: breathes, 11: venomous, \
        12: fins, 13: legs, 14: tail, 15: domestic, 16: catsize\n")

    print("Select features by number or input all for all of them: ")
    while used_features.size == 0:
        input_features = input().rstrip() 

        if input_features != "all":
            str_features = input_features.split(' ')

            # Try to seperate features
            try:
                for feature in str_features:
                    used_features = np.append(used_features, [int(feature)-1],
                            axis=0)

            except:
                print("Please enter integer seperated by a space")

        elif input_features == "all":
            break

    return first_class, second_class, used_features
