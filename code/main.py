#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from data_fct import *
from teacher import *
from custom_fct import *

def main():

    extract_MNIST_data()

    # Variables
    nb_classes = 7

    data = extract_zoo_data() # Data located in the zoo.data file
    classes = sort_zoo_data(data, nb_classes) # Sort examples of different classes

    if classes is None:
        exit(1)

    """

    first_class, second_class, used_features = custom_input() 

    used_classes = [classes[first_class-1], classes[second_class-1]]

    train_set, test_set = create_sets(used_classes, len(used_classes), features=used_features)

    if train_set is None or test_set is None:
        exit(1)

    model = svm.LinearSVC() # Declare SVM model
    model.fit(train_set[:, :-1], train_set[:, -1]) # Train model with data
    full_train_score = model.score(train_set[:, :-1], train_set[:, -1]) # Train score for fully trained model
    full_test_score = model.score(test_set[:, :-1], test_set[:, -1])    # Test score for fully trained model
    print("\nFull train set length", len(train_set))
    print("\nFully trained coefficiants\n", model.coef_)
    print("\nFully trained test score", full_test_score)

    # Find optimal set
    delta = 0.1
    N = np.power(2, len(train_set))-1

    optimal_set, accuracy, example_nb = create_teacher_set(train_set, np.log(N/delta))

    if optimal_set is None:
        exit(1)

    print("\nOptimal train set length", len(optimal_set))
    print("\nOptimal set\n", optimal_set)

    # Train model with optimal set
    optimal_model = svm.LinearSVC()
    optimal_model.fit(optimal_set[:, :-1], optimal_set[:, -1])
    optimal_test_score = optimal_model.score(test_set[:, :-1], test_set[:, -1])
    print("\nOptimaly trained coefficiants\n", optimal_model.coef_)
    print("\nOptimaly trained test score", optimal_test_score) 
    print("\nTraining set compression rate:", 1-len(optimal_set)/len(train_set))

    plot_data(full_train_score, accuracy, example_nb)
    """ 
main()
