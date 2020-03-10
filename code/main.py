#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import matplotlib.pyplot as plt
from data_fct import *
from teacher import *
from sklearn import svm

def main():
    nb_classes = 7
    data = extract_data() # Data locate in the .data file
    classes = sort_zoo_data(data, nb_classes) # Sort examples of different classes

    if classes is None:
        exit(1)

    # Using only two classes
    used_classes = [classes[3], classes[6]]
    used_features = [2, 3, 4] # Used features 

    train_set, test_set = create_sets(used_classes, len(used_classes), features=used_features)

    if train_set is None or test_set is None:
        exit(1)

    model = svm.LinearSVC() # Declare SVM model
    model.fit(train_set[:, :-1], train_set[:, -1]) # Train model with data
    print("Fully trained coefficiants\n", model.coef_)
    print("\nFully trained test score", model.score(test_set[:, :-1], test_set[:, -1]))

    # Find optimal set
    delta = 0.1
    N = np.power(2, len(train_set))-1

    optimal_set = create_teacher_set(train_set, np.log(N/delta), 10)

    if optimal_set is None:
        exit(1)

    print("\nOptimal set\n", optimal_set)

    # Train model with optimal set
    optimal_model = svm.LinearSVC()
    optimal_model.fit(optimal_set[:, :-1], optimal_set[:, -1])
    print("Optimaly trained coefficiants\n", optimal_model.coef_)
    print("\nOptimaly trained test score", optimal_model.score(train_set[:, :-1], train_set[:, -1]))


main()
