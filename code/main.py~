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
    data = extract_data() # Data locate in the .data file
    classes = sort_zoo_data(data) # Sort examples of different classes

    # Using only two classes
    mammals = classes[0]
    birds = classes[1]
    features = [2] # Used features 

    train_set, test_set = divide_2_classes(mammals, birds, features=[i for i in range(1, len(mammals[0])-1)])

    if isinstance(train_set, int):
        return

    model = svm.LinearSVC() # Declare SVM model
    model.fit(train_set[:, :-1], train_set[:, -1]) # Train model with data
    print("Fully trained coefficiants", model.coef_)
    print("Fully trained test score", model.score(test_set[:, :-1], test_set[:, -1]))

    # Find optimal set
    delta = 0.1
    N = 10
    optimal_set = teacher_set(train_set, np.log(N/delta), 1000)

    if not isinstance(optimal_set, int):
        print("Optimal set", optimal_set)

        # Train model with optimal set
        optimal_model = svm.LinearSVC()
        optimal_model.fit(optimal_set[:, :-1], optimal_set[:, -1])
        print("Optimaly trained coefficiants", optimal_model.coef_)
        print("Optimaly trained test score", optimal_model.score(train_set[:, :-1], train_set[:, -1]))


main()
