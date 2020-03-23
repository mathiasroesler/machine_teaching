#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from zoo_data_fct import *
from mt_fct import *
from custom_fct import *

def main_zoo():
    """ Main function for the zoo data.
    Using a one vs one strategy with an SVM.
    """

    # Variables
    nb_classes = 7

    data = extract_zoo_data() # Data located in the zoo.data file
    classes = sort_zoo_data(data, nb_classes) # Sort examples of different classes

    if classes is None:
        exit(1)

    first_class, second_class, used_features = custom_input() 

    used_classes = [classes[first_class-1], classes[second_class-1]]

    train_set, test_set = create_sets(used_classes, len(used_classes), features=used_features)

    if train_set is None or test_set is None:
        exit(1)

    # Train model with all the train set
    full_test_score = create_svm_model(train_set, test_set)

    # Find optimal set
    delta = 0.1
    N = np.power(2, len(train_set))-1
    max_iter = 5

    optimal_set, accuracy, example_nb = create_teacher_set(train_set, test_set, np.log(N/delta), full_test_score)

    if optimal_set is None:
        exit(1)

    # Train model with optimal set
    create_svm_model(optimal_set, test_set)

    plot_data(full_test_score, accuracy, example_nb)

main_zoo()

