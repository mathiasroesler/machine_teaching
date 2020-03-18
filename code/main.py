#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from zoo_data_fct import *
from mnist_data_fct import *
from teacher import *
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


def main_mnist():
    """ Main function for the mnist data. 
    Using a one vs all strategy with an SVM.
    """

    score_ratios = np.zeros(shape=(10, 1), dtype=np.float32)

    for class_nb in range(1):

        mnist_train, mnist_test = extract_mnist_data(normalize=False) # Extract data from files
        mnist_train, mnist_test = prep_data(mnist_train, mnist_test, class_nb) # Prep data for one vs all

        if mnist_train is None:
            exit(1)

        # Train model with the all the examples
        full_test_score = create_svm_model(mnist_train, mnist_test)

        # Find optimal set
        delta = 0.1
        N = 10000
        set_limit = 200
        max_iter = 10

        optimal_set, accuracy, example_nb = create_teacher_set(mnist_train, mnist_test, np.log(N/delta), set_limit, max_iter=101)
        opt_set, acc, ex_nb = create_teacher_set(mnist_train, mnist_test, np.log(N/delta), set_limit, max_iter=100)

        if optimal_set is None:
            exit(1)

        # Train model with optimal set
        optimal_test_score = create_svm_model(optimal_set, mnist_test)

        score_ratios[class_nb] = optimal_test_score/full_test_score

        print("Test score ratio: opt/full", score_ratios[class_nb])

        plt.plot(ex_nb, acc, 'ko-', label="Minimally correlated trained model")

        plot_data(full_test_score, accuracy, example_nb)

    print("\nMean test score ratio:", np.mean(score_ratios))


print("Select zoo or mnist:")
data_name = input().rstrip()

while data_name != "zoo" and data_name != "mnist":
    print("Select zoo or mnist:")
    data_name = input()

if data_name == "zoo":
    main_zoo()

else:
    main_mnist()
