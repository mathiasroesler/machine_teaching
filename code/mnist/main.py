#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from mnist_data_fct import *
from mt_fct import *
from custom_fct import *


def main_mnist(model_type):
    """ Main function for the mnist data. 
    Using a one vs all strategy with an SVM or a CNN.
    Input:  model_type -> str, {'svm', 'cnn'} model used for the
                student.
    """

    score_ratios = np.zeros(shape=(10, 1), dtype=np.float32)

    for class_nb in range(10):

        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = extract_mnist_data(model_type, normalize=True) # Extract data from files
        mnist_train_labels, mnist_test_labels = prep_data(model_type, mnist_train_labels, mnist_test_labels, class_nb) # Prep data for one vs all

        if mnist_train_labels is None:
            exit(1)

        # Train model with the all the examples
        full_test_score = train_student_model(model_type, mnist_train_data, mnist_train_labels, mnist_test_data, mnist_test_labels)

        # Find optimal set
        delta = 0.1
        N = 10000
        set_limit = 2000

        optimal_data, optimal_labels, accuracy, example_nb = create_teacher_set(model_type, mnist_train_data, mnist_train_labels, mnist_test_data, mnist_test_labels, np.log(N/delta), set_limit) 

        if optimal_data is None:
            exit(1)

        # Train model with optimal set
        optimal_test_score = train_student_model(model_type, optimal_data, optimal_labels, mnist_test_data, mnist_test_labels)

        score_ratios[class_nb] = optimal_test_score/full_test_score

        print("Test score ratio: opt/full", score_ratios[class_nb])

        #plt.plot(ex_nb, acc, 'ko-', label="Minimally correlated trained model")

        plot_data(full_test_score, accuracy, example_nb)

    print("\nMean test score ratio:", np.mean(score_ratios))


print("Select svm or cnn:")
model_type = input().rstrip()

while model_type != "svm" and model_type != "cnn":
    print("Select svm or cnn:")
    model_type = input().rstrip()

main_mnist(model_type)
