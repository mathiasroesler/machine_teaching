#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching using SVM.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from data_fct import *
from mt_fct import *
from custom_fct import *
from plot_fct import *
from curriculum_fct import *


def main(normalize=True):
    """ Main function for the mnist data and machine teaching. 
    Using a one vs all strategy with an SVM model.
    Input: normalize -> bool, normalize the data if True
    """

    # Variables
    delta = 0.1
    N = 10000
    set_limit = 1000
    class_nb = 0
    iteration_nb = 1

    for i in range(iteration_nb):
        # Extract data from files
        mnist_train_data, mnist_test_data, mnist_train_labels, mnist_test_labels = extract_mnist_data(normalize) 

        # Prep data for one vs all
        mnist_train_labels, mnist_test_labels = prep_data(mnist_train_labels, mnist_test_labels, class_nb) 

        # Check for errors
        if mnist_train_labels is None:
            exit(1)
    
        # Train model with curriculum
        print("\nCurriculum training")
        cur_accuracies = continuous_training(mnist_train_data, mnist_train_labels, mnist_test_data, mnist_test_labels, 3)

        print("Final accuracy on test data: ", cur_accuracies[-1])
        
        # Train model with the all the examples
        print("\nFull training")
        full_accuracy = train_student_model(mnist_train_data, mnist_train_labels, mnist_test_data, mnist_test_labels)

        print("Final accuracy on test data: ", full_accuracy)
        
        # Find optimal set
        optimal_data, optimal_labels, mt_train_accuracies, example_nb, missed_len = create_teacher_set(mnist_train_data, mnist_train_labels, mnist_test_data, mnist_test_labels, np.log(N/delta), set_limit) 

        
        # Check for errors
        if optimal_data is None:
            exit(1)

        # Train model with optimal set
        print("\nMachine teaching training")
        mt_accuracy = train_student_model(optimal_data, optimal_labels, mnist_test_data, mnist_test_labels)
        #optimal_test_score = continuous_training(optimal_data, optimal_labels, mnist_test_data, mnist_test_labels, 3)

        print("Final accuracy on test data: ", mt_accuracy)

        #positive_average, negative_average = average_examples(mnist_train_data, mnist_train_labels)

        """
        plot_data(full_accuracy, accuracy, example_nb, missed_len)
        plot_avg_dist(optimal_data, optimal_labels, positive_average, negative_average)
        plot_example_dist(optimal_data, optimal_labels)
        """

main(True)
