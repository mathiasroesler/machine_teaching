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


def main(data_name):
    """ Main function for the mnist data and machine teaching. 
    Using a one vs all strategy with an SVM model.
    Input:  data_name -> str {'cifar', 'mnist'}, name of the data to
            extract.
    """

    # Variables
    delta = 0.1
    N = 10000
    set_limit = 1000
    class_nb = 0
    iteration_nb = 1

    for i in range(iteration_nb):
        # Extract data from files
        train_data, test_data, train_labels, test_labels = extract_data(data_name) 

        # Prep data for one vs all
        train_labels, test_labels = prep_data(train_labels, test_labels, class_nb) 

        # Check for errors
        if train_labels is None:
            exit(1)
    
        # Train model with curriculum
        print("\nCurriculum training")
        cur_accuracies = continuous_training(train_data, train_labels, test_data, test_labels, 3)

        print("Final accuracy on test data: ", cur_accuracies[-1])
        
        # Train model with the all the examples
        print("\nFull training")
        full_accuracy = train_student_model(train_data, train_labels, test_data, test_labels)

        print("Final accuracy on test data: ", full_accuracy)
        
        # Find optimal set
        optimal_data, optimal_labels, mt_train_accuracies, example_nb, missed_len = create_teacher_set(train_data, train_labels, test_data, test_labels, np.log(N/delta), set_limit) 

        
        # Check for errors
        if optimal_data is None:
            exit(1)

        # Train model with optimal set
        print("\nMachine teaching training")
        mt_accuracy = train_student_model(optimal_data, optimal_labels, test_data, test_labels)
        #optimal_test_score = continuous_training(optimal_data, optimal_labels, test_data, test_labels, 3)

        print("Final accuracy on test data: ", mt_accuracy)

        #positive_average, negative_average = average_examples(train_data, train_labels)

        """
        plot_data(full_accuracy, accuracy, example_nb, missed_len)
        plot_avg_dist(optimal_data, optimal_labels, positive_average, negative_average)
        plot_example_dist(optimal_data, optimal_labels)
        """

print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)
