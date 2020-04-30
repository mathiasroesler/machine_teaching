#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
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
    Using a one vs all strategy with a CNN model.
    Input:  data_name -> str {'mnist', 'cifar'}, name of the dataset
            to use.
    """

    # Variables
    delta = 0.1
    N = 10000
    set_limit = 1000
    epochs = 6
    class_nb = 0
    iteration_nb = 1
    mt_list = []   # List containing the accuracies of an iterarion for machine teaching
    cur_list = []  # List containing the accuracies of an iteration for curriculum learning
    full_list = [] # List containing the accuracies of an iteration for fully trained model 

    for i in range(iteration_nb):
        # Extract data from files
        train_data, test_data, train_labels, test_labels = extract_data(data_name)

        # Prep data for one vs all
        train_labels, test_labels = prep_data(train_labels, test_labels, class_nb) 

        # Check for errors
        if train_labels is None:
            exit(1)

        cur_accuracy = continuous_training(train_data, train_labels, test_data, test_labels, 3, epochs=epochs//3)
        
        # Train model with the all the examples
        full_test_score = train_student_model(train_data, train_labels, test_data, test_labels, epochs=epochs)
        
        # Find optimal set
        optimal_data, optimal_labels, mt_accuracy, example_nb, missed_len = create_teacher_set(train_data, train_labels, test_data, test_labels, np.log(N/delta), set_limit, epochs=1) 

        
        # Check for errors
        if optimal_data is None:
            exit(1)

        # Train model with optimal set
        optimal_test_score = train_student_model(optimal_data, optimal_labels, test_data, test_labels, epochs=epochs)
        #optimal_test_score = continuous_training(optimal_data, optimal_labels, test_data, test_labels, 3, epochs=epochs//3)

        mt_list.append(optimal_test_score[0]) # Append the accuracy of the machine teaching training
        cur_list.append(cur_accuracy[0])      # Append the accuracy of the curriculum learning training
        full_list.append(full_test_score[0])  # Append the accuracy of the full training

        #positive_average, negative_average = average_examples(train_data, train_labels)
        #plot_comp(full_test_score, cur_accuracy, mt_accuracy, example_nb[-1])

        """
        plot_data(full_test_score, accuracy, example_nb, missed_len)
        plot_avg_dist(optimal_data, optimal_labels, positive_average, negative_average)
        plot_example_dist(optimal_data, optimal_labels)
        """

    full_avg, cur_avg, mt_avg = estimate_average(full_list, cur_list, mt_list, iteration_nb)

    plot_comp(full_avg, cur_avg, mt_avg)

print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)
