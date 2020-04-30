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
    epochs = 12
    class_nb = 0
    iteration_nb = 1
    mt_accuracies = np.zeros(epochs+1, dtype=np.float32)   # List containing the accuracies for machine teaching
    cur_accuracies = np.zeros(epochs+1, dtype=np.float32)  # List containing the accuracies for curriculum learning
    full_accuracies = np.zeros(epochs+1, dtype=np.float32) # List containing the accuracies for fully trained model 

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
        cur_accuracies += continuous_training(train_data, train_labels, test_data, test_labels, 3, epochs=epochs//3)
        
        # Train model with the all the examples
        print("\nFull training")
        full_accuracies += train_student_model(train_data, train_labels, test_data, test_labels, epochs=epochs)
        
        # Find optimal set
        print("\nGenerating optimal set")
        optimal_data, optimal_labels, mt_accuracy, example_nb, missed_len = create_teacher_set(train_data, train_labels, test_data, test_labels, np.log(N/delta), set_limit, epochs=epochs//3) 

        
        # Check for errors
        if optimal_data is None:
            exit(1)

        # Train model with teaching set
        print("\nMachine teaching training")
        mt_accuracies += train_student_model(optimal_data, optimal_labels, test_data, test_labels, epochs=epochs)
        #mt_accuracies = continuous_training(optimal_data, optimal_labels, test_data, test_labels, 3, epochs=epochs//3)

    plot_comp(full_accuracies[:-1]/iteration_nb, cur_accuracies[:-1]/iteration_nb, mt_accuracies[:-1]/iteration_nb)

print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)
