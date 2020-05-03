#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from time import process_time
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
    epochs = 10 
    class_nb = 3
    iteration_nb = 1
    mt_accuracies = np.zeros(epochs+1, dtype=np.float32)   # List containing the accuracies for machine teaching
    cur_accuracies = np.zeros(epochs+1, dtype=np.float32)  # List containing the accuracies for curriculum learning
    full_accuracies = np.zeros(epochs+1, dtype=np.float32) # List containing the accuracies for fully trained model 
    times = np.zeros(3, dtype=np.float32)                  # List containing the time of each method

    for i in range(iteration_nb):
        # Extract data from files
        train_data, test_data, train_labels, test_labels = extract_data(data_name)

        # Prepare test data labels
        test_labels = prep_data(test_labels, class_nb)
        
        # Train model with curriculum
        print("\nCurriculum training")
        tic = process_time()
        cur_accuracies += class_training(train_data, train_labels, test_data, test_labels, class_nb, epochs=epochs//10)
        toc = process_time()

        # Add curriculum time
        times[0] += toc-tic

        # Prepare train data labels
        train_labels = prep_data(train_labels, class_nb)

        """
        # Train model with curriculum
        print("\nCurriculum training")
        cur_accuracies += continuous_training(train_data, train_labels, test_data, test_labels, 3, epochs=epochs//3)
        """
        
        # Find optimal set
        print("\nGenerating optimal set")
        tic = process_time()
        optimal_data, optimal_labels, example_nb, missed_len = create_teacher_set(train_data, train_labels, np.log(N/delta), set_limit, epochs=epochs//3) 

        # Train model with teaching set
        print("\nMachine teaching training")
        mt_accuracies += train_student_model(optimal_data, optimal_labels, test_data, test_labels, epochs=epochs)
        toc = process_time()

        # Add machine teaching time
        times[1] += toc-tic


        # Train model with the all the examples
        print("\nFull training")
        tic = process_time()
        full_accuracies += train_student_model(train_data, train_labels, test_data, test_labels, epochs=epochs)
        toc = process_time()

        # Add full training time
        times[2] += toc-tic


    # Average time and accuracies
    times = times/iteration_nb
    cur_accuracies = cur_accuracies/iteration_nb
    full_accuracies = full_accuracies/iteration_nb
    mt_accuracies = mt_accuracies/iteration_nb

    print("Test accuracies: curriculum ", cur_accuracies[-1], " machine teaching ", mt_accuracies[-1], " full ", full_accuracies[-1])
    print("Times : curriculum : %.2f" % times[0], "s machine teaching %.2f" % times[1], "s full %.2f" % times[2], "s")
    plot_comp(full_accuracies[:-1], cur_accuracies[:-1], mt_accuracies[:-1])

print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)
