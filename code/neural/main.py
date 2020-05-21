#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 21/5/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from time import process_time
from data_fct import *
from strategies import *
from custom_fct import *
from plot_fct import *


def main(data_name):
    """ Main function for the mnist data and machine teaching. 
    Using a one vs all strategy with a CNN model.
    Input:  data_name -> str {'mnist', 'cifar'}, name of the dataset
            to use.
    """

    # Variables for machine teaching
    exp_rate = 150

    # Variables for self-paced learning
    threshold = 0.1
    growth_rate = 1.3

    # Variables for neural networks
    epochs = 6
    batch_size = 32

    # Variables for plotting
    plot_types = ['ro-', 'bo-', 'go-', 'ko-']
    plot_labels = ["MT", "CL", "SPL", "Full"]

    # Other variables
    class_nb = 3
    multiclass = True 
    iteration_nb = 1
    loop_ite = 3

    # Accuracy lists
    mt_accuracies = np.zeros(epochs+1, dtype=np.float32)   # Machine teaching
    cur_accuracies = np.zeros(epochs+1, dtype=np.float32)  # Curriculum
    full_accuracies = np.zeros(epochs+1, dtype=np.float32) # Full
    spc_accuracies = np.zeros(epochs+1, dtype=np.float32)  # Self-paced curriculum
    times = np.zeros(len(plot_types), dtype=np.float32)     # List containing the time of each method
    acc_list = [mt_accuracies, cur_accuracies, spc_accuracies, full_accuracies]

    for i in range(iteration_nb):
        print("\nITERATION", i+1)
        # Extract data from files
        train_data, test_data, train_labels, test_labels = extract_data(data_name)

        # Prepare test data labels
        test_labels = prep_data(test_labels, class_nb, multiclass)


        ### FULL TRAIN ###
        # Train model with the all the examples
        print("\nFull training")
        print("\nSet length:", len(train_data))
        tic = process_time()
        acc_list[3] += classic_training(train_data, train_labels, test_data, test_labels, class_nb, epochs=epochs, multiclass=multiclass)
        toc = process_time()

        # Add full training time
        times[3] += toc-tic


        ### MT TRAIN ###
        # Find optimal set
        print("\nGenerating optimal set")
        tic = process_time()
        optimal_data, optimal_labels, example_nb = create_teacher_set(train_data, train_labels, exp_rate, acc_list[3][-2], batch_size=batch_size, epochs=4, multiclass=multiclass)

        # Train model with teaching set
        print("\nMachine teaching training")
        print("\nSet length: ", example_nb[-1])
        acc_list[0] += classic_training(optimal_data, optimal_labels, test_data, test_labels, class_nb, epochs=epochs, multiclass=multiclass)
        toc = process_time()

        # Add machine teaching time
        times[0] += toc-tic


        ### CURRICULUM TRAIN ###
        # Train model with curriculum
        print("\nCurriculum training")
        tic = process_time()
        acc_list[1] += curriculum_training(train_data, train_labels, test_data, test_labels, class_nb, epochs=epochs, multiclass=multiclass)
        toc = process_time()

        # Add curriculum time
        times[1] += toc-tic


        ### SPC TRAIN ###
        # Train model with SPC
        print("\nGenerating SPC set")
        tic = process_time()

        print("\nSelf-paced curriculum training")
        acc_list[2] += classic_training(train_data, train_labels, test_data, test_labels, class_nb, epochs=epochs, multiclass=multiclass, threshold=threshold, growth_rate=growth_rate)
        toc = process_time()
        
        # Add SPC time
        times[2] += toc-tic

    # Average time and accuracies
    times = times/iteration_nb

    for k in range(len(acc_list)):
        acc_list[k] = acc_list[k]/iteration_nb

    display(acc_list, plot_labels, times)
    plot_comp(acc_list, plot_types, plot_labels)


print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)
