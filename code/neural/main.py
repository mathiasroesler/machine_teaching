#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 7/5/2020
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
    delta = 0.1
    N = 10000
    set_limit = 200

    # Variables for neural networks
    epochs = 2
    batch_size = 32

    # Variables for plotting
    plot_types = ['ro-', 'bo-', 'go-', 'ko-']
    plot_labels = ["MT", "Curriculum", "SPC", "Full"]

    # Other variables
    class_nb = 3
    multiclass = True 
    iteration_nb = 1
    loop_ite = 1

    # Accuracy lists
    mt_accuracies = np.zeros(epochs+1, dtype=np.float32)   # Machine teaching
    cur_accuracies = np.zeros(epochs+1, dtype=np.float32)  # Curriculum
    full_accuracies = np.zeros(epochs+1, dtype=np.float32) # Full
    spc_accuracies = np.zeros(epochs+1, dtype=np.float32)  # Self-paced curriculum
    times = np.zeros(len(plot_types), dtype=np.float32)     # List containing the time of each method
    acc_list = [mt_accuracies, cur_accuracies, spc_accuracies, full_accuracies]

    for i in range(iteration_nb):
        # Extract data from files
        train_data, test_data, train_labels, test_labels = extract_data(data_name)

        # Prepare test data labels
        test_labels = prep_data(test_labels, class_nb, multiclass)

        ### MT TRAIN ###
        # Find optimal set
        print("\nGenerating optimal set")
        tic = process_time()
        optimal_data, optimal_labels, example_nb = create_teacher_set(train_data, train_labels, np.log(N/delta), set_limit, batch_size=2, epochs=1, multiclass=multiclass)

        # Train model with teaching set
        print("\nMachine teaching training")
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
        spc_data, spc_labels = create_spc_set(train_data, train_labels, loop_ite, class_nb, epochs=1, multiclass=multiclass)

        print("\nSelf-paced curriculum training")
        acc_list[2] += classic_training(spc_data, spc_labels, test_data, test_labels, class_nb, epochs=epochs, multiclass=multiclass, shuffle=False)
        toc = process_time()
        
        # Add SPC time
        times[2] += toc-tic


        ### FULL TRAIN ###
        # Train model with the all the examples
        print("\nFull training")
        tic = process_time()
        acc_list[3] += classic_training(train_data, train_labels, test_data, test_labels, class_nb, epochs=epochs, multiclass=multiclass)
        toc = process_time()

        # Add full training time
        times[3] += toc-tic


    # Average time and accuracies
    times = times/iteration_nb

    for accuracies in acc_list:
        accuracies = accuracies/iteration_nb

    display(acc_list, plot_labels, times)
    plot_comp(acc_list, plot_types, plot_labels)


print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)
