#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 04/6/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from time import process_time
from data_fct import *
from misc_fct import *
from plot_fct import *
from custom_model import * 


def main(data_name):
    """ Main function for the mnist data and machine teaching. 
    Using a one vs all strategy with a CNN model.
    Input:  data_name -> str {'mnist', 'cifar'}, name of the dataset
            to use.
    """

    # Variables for machine teaching
    exp_rate = 150

    # Variables for self-paced learning
    threshold = 0.4
    growth_rate = 1.3

    # Variables for neural networks
    epochs = 2 
    batch_size = 32

    # Variables for plotting
    plot_types = ['ro-', 'bo-', 'go-', 'ko-']
    plot_labels = ["MT", "CL", "SPL", "Full"]

    # Other variables
    class_nb = 3
    iteration_nb = 1
    loop_ite = 3

    # Containers for time and accuracy
    times = np.zeros(len(plot_types), dtype=np.float32)
    train_acc_list = [0] * len(plot_types)
    test_acc_list = [0] * len(plot_types)


    for i in range(iteration_nb):
        print("\nITERATION", i+1)
        # Extract data from files
        train_data, test_data, train_labels, test_labels = extract_data(data_name)

        #test_labels = prep_data(test_labels, class_nb)
        #train_labels = prep_data(train_labels, class_nb)

        max_class_nb = find_class_nb(train_labels) 

        # Declare models
        model = CustomModel(train_data[0].shape, max_class_nb)
        CL_model = CustomModel(train_data[0].shape, max_class_nb)
        MT_model = CustomModel(train_data[0].shape, max_class_nb)
        SPL_model = CustomModel(train_data[0].shape, max_class_nb)


        ### FULL TRAIN ###
        # Train model with the all the examples
        print("\nFull training")
        print("\nSet length:", len(train_data))
        tic = process_time()

        model.train(train_data, train_labels, epochs)

        toc = process_time()

        # Test model
        model.test(test_data, test_labels)

        # Add full training time and accuracy
        train_acc_list[3] += model.train_acc
        test_acc_list[3] += model.test_acc
        times[3] += toc-tic


        ### MT TRAIN ###
        # Find optimal set
        print("\nGenerating optimal set")
        tic = process_time()
        optimal_data, optimal_labels, example_nb = create_teacher_set(train_data, train_labels, exp_rate, batch_size=batch_size, epochs=4)

        # Train model with teaching set
        print("\nMachine teaching training")
        print("\nSet length: ", example_nb[-1])
        MT_model.train(optimal_data, optimal_labels, epochs=epochs)

        toc = process_time()

        # Test model
        MT_model.test(test_data, test_labels)

        # Add machine teaching time
        train_acc_list[0] += MT_model.train_acc
        test_acc_list[0] += MT_model.test_acc
        times[0] += toc-tic


        ### CURRICULUM TRAIN ###
        # Train model with curriculum
        print("\nCurriculum training")
        tic = process_time()
        
        CL_model.CL_train(train_data, train_labels, epochs=epochs)

        toc = process_time()

        CL_model.test(test_data, test_labels)

        # Add curriculum time
        train_acc_list[1] += CL_model.train_acc
        test_acc_list[1] += CL_model.test_acc
        times[1] += toc-tic

        ### SP TRAIN ###
        # Train model with SP
        tic = process_time()

        print("\nSelf-paced training")

        SPL_model.SPL_train(train_data, train_labels, threshold, growth_rate, epochs=epochs)

        toc = process_time()

        SPL_model.custom_test(test_data, test_labels)
        
        # Add SPC time
        train_acc_list[2] += SPL_model.train_acc
        test_acc_list[2] += SPL_model.test_acc
        times[2] += toc-tic


    # Average time and accuracies
    times = times/iteration_nb

    for k in range(len(train_acc_list)):
        train_acc_list[k] = train_acc_list[k]/iteration_nb
        test_acc_list[k] = test_acc_list[k]/iteration_nb

    display(test_acc_list, plot_labels, times)
    plot_comp(train_acc_list, plot_types, plot_labels)
    

print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)

