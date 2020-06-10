#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 10/6/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

from time import process_time
from data_fct import *
from misc_fct import *
from plot_fct import *
from custom_model import * 


def main(data_name):
    """ Main function. 
    Input:  data_name -> str {'mnist', 'cifar'}, name of the dataset
            to use.
    """

    # Variables for machine teaching
    exp_rate = 150

    # Variables for self-paced learning
    warm_up = 5
    threshold = 0.4
    growth_rate = 1.3

    # Variables for neural networks
    archi_type = 1
    epochs = 2 
    batch_size = 32

    # Variables for plotting
    plot_types = ['ro-', 'bo-', 'go-', 'ko-']
    plot_labels = ["MT", "CL", "SPL", "Full"]

    # Other variables
    class_nb = -1
    iteration_nb = 1

    # Containers for time and accuracy
    times = np.zeros(len(plot_types), dtype=np.float32)
    train_acc_list = [0] * len(plot_types)

    # Extract data from files
    train_data, test_data, train_labels, test_labels = extract_data(data_name)

    if class_nb != -1:
        print("\nBinary classifaction mode")
        train_labels = prep_data(train_labels, class_nb=class_nb)
        test_labels = prep_data(test_labels, class_nb=class_nb)

    else:
        print("\nMulti-class classification mode")

    try:
        file_name = data_name + "_indices.npy"
        print("Loading data from file", file_name)
        optimal_indices = np.load(file_name)

    except FileNotFoundError:
        print("The file", file_name, "was not found.")
        print("\nGenerating optimal set")
        optimal_indices = create_teacher_set(train_data, train_labels, exp_rate, target_acc=0.4, batch_size=batch_size, epochs=5)

    optimal_data = tf.gather(train_data, optimal_indices)
    optimal_labels = train_labels[optimal_indices]

    max_class_nb = find_class_nb(train_labels) 
    data_shape = train_data[0].shape

    # Declare models
    model = CustomModel(data_shape, max_class_nb, archi_type)
    CL_model = CustomModel(data_shape, max_class_nb, archi_type)
    MT_model = CustomModel(data_shape, max_class_nb, archi_type)
    SPL_model = CustomModel(data_shape, max_class_nb, archi_type, warm_up, threshold, growth_rate)

    for i in range(iteration_nb):
        print("\nITERATION", i+1)

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
        times[3] += toc-tic


        ### MT TRAIN ###
        # Find optimal set
        tic = process_time()

        # Train model with teaching set
        print("\nMachine teaching training")
        print("\nSet length: ", len(optimal_indices))
        MT_model.train(optimal_data, optimal_labels, epochs=epochs)

        toc = process_time()

        # Test model
        MT_model.test(test_data, test_labels)

        # Add machine teaching time
        train_acc_list[0] += MT_model.train_acc
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
        times[1] += toc-tic

        ### SP TRAIN ###
        # Train model with SP
        tic = process_time()

        print("\nSelf-paced training")

        SPL_model.SPL_train(train_data, train_labels, epochs=epochs)

        toc = process_time()

        SPL_model.test(test_data, test_labels)
        
        # Add SPC time
        train_acc_list[2] += SPL_model.train_acc
        times[2] += toc-tic

        # Reset weights of the models
        MT_model.reset_model(data_shape, archi_type)
        CL_model.reset_model(data_shape, archi_type)
        SPL_model.reset_model(data_shape, archi_type)
        model.reset_model(data_shape, archi_type)

    test_acc_list = [MT_model.test_acc, CL_model.test_acc, SPL_model.test_acc, model.test_acc]

    # Average time and accuracies
    times = times/iteration_nb

    for k in range(len(train_acc_list)):
        train_acc_list[k] = train_acc_list[k]/iteration_nb

    display(test_acc_list, plot_labels, times)
    plot_train_acc(train_acc_list, plot_types, plot_labels)
    plot_test_acc(test_acc_list, plot_labels)
    
print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnsit:")
    data_name = input().rstrip()

main(data_name)

