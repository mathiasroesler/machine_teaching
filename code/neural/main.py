#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 18/6/2020
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
    warm_up = 10 
    threshold = 0.6
    growth_rate = 1.3

    # Variables for neural networks
    archi_type = 1
    epochs = 2
    batch_size = 128

    # Variables for plotting
    plot_types = ['ro-', 'bo-', 'go-', 'ko-']

    # Other variables
    strat_names = ["Full", "MT", "CL", "SPL"]
    class_nb = -1
    iteration_nb = 2

    # Dictionnaries for time and accuracy
    times = dict()
    train_acc_dict = dict()
    test_acc_dict = dict()

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
    models = dict()

    for strat in strat_names:
        times[strat] = 0
        train_acc_dict[strat] = [0]
        models[strat] = CustomModel(data_shape, max_class_nb, archi_type, warm_up, threshold, growth_rate)

    for i in range(iteration_nb):
        print("\nITERATION", i+1)

        for strat in strat_names:
            # For each strategy 
            model = models.get(strat)
            tic = process_time()

            # Train model
            print("\n"+ strat+" training")

            if strat == "MT":
                model.train(optimal_data, optimal_labels, strat, epochs, batch_size)

            else:
                model.train(train_data, train_labels, strat, epochs, batch_size)

            toc = process_time()

            # Test model
            model.test(test_data, test_labels)

            # Save train accuracy and time
            train_acc_dict.update({strat: train_acc_dict.get(strat) + model.train_acc})
            times.update({strat: times.get(strat) + toc-tic})

    # Average time and accuracies
    for strat in strat_names:
        times.update({strat: times.get(strat)/iteration_nb})
        train_acc_dict.update({strat: train_acc_dict.get(strat)/iteration_nb})
        test_acc_dict[strat] = models.get(strat).test_acc

    plot_train_acc(train_acc_dict, plot_types)
    plot_test_acc(test_acc_dict)

    
print("Select cifar or mnist:")
data_name = input().rstrip()

while data_name != "cifar" and data_name != "mnist":
    print("Select cifar or mnist:")
    data_name = input().rstrip()

main(data_name)

