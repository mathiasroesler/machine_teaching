#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Custom functions for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_data(full_train_score, accuracy, example_nb):
    """ Plots the data.
    Input:  full_train_score -> float, accuracy for the fully trained model.
            accuracy -> np.array[int], accuracy at each iteration for
                the optimally trained model.
            example_nb -> np.array[int], number of examples at each 
                iteration for the optimally trained model.
    Output:
    """

    plt.plot(example_nb, accuracy,'bo-', label="Optimally trained model") # Plot for the optimal trained model 
    plt.plot(example_nb, [full_train_score for i in range(len(example_nb))], 'ro-', label="Fully trained model") # Plot for the fully trained model

    plt.xlabel("Teaching set size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, which="both") # Add grid
    plt.show()


def custom_input():
    """ Input function for the classes and features.
    Input: 
    Output: first_class -> int, first class to use.
            second_class -> int, second class to use.
            used_features -> np.array[int], list of features.
    """

    first_class = 0
    second_class = 0
    used_features = np.array([], dtype=np.intc)

    # User selects the classes
    print("1: mammals, 2: birds, 3: reptiles, 4: fish, 5: amphibians, 6: insects, 7: molluscs\n")

    print("Select first class between 1 and 7: ")
    while first_class < 1 or first_class > 8:
        try:
            first_class = int(input())
        except:
            print("Please enter an integer")

    print("Select second class between 1 and 7: ")
    while second_class < 1 or second_class > 8:
        try:
            second_class = int(input())
        except:
            print("Please enter an integer")

    print("\n1: hair, 2: feathers, 3: eggs, 4: milk, 5: airborne, 6: aquatic, 7: predator, 8: toothed, 9: backbone, 10: breathes, 11: venomous, 12: fins, 13: legs, 14: tail, 15: domestic, 16: catsize\n")

    print("Select features by number or input all for all of them: ")
    while used_features.size == 0:
        input_features = input().rstrip() # Recuperate input without \n character

        if input_features != "all":
            str_features = input_features.split(' ')

            try:
                for feature in str_features:
                    used_features = np.append(used_features, [int(feature)-1], axis=0)

            except:
                print("Please enter integer seperated by a space")

        elif input_features == "all":
            break

    return first_class, second_class, used_features
    


