#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Main program for machine teaching.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import matplotlib.pyplot as plt
from data_fct import *
from teacher import *
from sklearn import svm

def main():
    # Variables
    first_class = 0
    second_class = 0
    used_features = np.array([], dtype=np.intc)
    nb_classes = 7

    data = extract_data() # Data locate in the .data file
    classes = sort_zoo_data(data, nb_classes) # Sort examples of different classes

    if classes is None:
        exit(1)

    # User selects the classes
    print("1: mammals, 2: birds, 3:reptiles, 4:fish, 5: amphibians, 6: insects, 7: molluscs\n")

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

    print("\n1: hair, 2: feathers, 3:eggs, 4: milk, 5: airborne, 6: aquatic, 7: predator, 8: toothed, 9: backbone, 10: breathes, 11: venomous, 12: fins, 13: legs, 14: tail, 15: domestic, 16: catsize\n")

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
    

    used_classes = [classes[first_class-1], classes[second_class-1]]

    train_set, test_set = create_sets(used_classes, len(used_classes), features=used_features)

    if train_set is None or test_set is None:
        exit(1)

    model = svm.LinearSVC() # Declare SVM model
    model.fit(train_set[:, :-1], train_set[:, -1]) # Train model with data
    full_score = model.score(test_set[:, :-1], test_set[:, -1])
    print("\nFull train set length", len(train_set))
    print("\nFully trained coefficiants\n", model.coef_)
    print("\nFully trained test score", full_score)

    # Find optimal set
    delta = 0.1
    N = np.power(2, len(train_set))-1

    optimal_set = create_teacher_set(train_set, np.log(N/delta), 10)

    if optimal_set is None:
        exit(1)

    print("\nOptimal train set length", len(optimal_set))
    print("\nOptimal set\n", optimal_set)

    # Train model with optimal set
    optimal_model = svm.LinearSVC()
    optimal_model.fit(optimal_set[:, :-1], optimal_set[:, -1])
    optimal_score = optimal_model.score(train_set[:, :-1], train_set[:, -1])
    print("\nOptimaly trained coefficiants\n", optimal_model.coef_)
    print("\nOptimaly trained test score", optimal_score) 
    print("\nTraining set compression rate:", 1-len(optimal_set)/len(train_set))
    print("\nAccuracy compression rate:", 1-(optimal_score/full_score))


main()
