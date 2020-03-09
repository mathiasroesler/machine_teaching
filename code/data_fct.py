#!/usr/bin/env/ python3
# -*- coding: utf-8 -*-

"""
Contains the function to read and edit the data.
Date: 6/3/2020
Author: Mathias Roesler
Mail: roesler.mathias@cmi-figure.fr
"""

import os
import numpy as np
from os.path import isfile
from pathlib import Path
from numpy.random import default_rng 

def extract_data():
    """ Finds the data file in the data directory and extracts the data
    Input:
    Output: data -> np.array[list[str]]
    """

    cwd = Path(os.getcwd())
    data_dir = str(cwd.parent)+"/data"

    for filename in os.listdir(data_dir):
        # Go through all the files in the data directory
        if filename[-5:] == ".data" and isfile(data_dir+"/"+filename): 
            # If the file contains data extract it 
            f_descriptor = open(data_dir+"/"+filename, 'r')
            data = [line.rstrip().split(',') for line in f_descriptor]
            f_descriptor.close()
    
    return np.asarray(data)

def sort_zoo_data(data):
    """ Sorts the different classes of the zoo data
    Input:  data -> np.array[list[str]], list of features with
                the last element being the label
    Output: classes -> np.array[list[str]], list of features with
                the last element being the label
    """

    mammal = []
    bird = []
    reptile = []
    fish = []
    amphibian = []
    insect = []
    mollusc = []

    for example in data:
        if example[-1] == '1':
            mammal.append(example)

        elif example[-1] == '2':
            bird.append(example)

        elif example[-1] == '3':
            reptile.append(example)

        elif example[-1] == '4':
            fish.append(example)

        elif example[-1] == '5':
            amphibian.append(example)

        elif example[-1] == '6':
            insect.append(example)

        else:
            mollusc.append(example)

    return np.array([mammal, bird, reptile, fish, amphibian, insect, mollusc])


def generate_indices(classes, percent, nb_features):
    """ Generates the indices to create a set.
    Input:  class_sizes -> list[int], list of the number of examples
                in each class.
            class_indices -> list[list[int]], list of the indices for
                the examples to be added to the set.
    Output: indices -> np.array[np.array[int]], list of indices for the examples
                to be chosen from each class.
    """

    nb_classes = len(classes)
    lengths = [len(current_class) for current_class in classes] # Number of examples in each class 
    nb_examples = [percent*lengths[i]//100 for i in range(nb_classes)] # Number of examples for the set for each class

    example_set = -np.ones(shape=(np.sum(nb_examples), nb_features+1)) # Set of examples containing the features and the label

    rng = default_rng(0) # Set seed
    indices = [rng.choice(range(lengths[i]), nb_examples[i], replace=False) for i in range(nb_classes)]

    return indices
    


def divide_2_classes(class1, class2, percent=80, features=None):
    """ Divides the two input classes into test and train sets using
        the input features. The first column of the set is the features
        The train set is composed of percent% of the examples.
        Returns -1 and -1 if an error occured.
        Input:  class1 -> np.array[list[str]], list of features with last element
                    being the label.
                class2 -> np.array[list[str]], list of features with last element
                    being the label.
                percent -> int
                features -> int or list[int]
        Output: test_set -> np.array[list[str]], each row is the features for
                    an example with the last element being the label.
                train_set -> np.array[list[str]], each row is the features for
                    an example with the last element being the label.
    """

    if features == None:
        # If no features were specified use all of them
        features = [i for i in range(len(class1[0])-1)]
        print(features)

    # Check for inconsistencies
    if (percent < 0 or percent > 100):
        print("Error in function divide_2_classes: percentage is inconsistent")
        return -1, -1 
    
    if isinstance(features, int):
        if features > len(class1[0]):
            print("Error in function divide_2_classes: selected features exceeds array")
            return -1, -1 

        if features < 0:
            print("Error in function divide_2_classes: the features must be a positive integer")
            return -1, -1

        features = [features] # Convert int feature into a list

    elif isinstance(features, list):
        for feature in features:
            if feature > len(class1[0]):
                print("Error in function divide_2_classes: selected features exceeds array")
                return -1, -1 

    else:
        print("Error in function divide_2_classes: the features argument must be an integer or a list of integers")
        return -1, -1

    # Estimate set sizes
    class1_size = len(class1)
    class1_nb_train_ex = percent*class1_size//100
    class1_nb_test_ex = class1_size - class1_nb_train_ex

    class2_size = len(class2)
    class2_nb_train_ex = percent*class2_size//100
    class2_nb_test_ex = class2_size - class2_nb_train_ex

    # Defines the sets
    train_set = -np.ones(shape=(class1_nb_train_ex + class2_nb_train_ex, len(features)+1))
    test_set = -np.ones(shape=(class1_nb_test_ex + class2_nb_test_ex, len(features)+1))

    set_indices = generate_indices([class1, class2], percent, len(features)) 

    # Randomly select percent% of examples for the train sets
    rng = default_rng(0) # Set seed for repeatability
    mammals_indices = rng.choice(range(0, class1_size), class1_nb_train_ex, replace=False)
    birds_indices = rng.choice(range(0, class2_size), class2_nb_train_ex, replace=False)

    for i in range(class1_nb_train_ex):
        for j in range(len(features)):
            train_set[i][j] = class1[mammals_indices[i]][features[j]] # Add features
        train_set[i][-1] = class1[mammals_indices[i]][-1]      # Add label

    for i in range(class1_nb_train_ex, class2_nb_train_ex+class1_nb_train_ex):
        for j in range(len(features)):
            train_set[i][j] = class2[birds_indices[i-class1_nb_train_ex]][features[j]]  # Add features
        train_set[i][-1] = class2[birds_indices[i-class1_nb_train_ex]][-1]       # Add label
    
    for i in range(class1_nb_test_ex):
        for j in range(len(features)):
            test_set[-i][j] = class1[mammals_indices[-i]][features[j]] # Add features from end
        test_set[-i][-1] = class1[mammals_indices[-i]][-1]      # Add label from end

    for i in range(class1_nb_test_ex, class2_nb_test_ex+class1_nb_test_ex):
        for j in range(len(features)):
            test_set[-i][j] = class2[birds_indices[-i+class1_nb_test_ex]][features[j]]  # Add features from end
        test_set[-i][-1] = class2[birds_indices[-i+class1_nb_test_ex]][-1]       # Add label from end

    # Shuffle sets
    rng.shuffle(train_set)
    rng.shuffle(test_set)


    return train_set, test_set

