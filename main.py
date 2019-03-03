import numpy as np
import math

def entropy(big_list):
    elements_in_list = set(big_list)
    length = len(big_list)
    entropy_sum = 0

    for element in elements_in_list:
        frequency = np.count_nonzero(big_list == element)
        x = frequency/length
        entropy_sum -=  x * math.log2(x)
    return entropy_sum

def binary_split(array2d, feature_index):
    sets = []
    element = []
    for array in array2d:
        element.append(array[feature_index])
    big_list = np.array(element)

    elements = set(big_list)
    for element in elements:
        sets.append(set(np.where(big_list == element )[0]))  
    return sets

def information_gain(features, main_label, feature_index):
    main_node_entropy = entropy(main_label)
    
    split_according_to_feature = binary_split(features, feature_index) 
    label_length = len(main_label)

    info_gain = main_node_entropy

    #Calculating Sumation
    for split_feature_index in split_according_to_feature:
        split_set_length = len(split_feature_index)
        label_list_feature = []

        for index in split_feature_index:
            label_list_feature.append(main_label[index])
        big_list = np.array(label_list_feature)
        info_gain -= (split_set_length/label_length)*entropy(big_list)

    return info_gain

def determine_best_split(X, y):
    info_gain_list=[]

    for feature_index,training_sample in enumerate(X[0]):
        info_gain_list.append(information_gain(X,y,feature_index))

    return info_gain_list.index(max(info_gain_list))

