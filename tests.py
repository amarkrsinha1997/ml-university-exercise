import numpy as np
from math import isclose

def test_entropy(entropy):
    yvec = np.array([
        [1, 1, 1, 1, 1, 1], # y[0]
        [0, 1, 1, 1, 1, 1], # y[1]
        [0, 0, 1, 1, 1, 1], # y[2]
        [0, 0, 0, 1, 1, 1], # y[3]
        [0, 0, 0, 1, 2, 2] #y[4]
    ])
    answers = np.array([0.0, 0.6500, 0.9183, 1.0, 1.4591]) # entropies for y[1] ... y[3]
    for y, answer in zip(yvec, answers):
        if not isclose(entropy(y), answer, abs_tol=0.0001):
            print("Your answer for ", y, ": ", entropy(y))
            print("Correct answer: ", answer)
            return False
    return True


def test_binary_split(binary_split):
    X = np.array([
     [0, 2], [0, 3], [0, 3],
     [1, 2], [1, 3], [1, 3]
    ])    
    answers = [
        [{0, 1, 2}, {3, 4, 5}], # when splitting X on feature0, we can put
        # {x[0], x[1], x[2]} in one group, and {x[3], x[4], x[5]} in another
        [{0, 3}, {1, 2, 4, 5}], #similarly, on feature1 we can make these groups
    ]
    for feature, answer in enumerate(answers):
        split = binary_split(X, feature)
        if not answer == split or answer == split[::-1]:
            print("Items in node to split:\n", X)
            print("Your answer for feature {}:\n".format(feature), split)
            print("Correct answer:\n", answer)
            print()
            return False
    return True


def test_information_gain(information_gain):
    X = np.array([
        [2, 0, 0], [2, 0, 1], [2, 0, 0], [1, 0, 0], [0, 1, 0],
        [0, 1, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0],
        [1, 1, 1], [1, 0, 1], [2, 1, 0], [1, 0, 1]
    ])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    answers = [.029, .151, .048]
    for feature, answer in enumerate(answers):
        if not isclose(information_gain(X, y, feature), answer, abs_tol=0.001):
            return False
        return True
    

def test_determine_best_split(determine_best_split):
    X = np.array([
        ["high", "no", "fair"], ["high", "no", "excellent"], ["high", "no", "fair"], ["medium", "no", "fair"], ["low", "yes", "fair"],
        ["low", "yes", "excellent"], ["low", "yes", "excellent"], ["medium", "no", "fair"], ["low", "yes", "fair"], ["medium", "yes", "fair"],
        ["medium", "yes", "excellent"], ["medium", "no", "excellent"], ["high", 1, 0], ["medium", "no", "excellent"]
    ])
    y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
    answer = 1
    return determine_best_split(X, y) == answer
