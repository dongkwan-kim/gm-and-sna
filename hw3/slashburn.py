import numpy as np


# No external imports are allowed other than numpy


def slashburn(adjdict, k=1):
    # Start from here

    ordered_nodes = []  # modify this
    wwratio = 0  # modify this

    # End

    return ordered_nodes, wwratio


# You may add your own functions below to use them inside slashburn().
# However, remember that we will only do: from slashburn import slashburn
# to evaluate your implementation.
if __name__ == '__main__':
    test_adjdict = {1: [2],
                    2: [3, 4],
                    3: [4, 6],
                    4: [1, 5],
                    5: [6],
                    6: []}
    test_result = slashburn(test_adjdict)
    assert set(test_result[0]) == {4, 2, 6, 3, 5, 1}
    assert test_result[1] - 0.5 < 0.0001
