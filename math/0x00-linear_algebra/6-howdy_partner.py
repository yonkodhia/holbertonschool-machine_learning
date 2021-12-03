#!/usr/bin/env python3
"""
function cancat arrays
"""


def cat_arrays(arr1, arr2):
    """ function to concat 2 array """
    arr = [y for x in [arr1, arr2] for y in x]
    return arr
