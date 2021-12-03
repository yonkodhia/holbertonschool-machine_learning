#!/usr/bin/env python3
"""
how to add 2 arrays
"""


def add_arrays(arr1, arr2):
    arr = []
    if len(arr1) == len(arr2):
        for i in range(0, len(arr1)):
            arr.append(arr1[i] + arr2[i])
        return arr
