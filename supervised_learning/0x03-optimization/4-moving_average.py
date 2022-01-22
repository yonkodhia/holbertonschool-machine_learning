#!/usr/bin/env python3
""" 4-Moving Average """


def moving_average(data, beta):
    """ function that Moves Average """
    new_list = [0]
    a = []
    for i in range(len(data)):
        new_list.append((1 - beta) * data[i] + beta * new_list[i])
        a.append(new_list[i+1] / (1 - beta ** (i+1)))
    return a
