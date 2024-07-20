#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import numpy as np
from simpful import FuzzySet, Triangular_MF
from FederatedFuzzyFRT.utils.constants import NEGATIVE_INFINITE, POSITIVE_INFINITE


def create_fuzzy_sets_from_strong_partition(points):
    fuzzySets = []
    fuzzySets.append(FuzzySet(function=Triangular_MF(a=NEGATIVE_INFINITE, b=points[0], c=points[1]), term=0))

    for index in range(1, len(points) - 1):
        fuzzySets.append(FuzzySet(function=Triangular_MF(a=points[index - 1], b=points[index], c=points[index + 1]), term=index))

    fuzzySets.append(
        FuzzySet(function=Triangular_MF(a=points[-2], b=points[-1], c=POSITIVE_INFINITE), term=len(points) - 1))
    return fuzzySets


def t_norm(x, y, norm='product'):
    """
    Parameters
    ----------
    x : np.array
        Fist array of membership.
    y : np.array
        Second array of membership.
    norm : str
        Type of norm. Must be 'product', 'min', 'lukasiewicz' or 'hamacher'.

    Returns
    -------
    j : np.array
        Array of t-normed memberships
    """

    assert x.shape == y.shape
    # product t-norm
    if norm == 'product':
        return x * y
    # Minimum, or Godel t-norm
    elif norm == 'min':
        return np.minimum(x, y)
    # Lukasiewicz t-norm
    elif norm == 'lukasiewicz':
        return np.maximum(x + y - 1, np.zeros_like(x))
    # Hamacher t-norm
    elif norm == 'hamacher':
        return x * y / (x + y - x * y)
    else:
        raise Exception('Invalid norm')
