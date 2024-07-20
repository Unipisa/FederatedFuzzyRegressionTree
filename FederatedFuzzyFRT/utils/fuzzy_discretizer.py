#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import numpy as np


class FuzzyDiscretizer:

    def __init__(self, num_fuzzy_sets=7, method='equifreq'):

        assert method in ['equifreq', 'uniform']
        self.method = method
        assert num_fuzzy_sets >= 3
        self.num_fuzzy_sets = num_fuzzy_sets

    def run(self, data, continuous):
        N, M = data.shape
        splits = []
        for k in range(M):
            if continuous[k]:
                cut_points = None
                if self.method == 'equifreq':
                    cut_points = np.sort(data[:, k])[np.linspace(0, N - 1, self.num_fuzzy_sets, endpoint=True, dtype='int')]
                if self.method == 'uniform':
                    cut_points = np.linspace(np.min(data[:, k]), np.max(data[:, k]), self.num_fuzzy_sets)
                if cut_points is None or len(np.unique(cut_points)) < 3:
                    splits.append([])
                else:
                    splits.append(np.unique(cut_points))
            else:
                splits.append([])

        return splits
