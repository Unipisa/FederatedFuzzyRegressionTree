#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

from typing import Tuple
from sklearn.model_selection import KFold
import pandas as pd
from FederatedFuzzyFRT.utils.custom_logger import logger_info


def prepare_datasets_by_client(datasets_base_path: str = None,
                               dataset: str = None,
                               number_of_clients: int = 5,
                               random_state: int = 0,
                               shuffle: bool = True,
                               number_of_folds: int = 5) -> Tuple:
    """
        The data are assumed to be normalised as described in the article, i.e. by robust scaling (using 2.5 and 97.5 percentiles)
        applied to the input variables to remove outliers and trim the distribution in the range [0,1].
    """
    df = pd.read_csv(f'{datasets_base_path}/{dataset}.csv')

    logger_info(f'Processing dataset = {dataset}, shape = {df.shape}, number_of_clients = {number_of_clients}, '
                f'random_state = {random_state}, shuffle = {shuffle}, number_of_folds = {number_of_folds}')

    target_feature = df.columns.values[-1]
    variable_names = df.columns[:-1].values.tolist()
                
    X = df.drop(columns=[target_feature]).to_numpy()
    y = df[target_feature].values

    kf = KFold(n_splits=number_of_folds, random_state=random_state, shuffle=shuffle)

    dataset_by_client = {client_id: {
                                        'X_train': dict(),
                                        'y_train': dict(),
                                        'X_test': dict(),
                                        'y_test': dict()
                                        } for client_id in range(1, number_of_clients + 1)}

    for fold_id, (train_index, test_index) in enumerate(kf.split(X), 1):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        if number_of_clients > 1:
            kf_client = KFold(n_splits=number_of_clients, random_state=random_state, shuffle=True)

            for indexes, client_id in zip(kf_client.split(X_train), range(1, number_of_clients + 1)):
                _, train_index_it = indexes

                dataset_by_client[client_id]['X_train'][fold_id] = X_train[train_index_it]
                dataset_by_client[client_id]['y_train'][fold_id] = y_train[train_index_it]

            kf_client = KFold(n_splits=number_of_clients, random_state=random_state, shuffle=True)
            for indexes, client_id in zip(kf_client.split(X_test), range(1, number_of_clients + 1)):
                _, test_index_it = indexes

                dataset_by_client[client_id]['X_test'][fold_id] = X_test[test_index_it]
                dataset_by_client[client_id]['y_test'][fold_id] = y_test[test_index_it]
        else:
            client_id = 1
            dataset_by_client[client_id]['X_train'][fold_id] = X_train
            dataset_by_client[client_id]['y_train'][fold_id] = y_train
            dataset_by_client[client_id]['X_test'][fold_id] = X_test
            dataset_by_client[client_id]['y_test'][fold_id] = y_test

    return dataset_by_client, variable_names, target_feature
