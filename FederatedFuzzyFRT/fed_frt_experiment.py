#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import os
import json
import pandas as pd
import numpy as np
from FederatedFuzzyFRT.fed_frt_client import FedFRTClient
from FederatedFuzzyFRT.fed_frt_server import FedFRTServer
from FederatedFuzzyFRT.utils.fuzzy_discretizer import FuzzyDiscretizer
from FederatedFuzzyFRT.utils.fuzzy_sets_utils import create_fuzzy_sets_from_strong_partition
from FederatedFuzzyFRT.utils.stats_utils import compute_statistics
from FederatedFuzzyFRT.utils.custom_logger import logger_info
from FederatedFuzzyFRT.utils.generate_datasets import prepare_datasets_by_client as pdbc


class FedFRTExperiment:

    def __init__(self, **kwargs):
        self.M = kwargs.get("M")
        self.datasets_base_path = kwargs.get("datasets_base_path")
        self.dataset = kwargs.get("dataset")
        self.gain_threshold = kwargs.get("gain_threshold")
        self.max_number_rounds = kwargs.get("max_number_rounds")
        self.num_fuzzy_sets = kwargs.get("num_fuzzy_sets")
        self.max_depth = kwargs.get("max_depth")
        self.min_samples_split_ratio = kwargs.get("min_samples_split_ratio")
        self.base_output_folder = kwargs.get("base_output_folder")
        self.obfuscate = kwargs.get("obfuscate", True)
        self.random_state = kwargs.get("random_state")
        self.shuffle = kwargs.get("shuffle")
        self.number_of_iterations = kwargs.get("number_of_iterations")

    def execute(self):
        client_df_dict, features_names, target_feature = pdbc(self.datasets_base_path,
                                                              self.dataset,
                                                              self.M,
                                                              self.random_state,
                                                              self.shuffle,
                                                              self.number_of_iterations)
        features = [i for i in range(len(features_names))]
        df_simulation: pd.DataFrame = None

        model_output_folder = f'{self.base_output_folder}/models'
        stats_output_folder = f'{self.base_output_folder}/stats'

        os.makedirs(model_output_folder, exist_ok=True)
        os.makedirs(stats_output_folder, exist_ok=True)

        result = dict()
        result['dataset'] = self.dataset
        result['variables'] = features_names
        result['target'] = target_feature
        result['num_fuzzy_sets'] = self.num_fuzzy_sets
        result['max_depth'] = self.max_depth
        result['min_gain'] = self.gain_threshold
        result['min_samples_split_ratio'] = self.min_samples_split_ratio

        result['iterations'] = list()

        df_privacy_stats = None

        for iteration in range(1, self.number_of_iterations + 1):
            X_train = np.concatenate([client_df_dict[m]['X_train'][iteration] for m in range(1, self.M + 1)])
            logger_info(f'Iteration = {iteration}, X_train.shape = {X_train.shape}')

            n_features = X_train.shape[1]
            fuzzy_discretizer = FuzzyDiscretizer(self.num_fuzzy_sets, method="uniform")
            splits = fuzzy_discretizer.run(X_train, [True] * n_features)

            fuzzy_sets = []
            for k, points in enumerate(splits):
                if len(points) == 0:  # if the attribute doesn't have cut points
                    fuzzy_sets.append([])
                else:  # if the attribute is continuous and has cut points
                    fuzzy_sets.append(create_fuzzy_sets_from_strong_partition(points))

            fed_frt_server = FedFRTServer(iteration=iteration,
                                          features=features,
                                          features_names=features_names,
                                          target_feature=target_feature,
                                          num_fuzzy_sets=self.num_fuzzy_sets,
                                          gain_threshold=self.gain_threshold,
                                          fuzzy_sets=fuzzy_sets,
                                          max_depth=self.max_depth,
                                          max_number_rounds=self.max_number_rounds,
                                          min_samples_split_ratio=self.min_samples_split_ratio,
                                          obfuscate=self.obfuscate)

            fed_clients = [FedFRTClient(iteration=iteration,
                                        client_id=m,
                                        X_train=client_df_dict[m]['X_train'][iteration],
                                        y_train=client_df_dict[m]['y_train'][iteration],
                                        X_test=client_df_dict[m]['X_test'][iteration],
                                        y_test=client_df_dict[m]['y_test'][iteration],
                                        fuzzy_sets=fuzzy_sets,
                                        obfuscate=self.obfuscate) for m in range(1, self.M + 1)]

            iteration_stats = fed_frt_server.run_federation(fed_clients)
            result['iterations'].append(iteration_stats)

            for fed_tsk_client in fed_clients:
                df_tmp = fed_tsk_client.get_privacy_stats()
                if df_privacy_stats is None:
                    df_privacy_stats = df_tmp
                else:
                    df_privacy_stats = pd.concat([df_privacy_stats, df_tmp], axis=0)

            for fed_tsk_client in fed_clients:
                df_tmp = fed_tsk_client.evaluate_model(str(iteration))
                if df_simulation is None:
                    df_simulation = df_tmp
                else:
                    df_simulation = pd.concat([df_simulation, df_tmp], axis=0)

            compute_statistics(self.base_output_folder, df_simulation)

            fed_frt_server.save_model(model_output_folder, iteration)
        df_privacy_stats.to_csv(f'{self.base_output_folder}/privacy_stats.csv', index=False)

        stats_file = f'{stats_output_folder}/stats.json'
        with open(stats_file, "w") as file:
            json.dump(result, file)
        logger_info(f'Stats file generated: {stats_file}')
