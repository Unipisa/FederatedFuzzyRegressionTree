#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import pandas as pd
import os
import numpy as np
from typing import List, Tuple, Callable
from simpful import FuzzySet, Triangular_MF
from FederatedFuzzyFRT.fed_frt_node import FedFRTNode
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from FederatedFuzzyFRT.utils.constants import WS_INDEX

np_prod: Callable = np.prod
FS_quality_of_prediction = FuzzySet(function=Triangular_MF(a=-1, b=0, c=1), term='QUALITY OF PREDICTION')


def fuzzy_variance(client_stats: List[Tuple]):
    sum_wss = 0
    sum_ws = 0
    sum_wls = 0
    for WSS, WLS, WS, S, TNS in client_stats:
        sum_wss = sum_wss + WSS
        sum_ws = sum_ws + WS
        sum_wls = sum_wls + WLS
    if sum_ws == 0:
        return 0
    return (sum_wss / sum_ws) - (sum_wls / sum_ws)**2


def partition_weight(num_fuzzy_sets, idx_fuzzy_set, client_stats: List[Tuple]):
    numerator = sum([stats[idx_fuzzy_set][WS_INDEX] for stats in client_stats])
    denominator = sum([stats[idx_fs][WS_INDEX]
                       for stats in client_stats
                       for idx_fs in range(num_fuzzy_sets)])
    to_return = (numerator / denominator) if denominator != 0 else 0.0
    return to_return


def fuzzy_gain(num_fuzzy_sets, cur_variance, node_id, feature, client_stats):
    fvar = fuzzy_variance
    w = partition_weight
    stats_by_feature = [client_stat[node_id][feature] for client_stat in client_stats]
    stats_by_fs = [[client_stat[node_id][feature][idx_fs] for client_stat in client_stats]
                   for idx_fs in range(num_fuzzy_sets)]
    gain = cur_variance - sum([fvar(stats_by_fs[idx_fuzzy_set])*w(num_fuzzy_sets, idx_fuzzy_set, stats_by_feature)
                               for idx_fuzzy_set in range(num_fuzzy_sets)])
    return gain


def normalized_firing_strength_fn(x: np.ndarray, nodes: List[FedFRTNode], cache_activation_force):
    maf = cache_activation_force.get(nodes[-1].get_comp_id(), 0)
    return 0 if maf == 0 else np_prod([node.membership_degree(x) for node in nodes]) / maf


def compute_firing_strengths(x, antecedent_nodes, cache_activation_force):
    return np.array(list(map(lambda rule: normalized_firing_strength_fn(x, rule, cache_activation_force), antecedent_nodes)))


def compute_weight_rules_step1(antecedents: np.ndarray, consequents: np.ndarray,
                               training_x: np.ndarray, training_y: np.ndarray) -> Tuple:
    num_of_training_samples = len(training_x)
    num_of_rules = len(antecedents)
    is_zero_order = len(consequents.shape) == 1

    cache_activation_force = {rule[-1].get_comp_id(): rule[-1].mean_activation_force for rule in antecedents}
    cfs: Callable = lambda x: compute_firing_strengths(x, antecedents, cache_activation_force)
    firing_strengths_list = list(map(cfs, training_x))
    firing_strengths = np.array(firing_strengths_list)

    prediction_matrix = np.zeros((num_of_training_samples, num_of_rules))

    for consequent, index_consequent in zip(consequents, range(num_of_rules)):
        if not is_zero_order:
            w0 = consequent[0]
            consequent = consequent[1:]
            for sample, index_sample in zip(training_x, range(num_of_training_samples)):
                result = (consequent * sample).sum() + w0
                prediction_matrix[index_sample, index_consequent] = result
        else:
            for sample, index_sample in zip(training_x, range(num_of_training_samples)):
                result = consequent
                prediction_matrix[index_sample, index_consequent] = result

    absolute_error_matrix = np.zeros((num_of_training_samples, num_of_rules))

    for index_rule in range(num_of_rules):
        for true_value, index_sample in zip(training_y, range(num_of_training_samples)):
            predicted_value = prediction_matrix[index_sample, index_rule]
            absolute_error = abs(predicted_value - true_value)
            absolute_error_matrix[index_sample, index_rule] = absolute_error

    max_ae = np.amax(absolute_error_matrix)
    min_ae = np.amin(absolute_error_matrix)

    return firing_strengths, absolute_error_matrix, max_ae, min_ae


def compute_weight_rules_step2(firing_strengths,
                               num_of_training_samples: int,
                               num_of_rules: int,
                               absolute_error_matrix,
                               min_ae,
                               delta_max_min_ae) -> List:

    normalized_absolute_error_matrix = np.zeros((num_of_training_samples, num_of_rules))

    for i in range(num_of_training_samples):
        for j in range(num_of_rules):
            normalized_value = (absolute_error_matrix[i, j] - min_ae) / delta_max_min_ae
            normalized_absolute_error_matrix[i, j] = normalized_value

    membership_values_matrix = np.zeros((num_of_training_samples, num_of_rules))

    for i in range(num_of_training_samples):
        for j in range(num_of_rules):
            normalized_value = normalized_absolute_error_matrix[i, j]
            membership_values_matrix[i, j] = FS_quality_of_prediction.get_value(normalized_value)

    conf_supp_partial_data = []

    for j in range(num_of_rules):
        sum_weighted_membership_value = 0
        sum_firing_strength = 0
        for i in range(num_of_training_samples):
            firing_strength = firing_strengths[i, j]
            res = firing_strength * membership_values_matrix[i, j]

            sum_firing_strength += firing_strength
            sum_weighted_membership_value += res

        data = (sum_weighted_membership_value, sum_firing_strength, num_of_training_samples)
        conf_supp_partial_data.append(data)

    return conf_supp_partial_data


def compute_statistics(base_output_folder: str, df_simulation: pd.DataFrame, iteration: int = None, m: int = None) -> None:
    if df_simulation is None:
        return
    if iteration and m:
        simulation_file_path = os.path.join(base_output_folder, f'simulation_results_m_{m}_it_{iteration}.csv')
        mse_per_fold_path = base_output_folder + f'/MSE_per_fold_m_{m}_it_{iteration}.csv'
        mean_mse_per_fold_path = base_output_folder + f'/MSE_mean_per_fold_{m}_it_{iteration}.csv'
        final_results_path = base_output_folder + f'/final_results_{m}_it_{iteration}.csv'
    else:
        simulation_file_path = os.path.join(base_output_folder, 'simulation_results.csv')
        mse_per_fold_path = base_output_folder + '/MSE_per_fold.csv'
        mean_mse_per_fold_path = base_output_folder + '/MSE_mean_per_fold.csv'
        final_results_path = base_output_folder + '/final_results.csv'

    df_simulation.to_csv(simulation_file_path, index=False)

    result_dict = {
        'client_id': list(),
        'fold_iteration': list(),
        'type': list(),
        'MSE': list(),
        'RMSE': list()
    }

    grouped = df_simulation.groupby(by=['client_id', 'fold_iteration', 'type'])
    for index, grouped_data in grouped:
        client_id, fold_iteration, type_data = index
        true_values = grouped_data['true_value'].values
        predicted_values = grouped_data['predicted_value'].values

        mse = mean_squared_error(y_true=true_values, y_pred=predicted_values)
        rmse = root_mean_squared_error(y_true=true_values, y_pred=predicted_values)
        result_dict['client_id'] += [client_id]
        result_dict['fold_iteration'] += [fold_iteration]
        result_dict['type'] += [type_data]
        result_dict['MSE'] += [mse]
        result_dict['RMSE'] += [rmse]

    result_df = pd.DataFrame(data=result_dict)
    result_df = result_df.sort_values(['client_id', 'fold_iteration', 'type'], ascending=[True, True, False])
    result_df.to_csv(mse_per_fold_path, index=False)
    result_mean_per_df = result_df.drop(columns=['fold_iteration']).groupby(by=['client_id', 'type']).mean()
    result_mean_per_df['MSE_std'] = result_df.groupby(by=['client_id', 'type']).std()['MSE']
    result_mean_per_df['RMSE_std'] = result_df.groupby(by=['client_id', 'type']).std()['RMSE']
    result_mean_per_df = result_mean_per_df.rename(columns={'MSE': 'MSE_mean'})
    result_mean_per_df = result_mean_per_df.rename(columns={'RMSE': 'RMSE_mean'})
    result_mean_per_df = result_mean_per_df.sort_values(['client_id', 'type'], ascending=[True, False])
    result_mean_per_df.to_csv(mean_mse_per_fold_path)

    result_final_df = result_mean_per_df.groupby(by=['type']).mean()
    result_final_df['MSE_std'] = result_mean_per_df.groupby(by=['type']).std()['MSE_mean']
    result_final_df['RMSE_std'] = result_mean_per_df.groupby(by=['type']).std()['RMSE_mean']
    result_final_df = result_final_df.sort_values(['type'], ascending=[False])
    result_final_df.to_csv(final_results_path)
