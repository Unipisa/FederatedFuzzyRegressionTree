#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import pandas as pd
import numpy as np
from FederatedFuzzyFRT.fed_frt_model import FedFRTModel
from FederatedFuzzyFRT.fed_frt_node import NODE_ROOT_ID, FedFRTNode
from FederatedFuzzyFRT.utils.constants import WS_INDEX
from FederatedFuzzyFRT.utils.fuzzy_sets_utils import t_norm
from FederatedFuzzyFRT.utils.stats_utils import (compute_firing_strengths, compute_weight_rules_step1,
                                                 compute_weight_rules_step2)
from FederatedFuzzyFRT.utils.custom_logger import logger_info
from typing import Callable, List, Tuple

np_square: Callable = np.square


class FedFRTClient:

    def __init__(self, **kwargs):

        self.iteration = kwargs.get("iteration")
        self.client_id = kwargs.get("client_id")
        self.X_train = kwargs.get("X_train")
        self.y_train = kwargs.get("y_train")
        self.X_test = kwargs.get("X_test")
        self.y_test = kwargs.get("y_test")
        self.fuzzy_sets = kwargs.get("fuzzy_sets")
        self.tmp_membership_degree = dict()
        self.tmp_row_vector = dict()
        if self.X_train is not None:
            self.training_set = np.concatenate((self.X_train, self.y_train.reshape(self.y_train.shape[0], 1)), axis=1)
            self.tmp_row_vector[NODE_ROOT_ID] = np.array([idx for idx in range(self.training_set.shape[0])])
            self.tmp_membership_degree[NODE_ROOT_ID] = np.ones(self.training_set.shape[0])
        self.firing_strengths = None
        self.obfuscate = kwargs.get("obfuscate", True)
        self.df_train_original = kwargs.get("df_train_original")
        self.df_test_ue_original = kwargs.get("df_test_ue_original")
        self.total_number_obfuscation_check = 0
        self.total_number_obfuscation_applied_case1 = 0
        self.total_number_obfuscation_applied_case2 = 0
        self.total_number_obfuscation_applied_case3 = 0

    def get_stats_for_root_node(self) -> Tuple:
        membership_degree = self.tmp_membership_degree.get(NODE_ROOT_ID)
        rows = self.training_set
        WSS = (np_square(rows[:, -1]) * membership_degree).sum()
        WLS = (rows[:, -1] * membership_degree).sum()
        WS = membership_degree.sum()
        return WSS, WLS, WS, None, None

    def compute_stats(self, round_t, v_t, mem_threshold):
        tmp_membership_degree = dict()
        tmp_row_vector = dict()
        logger_info(f"round_t = {round_t}, client_id = {self.client_id} " + "=" * 10)
        results = dict()
        for node_q, h_q in v_t:
            stats = dict()
            node_id = node_q.id
            node_comp_id = node_q.get_comp_id()
            membership_degree = self.tmp_membership_degree.get(node_comp_id)
            mask_rows = self.tmp_row_vector.get(node_comp_id)
            rows = self.training_set
            logger_info(f'node_q = {node_q.get_comp_id()}, h_q = {h_q}')
            for feature in h_q:
                stats_feature = list()
                fuzzy_sets = self.fuzzy_sets[feature]
                row_vector, membership_vector, mask_idx = self._multidivide(rows, membership_degree, mask_rows, feature)
                for i_fs in range(len(fuzzy_sets)):
                    current_fuzzy_set = fuzzy_sets[i_fs]
                    fs_row_vector = row_vector[i_fs]
                    fs_membership_vector = membership_vector[i_fs]
                    fs_mask_idx = mask_idx[i_fs]
                    if fs_row_vector.shape[0] > 0:
                        WSS = (np_square(fs_row_vector[:, -1]) * fs_membership_vector).sum()
                        WLS = (fs_row_vector[:, -1] * fs_membership_vector).sum()
                        WS = fs_membership_vector.sum()
                        S = len(fs_membership_vector)
                        NTH = sum([1 for mem in fs_membership_vector if mem >= mem_threshold])
                        stats_feature.append((WSS, WLS, WS, S, NTH))
                    else:
                        stats_feature.append((0, 0, 0, 0, 0))
                    tmp_key = f'{node_id}_{feature}_{current_fuzzy_set.get_term()}'
                    tmp_row_vector[tmp_key] = fs_mask_idx
                    tmp_membership_degree[tmp_key] = fs_membership_vector
                stats[feature] = stats_feature
            results[node_id] = stats

        self.tmp_membership_degree = {**self.tmp_membership_degree, **tmp_membership_degree}
        t1 = len(self.tmp_row_vector)
        self.tmp_row_vector = {**self.tmp_row_vector, **tmp_row_vector}
        assert t1 + len(tmp_row_vector) == len(self.tmp_row_vector)
        return self.__check_and_nullify(v_t, results)

    def __check_and_nullify(self, v_t, results):
        if self.obfuscate:
            for node_q, h_q in v_t:
                node_id = node_q.id
                feature_stats_list = results[node_id]
                for feature in h_q:
                    feature_stats = feature_stats_list.get(feature)
                    for i_fs in range(len(feature_stats)):
                        self.total_number_obfuscation_check = self.total_number_obfuscation_check + 1
                        WSS, WLS, WS, S, NTH = feature_stats[i_fs]
                        if 1 <= S <= 2:
                            self.total_number_obfuscation_applied_case1 = self.total_number_obfuscation_applied_case1 + 1
                            feature_stats[i_fs] = (0, 0, 0, 0, 0)
                            logger_info(f'STATS NODE, IN OBFUSCATE CASE 1 client_id = {self.client_id}, feature = {feature}, i_fs = {i_fs}, WS = {WS}, S = {S}')
                        elif node_q.get_comp_id() == NODE_ROOT_ID and WS == S:
                            self.total_number_obfuscation_applied_case3 = self.total_number_obfuscation_applied_case3 + 1
                            feature_stats[i_fs] = (0, 0, 0, 0, 0)
                            logger_info(f'STATS ROOT NODE, IN OBFUSCATE client_id = {self.client_id}, feature = {feature}, i_fs = {i_fs}, WS = {WS}, S = {S}')
                        else:
                            WS_j_prev = None if i_fs == 0 else feature_stats[i_fs - 1][WS_INDEX]
                            WS_j_next = feature_stats[i_fs + 1][WS_INDEX] if i_fs < len(feature_stats) - 1 else None
                            if WS > 0 and (WS_j_prev is None or WS_j_prev == 0) and (WS_j_next is None or WS_j_next == 0):
                                self.total_number_obfuscation_applied_case2 = self.total_number_obfuscation_applied_case2 + 1
                                feature_stats[i_fs] = (0, 0, 0, 0, 0)
                                logger_info(f'STATS NODE, IN OBFUSCATE client_id = {self.client_id}, feature = {feature}, i_fs = {i_fs}, WS = {WS}, S = {S}')
        return results

    def get_privacy_stats(self):
        data = {
            'client_id': [self.client_id],
            'iteration': [self.iteration],
            'total_number_obfuscation_check': [self.total_number_obfuscation_check],
            'total_number_obfuscation_applied_case1': [self.total_number_obfuscation_applied_case1],
            'total_number_obfuscation_applied_case2': [self.total_number_obfuscation_applied_case2],
            'total_number_obfuscation_applied_case3': [self.total_number_obfuscation_applied_case3]
        }
        return pd.DataFrame(data)

    def _multidivide(self, rows, membership, mask_rows, feature) -> Tuple:
        assert len(mask_rows) == len(membership)
        mem_vect = list()
        row_vect = list()
        mask_vect = list()
        if rows.shape[0] == 0:
            for fuzzy_set in self.fuzzy_sets[feature]:
                row_vect.append(np.array([]))
                mem_vect.append(np.array([]))
                mask_vect.append(np.array([]))
        else:
            for fuzzy_set in self.fuzzy_sets[feature]:
                mask_idx_rows = np.array([idx for idx in mask_rows if fuzzy_set.get_value(rows[idx][feature]) != 0])
                if mask_idx_rows.shape[0] > 0:
                    filtered_rows = rows[mask_idx_rows, :]
                    mask_idx_membership = np.array([i for i in range(len(mask_rows)) if fuzzy_set.get_value(rows[mask_rows[i]][feature]) != 0])
                    activation_force = membership[mask_idx_membership]
                    row_vect.append(filtered_rows)
                    mem_vect.append(t_norm(np.array(list(map(lambda x: fuzzy_set.get_value(x), filtered_rows[:, feature]))), activation_force))
                    mask_vect.append(mask_idx_rows)
                else:
                    row_vect.append(np.array([]))
                    mem_vect.append(np.array([]))
                    mask_vect.append(np.array([]))
        return row_vect, mem_vect, mask_vect

    def compute_rule_weights_step1(self, antecedents, consequents):
        X_train = self.X_train
        y_train = self.y_train
        firing_strengths, absolute_error_matrix, max_ae, min_ae = compute_weight_rules_step1(antecedents,
                                                                                             consequents,
                                                                                             X_train,
                                                                                             y_train)
        self.firing_strengths = firing_strengths
        self.absolute_error_matrix = absolute_error_matrix
        return max_ae, min_ae

    def compute_weight_rules_step2(self,
                                   num_of_rules,
                                   min_ae,
                                   delta_max_min_ae) -> List:

        return compute_weight_rules_step2(self.firing_strengths,
                                          len(self.X_train),
                                          num_of_rules,
                                          self.absolute_error_matrix,
                                          min_ae,
                                          delta_max_min_ae)

    def send_model(self, model: FedFRTModel):
        self._model = model

    def evaluate_model(self, iteration):
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test

        model = self._model
        client_id = self.client_id

        data = []
        data_append = data.append

        if X_train is not None:
            for input_vector, y_true in zip(X_train, y_train):
                result, rule_id, t_num_rules = model.predict(np.array([input_vector]))[0]
                data_append([client_id, iteration, 'train', rule_id, t_num_rules, y_true, result])

        if X_test is not None:
            for input_vector, y_true in zip(X_test, y_test):
                result, rule_id, t_num_rules = model.predict(np.array([input_vector]))[0]
                data_append([client_id, iteration, 'test', rule_id, t_num_rules, y_true, result])

        columns = ['client_id', 'fold_iteration', 'type', 'rule_id', 'num_rules_activated', 'true_value', 'predicted_value']
        to_return_df = pd.DataFrame(data, columns=columns)
        return to_return_df

    def compute_matrices_for_consequents(self, rules: List[List[FedFRTNode]]):

        X_train = self.X_train
        y_train = self.y_train
        leave_nodes = [rule[-1] for rule in rules]
        cache_activation_force = {node.get_comp_id(): node.mean_activation_force for node in leave_nodes}
        cfs: Callable = lambda x: compute_firing_strengths(x, rules, cache_activation_force)
        firing_strengths_list = list(map(cfs, self.X_train))
        firing_strengths = np.array(firing_strengths_list)

        X_train = X_train.copy()
        y_train = y_train.copy()

        f = firing_strengths.copy()
        mf, nf = f.shape

        u = np.unique(X_train[:, -1])
        if u.shape[0] != 1 or u[0] != 1:
            X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))

        results = []

        for i in range(0, nf):
            n = leave_nodes[i]
            # Select firing strength of the selected rule
            _w = f[:, i]

            mask_rows = np.array(list(map(lambda _w_x: _w_x > 0, _w)))

            w = _w[mask_rows]
            _x = X_train[mask_rows, :]
            _y = y_train[mask_rows]

            # Weight input with firing strength
            xw = _x * np.sqrt(w[:, np.newaxis])

            # Weight output with firing strength
            yw = _y * np.sqrt(w)

            dot_x_t_x = np.dot(xw.T, xw)
            dot_x_t_y = np.dot(xw.T, yw)

            results.append((dot_x_t_x, dot_x_t_y, len(self.tmp_row_vector.get(n.get_comp_id(), []))))
        return results
