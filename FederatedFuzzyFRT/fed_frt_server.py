#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import numpy as np
from typing import Dict, List, Tuple
from FederatedFuzzyFRT.fed_frt_client import FedFRTClient
from FederatedFuzzyFRT.fed_frt_model import FedFRTModel
from FederatedFuzzyFRT.fed_frt_node import FedFRTNode
from FederatedFuzzyFRT.utils.stats_utils import fuzzy_variance, fuzzy_gain
from FederatedFuzzyFRT.utils.custom_logger import logger_info


class FedFRTServer:

    def __init__(self, **kwargs):
        FedFRTNode.reset_node_id()
        self.iteration = kwargs.get("iteration")
        self.features_names = kwargs.get("features_names")
        self.features = kwargs.get("features")
        self.num_features = len(self.features_names)
        self.target_feature = kwargs.get("target_feature")
        self.num_fuzzy_sets = kwargs.get("num_fuzzy_sets")
        self.fuzzy_sets = kwargs.get("fuzzy_sets")
        self.gain_threshold = kwargs.get("gain_threshold")
        self.max_depth = kwargs.get("max_depth")
        self.max_number_rounds = kwargs.get("max_number_rounds")
        self.min_samples_split_ratio = kwargs.get("min_samples_split_ratio")
        self.v_dict: Dict = None
        self.root_node: FedFRTNode = None
        self.model = None

    def init_frt(self, **kwargs):
        variance = kwargs.get("variance")
        num_samples = kwargs.get("num_samples")
        num_samples_above_thr = kwargs.get("num_samples_above_thr")
        self.root_node: FedFRTNode = FedFRTNode(variance=variance,
                                                num_samples=num_samples,
                                                num_samples_above_thr=num_samples_above_thr,
                                                mean_activation_force=1.0)
        self.v_dict: Dict = dict()
        self.v_dict[1] = [(self.root_node, list(self.features))]

    def build_model(self):
        self.model: FedFRTModel = FedFRTModel(self.features_names, self.root_node)
        return self.model

    def get_model(self) -> FedFRTModel:
        return self.model

    def get_v(self, p_round: int) -> List[Tuple]:
        return self.v_dict.get(p_round)

    def get_root_node(self) -> FedFRTNode:
        return self.root_node

    def grow_tree(self, current_round: int, depth: int, min_samples_split: int, clients_stats: List) -> None:
        next_round = current_round + 1
        v_t_1 = list()
        for node_q, h_q in self.v_dict.get(current_round):
            logger_info(f'depth = {depth}, node_id = {node_q.get_comp_id()} variance = {node_q.variance}, gain = {node_q.gain}')
            max_depth_validation = self.max_depth is not None and self.max_depth == depth
            min_num_split_validation = node_q.num_samples_above_thr < min_samples_split
            logger_info(f'max_depth_validation = {max_depth_validation}, min_num_split_validation = {min_num_split_validation}')
            if max_depth_validation or min_num_split_validation:
                node_q.mark_as_leaf()
                continue

            node_variance = node_q.variance

            best_gain = None
            best_feature = None

            for feature in h_q:
                if len(self.fuzzy_sets[feature]):
                    current_gain = fuzzy_gain(self.num_fuzzy_sets, node_variance, node_q.id, feature, clients_stats)
                    logger_info(f'feature = {feature}, current_gain = {current_gain}')
                    if (best_gain is None or current_gain > best_gain) and current_gain != node_variance:
                        best_gain = current_gain
                        best_feature = feature
                else:
                    logger_info(f"feature = {feature} don't have fuzzy-sets.")

            logger_info(f'depth = {depth}, best_feature = {best_feature}, best_gain = {best_gain}, node_variance = {node_variance}')

            if best_gain and best_gain >= self.gain_threshold:
                new_h_q = list(h_q)
                new_h_q.pop(new_h_q.index(best_feature))
                was_added_nodes = False
                for fs_idx, fs in enumerate(self.fuzzy_sets[best_feature]):
                    data_owner_sufficient_points = sum([stat[node_q.id][best_feature][fs_idx][3] for stat in clients_stats]) >= self.num_features + 1
                    if data_owner_sufficient_points:
                        fs_node_variance = fuzzy_variance([stat[node_q.id][best_feature][fs_idx] for stat in clients_stats])
                        fs_activation_forces = sum([stat[node_q.id][best_feature][fs_idx][2] for stat in clients_stats])
                        fs_node_samples = sum([stat[node_q.id][best_feature][fs_idx][3] for stat in clients_stats])
                        fs_node_samples_above_thr = sum([stat[node_q.id][best_feature][fs_idx][4] for stat in clients_stats])
                        logger_info(f'current_round = {current_round}, fs_node_samples_above_thr = {fs_node_samples_above_thr}')
                        mean_activation_force = fs_activation_forces / fs_node_samples
                        node_q_b_f_t = FedFRTNode(feature=best_feature,
                                                  feature_name=self.features_names[best_feature],
                                                  f_set=fs,
                                                  gain=best_gain,
                                                  depth=current_round,
                                                  parent=node_q,
                                                  variance=fs_node_variance,
                                                  num_samples=fs_node_samples,
                                                  num_samples_above_thr=fs_node_samples_above_thr,
                                                  mean_activation_force=mean_activation_force)
                        node_q.add_children(node_q_b_f_t)
                        was_added_nodes = True
                        if new_h_q:
                            v_t_1.append((node_q_b_f_t, new_h_q))
                        else:
                            node_q_b_f_t.mark_as_leaf()
                    else:
                        logger_info(f'Feature = {best_feature}, fs_idx = {fs_idx}, no enough points.')
                if not was_added_nodes:
                    node_q.mark_as_leaf()
            else:
                node_q.mark_as_leaf()
        self.v_dict[next_round] = v_t_1

    def run_federation(self, fed_clients: List[FedFRTClient]) -> Dict[str, float]:
        federation_stats = dict()
        initial_stats = [client.get_stats_for_root_node() for client in fed_clients]
        root_node_num_samples = sum([client_stats[2] for client_stats in initial_stats])
        root_node_variance = fuzzy_variance(initial_stats)
        logger_info(f'min_samples_split_ratio = {self.min_samples_split_ratio}, root_node_num_samples = {root_node_num_samples}')
        min_samples_split: int = int(self.min_samples_split_ratio * root_node_num_samples)
        self.init_frt(variance=root_node_variance, num_samples=root_node_num_samples,
                      num_samples_above_thr=root_node_num_samples)

        logger_info(f'root_node.variance = {self.root_node.variance}, '
                    f'total num_samples clients = {root_node_num_samples}, '
                    f'min_samples_split = {min_samples_split}')
        """
            FRT Growing
        """
        for round_t in range(1, self.max_number_rounds + 1):
            depth = round_t - 1

            mem_threshold = 0.5 ** (depth + 1)
            # At each round the server transmits frt_t and v_t to data owners
            frt_t, v_t = self.get_root_node(), self.get_v(round_t)
            logger_info(f'round_t = {round_t}, mem_threshold = 0.5 ** {depth + 1} = {mem_threshold}, number of nodes = {len(v_t)}')
            # Each data owner compute the statistics and obfuscate them before sending it to the server
            clients_stats = [fed_cli.compute_stats(round_t, v_t, mem_threshold) for fed_cli in fed_clients]

            self.grow_tree(round_t, depth, min_samples_split, clients_stats)

            next_round = round_t + 1
            v_t_1 = self.get_v(next_round)

            if not v_t_1:
                break
        """
            FRT Start Regression Estimate
        """
        model = self.build_model()
        antecedents = model.get_rules()

        logger_info(f'Total number of rules: {len(antecedents)}')

        consequents, num_samples_by_rule = FedFRTServer._run_compute_consequents(antecedents, fed_clients)
        rule_weights = FedFRTServer._run_compute_rule_weights(antecedents, consequents, fed_clients)

        model.init(consequents, num_samples_by_rule, rule_weights)
        """
            FRT Distribute Model
        """
        for fed_cli in fed_clients:
            fed_cli.send_model(self.model)

        federation_stats['num_nodes'] = model.num_nodes()
        federation_stats['num_leaves'] = model.num_leaves()
        federation_stats['max_depth'] = model.max_rule_length()
        federation_stats['rules'] = model.get_rules_features()
        federation_stats['complexity'] = model.get_model_complexity()
        return federation_stats

    @staticmethod
    def _run_compute_consequents(antecedents, fed_tsk_client_list: List[FedFRTClient]):
        client_responses = []

        for fed_tsk_client in fed_tsk_client_list:
            client_response = fed_tsk_client.compute_matrices_for_consequents(antecedents)
            client_responses.append(client_response)

        consequents = []
        for i in range(len(antecedents)):
            sum_dot_x_t_x_list = np.add.reduce([dot_x_t_x_list[i][0] for dot_x_t_x_list in client_responses])
            sum_dot_x_t_y_list = np.add.reduce([dot_x_t_y_list[i][1] for dot_x_t_y_list in client_responses])

            new_consequents, residuals, rank, s = np.linalg.lstsq(sum_dot_x_t_x_list, sum_dot_x_t_y_list)
            logger_info(f'RULE {i}, rank = {rank}, residuals = {residuals}, s = {s}')
            w_0 = new_consequents[-1]
            new_consequents = np.insert(new_consequents, 0, w_0, axis=0)
            new_consequents = np.delete(new_consequents, -1, axis=0)
            consequents.append(new_consequents)
        consequents = np.array(consequents)
        return consequents, [sum([dot_x_t_y_list[i][2] for dot_x_t_y_list in client_responses]) for i in range(len(antecedents))]

    @staticmethod
    def _run_compute_rule_weights(antecedents,
                                  consequents,
                                  fed_tsk_client_list: List[FedFRTClient]):
        max_ae = None
        min_ae = None
        for fed_tsk_client in fed_tsk_client_list:
            max_ae_client, min_ae_client = fed_tsk_client.compute_rule_weights_step1(antecedents=antecedents,
                                                                                     consequents=consequents)
            if max_ae is None or max_ae < max_ae_client:
                max_ae = max_ae_client
            if min_ae is None or min_ae > min_ae_client:
                min_ae = min_ae_client

        delta_max_min_ae = max_ae - min_ae
        num_of_rules = len(antecedents)
        conf_supp_partial_data_list = []
        for fed_tsk_client in fed_tsk_client_list:
            conf_supp_partial_data = fed_tsk_client.compute_weight_rules_step2(num_of_rules,
                                                                               min_ae,
                                                                               delta_max_min_ae)
            conf_supp_partial_data_list.append(conf_supp_partial_data)

        rule_weight = np.zeros(num_of_rules)

        for rule_idx in range(num_of_rules):
            sum_weighted_membership_value = sum([client_data[rule_idx][0] for client_data in conf_supp_partial_data_list])
            sum_firing_strength = sum([client_data[rule_idx][1] for client_data in conf_supp_partial_data_list])
            num_of_training_samples = sum([client_data[rule_idx][2] for client_data in conf_supp_partial_data_list])

            fuzzy_confidence = (sum_weighted_membership_value / sum_firing_strength) if sum_firing_strength != 0 else 0
            fuzzy_support = sum_weighted_membership_value / num_of_training_samples
            if fuzzy_support + fuzzy_confidence != 0:
                rule_weight[rule_idx] = (2 * fuzzy_support * fuzzy_confidence) / (fuzzy_support + fuzzy_confidence)
            else:
                rule_weight[rule_idx] = 0

        return rule_weight

    def save_model(self, model_output_folder, iteration, num_clients: int = None):
        if num_clients:
            output_file = f'{model_output_folder}/model_m_{num_clients}_it_{iteration}.txt'
        else:
            output_file = f'{model_output_folder}/model_{iteration}.txt'
        with open(output_file, 'w') as f:
            f.write(self.model.generate_rules_str())
        logger_info(f'Model generated: {output_file}')
        output_file_rules_matrix = f'{model_output_folder}/model_matrix_{iteration}.txt'
        model_matrix = self.model.get_model_matrix()
        with open(output_file_rules_matrix, 'w') as f:
            for model_rule in model_matrix:
                line = ",".join([str(item) for item in model_rule])
                f.write(line + '\n')
