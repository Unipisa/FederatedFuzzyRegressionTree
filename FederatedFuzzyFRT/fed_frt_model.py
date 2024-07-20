#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import numpy as np
from typing import List
from FederatedFuzzyFRT.fed_frt_node import FedFRTNode
from FederatedFuzzyFRT.utils.stats_utils import compute_firing_strengths


class FedFRTModel:

    def __init__(self, features_names: List[str], root_node: FedFRTNode):
        self._root_node = root_node
        self._antecedents = self._fetch_rules()
        self.features_names = features_names
        self._consequents = None
        self._weights = None

    def _fetch_rules(self):
        final_leave_paths = []
        final_leave_paths_append = final_leave_paths.append
        leave_paths = [[node] for node in self._root_node.get_child()]
        leave_paths_append = leave_paths.append
        leave_paths_pop = leave_paths.pop

        while leave_paths:
            top_path = leave_paths_pop(0)
            last_node = top_path[-1]
            if last_node.is_leaf:
                final_leave_paths_append(top_path)
            else:
                child = last_node.child
                for children in child:
                    leave_paths_append(top_path + [children])
        samples_by_rule = []
        for rule in final_leave_paths:
            last_node = rule[-1]
            samples_by_rule.append(last_node.num_samples)
        self.samples_by_rule = samples_by_rule
        return final_leave_paths

    def get_model_matrix(self):
        num_features = len(self.features_names)
        num_elements_by_row = num_features * 2 + 1
        matrix = [[-1] * num_elements_by_row for i in range(len(self._antecedents))]
        for rule_idx, rule in enumerate(self._antecedents):
            for node in rule:
                matrix[rule_idx][node.feature] = node.fSet.get_term()

        for consequent_idx, consequent in enumerate(self._consequents):
            for coefficient_idx, coefficient in enumerate(consequent):
                matrix[consequent_idx][num_features + coefficient_idx] = coefficient
        return matrix

    def get_rules(self):
        return self._antecedents

    def get_rules_features(self):
        return [[{
            'feature': node.feature,
            'name': node.feature_name,
            'fuzzy_set_index': node.fSet.get_term()
        } for node in rule] for rule in self._antecedents]

    def init(self, consequents, num_samples_by_rule, rule_weights):
        self._consequents = consequents
        self._num_samples_by_rule = num_samples_by_rule
        self._rule_weights = rule_weights
        self._cache_activation_force = {rule[-1].get_comp_id(): rule[-1].mean_activation_force for rule in self._antecedents}

    def _get_rule_maximum_weight(self):
        index_rule_weight = self._rule_weights.argmax()
        return self._antecedents[index_rule_weight], self._consequents[index_rule_weight], index_rule_weight

    def _single_predict(self, x):

        firing_strengths = compute_firing_strengths(x, self._antecedents, self._cache_activation_force)
        is_all_zero = np.all((firing_strengths == 0))
        num_activated_rules = 0

        if is_all_zero:
            _, consequent_weights, index_max_rules = self._get_rule_maximum_weight()
        else:
            max_values_firing = firing_strengths.max()

            index_same_firing_strengths = np.where(firing_strengths == max_values_firing)[0]

            num_activated_rules = len(index_same_firing_strengths)

            # Retrieve the relative weights
            rule_weight = self._rule_weights[index_same_firing_strengths]

            index_max_rules = index_same_firing_strengths[rule_weight.argmax()]

            consequent_weights = self._consequents[index_max_rules]

        w0 = consequent_weights[0]

        consequent_weights = consequent_weights[1:]

        result = (consequent_weights * x).sum() + w0

        return result, index_max_rules, num_activated_rules

    def predict(self, X):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like  of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        predict = self._single_predict
        return np.array(list(map(predict, X)))

    def generate_rules_str(self) -> str:
        rules_str = ""
        for idx, (antecedents, consequents, weight, num_samples) in enumerate(zip(self._antecedents, self._consequents, self._rule_weights, self.samples_by_rule)):
            rules_str += f"R{idx}:\t{self._get_antecedent_as_string(antecedents)}\n" + \
                         '\t\tTHEN: ' + self._get_consequent_as_string(consequents) + '\n' + \
                         f'\t\t(Num Samples: {num_samples})\n' + \
                         f'\t\t(Rule Weight: {weight:.2e})\n\n'

        return rules_str

    def _get_antecedent_as_string(self, antecedents: List) -> str:
        variables_name = self.features_names
        conditions = []
        conditions_append = conditions.append

        for idx, antecedent in enumerate(antecedents):
            variable_name = variables_name[antecedent.feature]
            conditions_append(f"({variable_name} IS FS{antecedent.fSet.get_term()})")

        conditions_str = " AND ".join(conditions)
        return f'IF {conditions_str}'

    def _get_consequent_as_string(self, consequents: List) -> str:
        parts = []
        parts_append = parts.append
        parts_append(f"{consequents[0]:.2e}")
        variables_name = self.features_names
        for consequent, variable_name in zip(consequents[1:], variables_name):
            parts_append(f"{'+ ' if consequent > 0 else '- '}({abs(consequent):.2e} * {variable_name})")
        return " ".join(parts)

    def num_nodes(self) -> np.array:
        return np.sum(list(map(lambda x: x._num_descendants(), self._root_node.child)))

    def num_leaves(self):
        return self._root_node._num_leaves()

    def max_rule_length(self):
        return max(list(map(lambda l: len(l), self._antecedents)))

    def get_model_complexity(self):
        total_nodes = self.num_nodes()
        leave_nodes = self.num_leaves()
        internal_nodes = total_nodes - leave_nodes
        return internal_nodes + leave_nodes * (len(self.features_names) + 1)
