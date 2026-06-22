"""
Module to generate diverse counterfactual explanations based on genetic algorithm
with an added causal regularizer in the loss.

- Keeps the DiCE genetic algorithm logic the same.
- Adds an SCM-based causal regularizer.
- Avoids double-penalizing endogenous features:
    * endogenous continuous features are excluded from normal proximity
    * endogenous features are penalized through the causal SCM loss

"""

import copy
import random
import timeit

import numpy as np
import pandas as pd
from raiutils.exceptions import UserConfigValidationException

from dice_ml import diverse_counterfactuals as exp
from dice_ml.constants import ModelTypes
from dice_ml.explainer_interfaces.explainer_base import ExplainerBase


class DiceGeneticCausal(ExplainerBase):

    def __init__(
            self,
            data_interface,
            model_interface,
            scm=None,
            exogenous_features=None,
            endogenous_features=None,
            causal_weight=1.0
    ):
        super().__init__(data_interface, model_interface)
        self.num_output_nodes = None

        self.cfs = []
        self.features_to_vary = []
        self.cf_init_weights = []
        self.loss_weights = []
        self.feature_weights_input = ''

        self.labelencoder = self.data_interface.fit_label_encoders()
        self.predicted_outcome_name = self.data_interface.outcome_name + '_pred'



        # adding the SCM formula
        self.scm = scm or {}

        self.exogenous_features = exogenous_features or []

        # Endogenous variables, i.e. variables determined by SCM equations.
        self.endogenous_features = endogenous_features or list(self.scm.keys())

        # Strength of causal regularizer.
        self.causal_weight = causal_weight

    # CODE SAME AS ORIGINAL
    def update_hyperparameters(self, proximity_weight, sparsity_weight,
                               diversity_weight, categorical_penalty):
        """Update hyperparameters of the loss function"""
        self.proximity_weight = proximity_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.categorical_penalty = categorical_penalty

    # CODE SAME AS DICE ORIGINAL
    def do_loss_initializations(self, yloss_type, diversity_loss_type, feature_weights,
                                encoding='one-hot'):
        """Initializes variables related to main loss function"""

        self.loss_weights = [yloss_type, diversity_loss_type, feature_weights]
        self.yloss_type = yloss_type
        self.diversity_loss_type = diversity_loss_type

        if feature_weights != self.feature_weights_input:
            self.feature_weights_input = feature_weights
            if feature_weights == "inverse_mad":
                normalized_mads = self.data_interface.get_valid_mads(normalized=False)
                feature_weights = {}
                for feature in normalized_mads:
                    feature_weights[feature] = round(1 / normalized_mads[feature], 2)

            feature_weights_list = []
            if encoding == 'one-hot':
                for feature in self.data_interface.encoded_feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        feature_weights_list.append(1.0)
            elif encoding == 'label':
                for feature in self.data_interface.feature_names:
                    if feature in feature_weights:
                        feature_weights_list.append(feature_weights[feature])
                    else:
                        # the weight is inversely proportional to max value
                        feature_weights_list.append(round(1 / self.feature_range[feature].max(), 2))
            self.feature_weights_list = [feature_weights_list]

    # NEW FUNCTION
    def decode_population_to_df(self, arr):
        """
        arr: numpy array shape (n, d) or (d,)
        returns dataframe in original feature space.
        """
        return self.label_decode(arr)

    # NEW FUNCTION
    def compute_causal_proximity_loss(self, cfs):
        """
        Causal regularizer on endogenous features only.

        Logic used here:
        - Exogenous/root features are not causally penalized here because normal DiCE
          proximity already penalizes their deviation from the original instance.
        - Endogenous features are penalized based on deviation from the SCM-implied value.
        - Continuous penalties are normalized by feature range to avoid scale dominance.

        Example:
            If SCM says Duration = f(Credit Amount), then this loss penalizes:
                |Duration_cf - f(Credit Amount_cf)|
            not:
                |Duration_cf - Duration_original|
        """
        if not self.scm:
            return np.zeros(len(cfs))

        cfs_df = self.decode_population_to_df(cfs).reset_index(drop=True)

        losses = []
        valid_endogenous_features = [
            feat for feat in self.endogenous_features
            if feat in self.scm
        ]

        for i in range(len(cfs_df)):
            row = cfs_df.iloc[i].to_dict()
            loss_i = 0.0

            for feat in valid_endogenous_features:
                scm_func = self.scm[feat]["func"]
                target_val = scm_func(row)

                if feat in self.data_interface.continuous_feature_names:
                    feat_min = self.data_interface.data_df[feat].min()
                    feat_max = self.data_interface.data_df[feat].max()
                    feat_range = max(feat_max - feat_min, 1e-6)
                    loss_i += abs(row[feat] - target_val) / feat_range
                else:
                    loss_i += 0.0 if row[feat] == target_val else 1.0

            losses.append(loss_i)

        losses = np.array(losses, dtype=float)
        denom = max(1, len(valid_endogenous_features))
        return losses / denom

    # IDENTICAL AS DICE NORMAL CODE
    def do_random_init(self, num_inits, features_to_vary, query_instance, desired_class, desired_range):
        remaining_cfs = np.zeros((num_inits, self.data_interface.number_of_features))
        kx = 0
        precisions = self.data_interface.get_decimal_precisions()

        while kx < num_inits:
            one_init = np.zeros(self.data_interface.number_of_features)
            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature in features_to_vary:
                    if feature in self.data_interface.continuous_feature_names:
                        one_init[jx] = np.round(
                            np.random.uniform(self.feature_range[feature][0], self.feature_range[feature][1]),
                            precisions[jx]
                        )
                    else:
                        one_init[jx] = np.random.choice(self.feature_range[feature])
                else:
                    one_init[jx] = query_instance[jx]

            if self.is_cf_valid(self.predict_fn_scores(one_init)):
                remaining_cfs[kx] = one_init
                kx += 1

        return remaining_cfs

    # IDENTICAL AS NORMAL DICE
    def do_KD_init(self, features_to_vary, query_instance, cfs, desired_class, desired_range):
        cfs = self.label_encode(cfs)
        cfs = cfs.reset_index(drop=True)

        self.cfs = np.zeros((self.population_size, self.data_interface.number_of_features))
        for kx in range(self.population_size):
            if kx >= len(cfs):
                break

            one_init = np.zeros(self.data_interface.number_of_features)

            for jx, feature in enumerate(self.data_interface.feature_names):
                if feature not in features_to_vary:
                    one_init[jx] = query_instance[jx]
                else:
                    if feature in self.data_interface.continuous_feature_names:
                        if self.feature_range[feature][0] <= cfs.iat[kx, jx] <= self.feature_range[feature][1]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if self.feature_range[feature][0] <= query_instance[jx] <= self.feature_range[feature][1]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = np.random.uniform(
                                    self.feature_range[feature][0], self.feature_range[feature][1]
                                )
                    else:
                        if cfs.iat[kx, jx] in self.feature_range[feature]:
                            one_init[jx] = cfs.iat[kx, jx]
                        else:
                            if query_instance[jx] in self.feature_range[feature]:
                                one_init[jx] = query_instance[jx]
                            else:
                                one_init[jx] = np.random.choice(self.feature_range[feature])

            self.cfs[kx] = one_init
            kx += 1

        new_array = [tuple(row) for row in self.cfs]
        uniques = np.unique(new_array, axis=0)

        if len(uniques) != self.population_size:
            remaining_cfs = self.do_random_init(
                self.population_size - len(uniques), features_to_vary, query_instance, desired_class, desired_range)
            self.cfs = np.concatenate([uniques, remaining_cfs])

    # IDENTICAL AS DICE
    def do_cf_initializations(self, total_CFs, initialization, algorithm, features_to_vary, desired_range,
                              desired_class,
                              query_instance, query_instance_df_dummies, verbose):
        """Initializes CFs and other related variables."""
        self.cf_init_weights = [total_CFs, algorithm, features_to_vary]

        if algorithm == "RandomInitCF":
            self.total_random_inits = total_CFs
            self.total_CFs = 1
        else:
            self.total_random_inits = 0
            self.total_CFs = total_CFs

        self.features_to_vary = features_to_vary

        self.cfs = []
        if initialization == 'random':
            self.cfs = self.do_random_init(
                self.population_size, features_to_vary, query_instance, desired_class, desired_range
            )

        elif initialization == 'kdtree':
            self.dataset_with_predictions, self.KD_tree, self.predictions = \
                self.build_KD_tree(self.data_interface.data_df.copy(),
                                   desired_range, desired_class, self.predicted_outcome_name)

            if self.KD_tree is None:
                self.cfs = self.do_random_init(
                    self.population_size, features_to_vary, query_instance, desired_class, desired_range
                )
            else:
                num_queries = min(len(self.dataset_with_predictions), self.population_size * self.total_CFs)
                indices = self.KD_tree.query(query_instance_df_dummies, num_queries)[1][0]
                KD_tree_output = self.dataset_with_predictions.iloc[indices].copy()
                self.do_KD_init(features_to_vary, query_instance, KD_tree_output, desired_class, desired_range)

        if verbose:
            print("Initialization complete! Generating counterfactuals...")

    # IDENTICAL AS DICE
    def do_param_initializations(self, total_CFs, initialization, desired_range, desired_class,
                                 query_instance, query_instance_df_dummies, algorithm, features_to_vary,
                                 permitted_range, yloss_type, diversity_loss_type, feature_weights,
                                 proximity_weight, sparsity_weight, diversity_weight, categorical_penalty, verbose):
        if verbose:
            print("Initializing initial parameters to the genetic algorithm...")

        self.feature_range = self.get_valid_feature_range(normalized=False)
        if len(self.cfs) != total_CFs:
            self.do_cf_initializations(
                total_CFs, initialization, algorithm, features_to_vary, desired_range, desired_class,
                query_instance, query_instance_df_dummies, verbose
            )
        else:
            self.total_CFs = total_CFs

        self.do_loss_initializations(yloss_type, diversity_loss_type, feature_weights, encoding='label')
        self.update_hyperparameters(proximity_weight, sparsity_weight, diversity_weight, categorical_penalty)

    def _generate_counterfactuals(self, query_instance, total_CFs, initialization="kdtree",
                                  desired_range=None, desired_class="opposite", proximity_weight=0.2,
                                  sparsity_weight=0.2, diversity_weight=5.0, categorical_penalty=0.1,
                                  algorithm="DiverseCF", features_to_vary="all", permitted_range=None,
                                  yloss_type="hinge_loss", diversity_loss_type="dpp_style:inverse_dist",
                                  feature_weights="inverse_mad", stopping_threshold=0.5, posthoc_sparsity_param=0.1,
                                  posthoc_sparsity_algorithm="binary", maxiterations=500, thresh=1e-2, verbose=False):
        """Generates diverse counterfactual explanations"""

        if not hasattr(self.data_interface, 'data_df') and initialization == "kdtree":
            raise UserConfigValidationException(
                "kd-tree initialization is not supported for private data"
                " interface because training data to build kd-tree is not available."
            )

        self.population_size = 10 * total_CFs
        self.start_time = timeit.default_timer()

        features_to_vary = self.setup(features_to_vary, permitted_range, query_instance, feature_weights)

        query_instance_orig = query_instance
        query_instance_orig = self.data_interface.prepare_query_instance(query_instance=query_instance_orig)
        query_instance = self.data_interface.prepare_query_instance(query_instance=query_instance)

        self.num_output_nodes = None
        if self.model.model_type == ModelTypes.Classifier:
            self.num_output_nodes = self.model.get_num_output_nodes2(query_instance)

        query_instance = self.label_encode(query_instance)
        query_instance = np.array(query_instance.values[0])
        self.x1 = query_instance

        test_pred = self.predict_fn_scores(query_instance)
        self.test_pred = test_pred
        desired_class = self.misc_init(stopping_threshold, desired_class, desired_range, test_pred[0])

        query_instance_df_dummies = pd.get_dummies(query_instance_orig)
        for col in self.data_interface.get_all_dummy_colnames():
            if col not in query_instance_df_dummies.columns:
                query_instance_df_dummies[col] = 0

        self.do_param_initializations(total_CFs, initialization, desired_range, desired_class, query_instance,
                                      query_instance_df_dummies, algorithm, features_to_vary, permitted_range,
                                      yloss_type, diversity_loss_type, feature_weights, proximity_weight,
                                      sparsity_weight, diversity_weight, categorical_penalty, verbose)

        query_instance_df = self.find_counterfactuals(query_instance, desired_range, desired_class, features_to_vary,
                                                      maxiterations, thresh, verbose)

        desired_class_param = self.decode_model_output(pd.Series(self.target_cf_class[0]))[0] \
            if hasattr(self, 'target_cf_class') else desired_class

        return exp.CounterfactualExamples(data_interface=self.data_interface,
                                          test_instance_df=query_instance_df,
                                          final_cfs_df=self.final_cfs_df,
                                          final_cfs_df_sparse=self.final_cfs_df_sparse,
                                          posthoc_sparsity_param=posthoc_sparsity_param,
                                          desired_range=desired_range,
                                          desired_class=desired_class_param,
                                          model_type=self.model.model_type)

    # IDENTICAL AS DICE
    def predict_fn_scores(self, input_instance):
        """Returns prediction scores."""
        input_instance = self.label_decode(input_instance)
        out = self.model.get_output(input_instance)
        if self.model.model_type == ModelTypes.Classifier and out.shape[1] == 1:
            out = np.hstack((1 - out, out))
        return out

    # IDENTICAL AS DICE
    def predict_fn(self, input_instance):
        """Returns actual prediction."""
        input_instance = self.label_decode(input_instance)
        preds = self.model.get_output(input_instance, model_score=False)
        return preds

    # IDENTICAL AS DICE
    def _predict_fn_custom(self, input_instance, desired_class):
        """Checks that the maximum predicted score lies in the desired class."""
        input_instance = self.label_decode(input_instance)
        output = self.model.get_output(input_instance, model_score=True)
        if self.model.model_type == ModelTypes.Classifier and np.array(output).shape[1] == 1:
            output = np.hstack((1 - output, output))
        desired_class = int(desired_class)
        maxvalues = np.max(output, 1)
        predicted_values = np.argmax(output, 1)

        for i in range(len(output)):
            if output[i][desired_class] == maxvalues[i]:
                predicted_values[i] = desired_class

        return predicted_values

    # IDENTICAL AS DICE
    def compute_yloss(self, cfs, desired_range, desired_class):
        """Computes the first part (y-loss) of the loss function."""
        yloss = 0.0
        if self.model.model_type == ModelTypes.Classifier:
            predicted_value = np.array(self.predict_fn_scores(cfs))
            if self.yloss_type == 'hinge_loss':
                maxvalue = np.full((len(predicted_value)), -np.inf)
                for c in range(self.num_output_nodes):
                    if c != desired_class:
                        maxvalue = np.maximum(maxvalue, predicted_value[:, c])
                yloss = np.maximum(0, maxvalue - predicted_value[:, int(desired_class)])
            return yloss

        elif self.model.model_type == ModelTypes.Regressor:
            predicted_value = self.predict_fn(cfs)
            if self.yloss_type == 'hinge_loss':
                yloss = np.zeros(len(predicted_value))
                for i in range(len(predicted_value)):
                    if not desired_range[0] <= predicted_value[i] <= desired_range[1]:
                        yloss[i] = min(abs(predicted_value[i] - desired_range[0]),
                                       abs(predicted_value[i] - desired_range[1]))
            return yloss

    # MODIFIED MINIMALLY FROM DICE
    def compute_proximity_loss(self, x_hat_unnormalized, query_instance_normalized):
        """
        Compute weighted proximity distance.

        Minimal change from original DiCE:
        - Original DiCE uses all continuous_feature_indexes.
        - This version excludes endogenous continuous features, because they are
          already penalized by compute_causal_proximity_loss().
        """
        x_hat = self.data_interface.normalize_data(x_hat_unnormalized)

        proximity_feature_indexes = [
            self.data_interface.feature_names.index(feat)
            for feat in self.data_interface.continuous_feature_names
            if feat not in self.endogenous_features
        ]

        # If all continuous features are endogenous, there is no normal proximity term.
        if len(proximity_feature_indexes) == 0:
            return np.zeros(len(x_hat_unnormalized))

        feature_weights = np.array(
            [self.feature_weights_list[0][i] for i in proximity_feature_indexes],
            dtype=float
        )

        product = np.multiply(
            abs(x_hat - query_instance_normalized)[:, proximity_feature_indexes],
            feature_weights
        )

        proximity_loss = np.sum(product, axis=1)
        return proximity_loss / sum(feature_weights)

    # IDENTICAL AS DICE
    def compute_sparsity_loss(self, cfs):
        """Compute sparsity as fraction of changed features."""
        sparsity_loss = np.count_nonzero(cfs - self.x1, axis=1)
        return sparsity_loss / len(self.data_interface.feature_names)

    # MODIFIED: only adds causal_proximity_loss term
    def compute_loss(self, cfs, desired_range, desired_class):
        """
        Computes the overall loss with added causal regularizer.
        """
        self.yloss = self.compute_yloss(cfs, desired_range, desired_class)

        self.proximity_loss = self.compute_proximity_loss(cfs, self.query_instance_normalized) \
            if self.proximity_weight > 0 else 0.0

        self.sparsity_loss = self.compute_sparsity_loss(cfs) \
            if self.sparsity_weight > 0 else 0.0

        self.causal_proximity_loss = self.compute_causal_proximity_loss(cfs) \
            if self.scm else 0.0

        total_loss = (
                self.yloss
                + (self.proximity_weight * self.proximity_loss)
                + (self.sparsity_weight * self.sparsity_loss)
                + (self.causal_weight * self.causal_proximity_loss)
        )

        self.loss = np.reshape(np.array(total_loss), (-1, 1))
        index = np.reshape(np.arange(len(cfs)), (-1, 1))
        self.loss = np.concatenate([index, self.loss], axis=1)
        return self.loss

    # SAME AS DICE
    def mate(self, k1, k2, features_to_vary, query_instance):
        """Performs mating and produces new offsprings"""
        one_init = np.zeros(self.data_interface.number_of_features)
        for j in range(self.data_interface.number_of_features):
            gp1 = k1[j]
            gp2 = k2[j]
            feat_name = self.data_interface.feature_names[j]

            prob = random.random()

            if prob < 0.40:
                one_init[j] = gp1
            elif prob < 0.80:
                one_init[j] = gp2
            else:
                if feat_name in features_to_vary:
                    if feat_name in self.data_interface.continuous_feature_names:
                        one_init[j] = np.random.uniform(
                            self.feature_range[feat_name][0],
                            self.feature_range[feat_name][1]
                        )
                    else:
                        one_init[j] = np.random.choice(self.feature_range[feat_name])
                else:
                    one_init[j] = query_instance[j]
        return one_init

    # SAME AS DICE
    def find_counterfactuals(self, query_instance, desired_range, desired_class,
                             features_to_vary, maxiterations, thresh, verbose):
        """Finds counterfactuals by generating cfs through the genetic algorithm"""
        population = self.cfs.copy()
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        cfs_preds = [np.inf] * self.total_CFs
        to_pred = None

        self.query_instance_normalized = self.data_interface.normalize_data(self.x1)
        self.query_instance_normalized = self.query_instance_normalized.astype('float')

        while iterations < maxiterations and self.total_CFs > 0:
            if abs(previous_best_loss - current_best_loss) <= thresh and \
                    (self.model.model_type == ModelTypes.Classifier and all(i == desired_class for i in cfs_preds) or
                     (self.model.model_type == ModelTypes.Regressor and
                      all(desired_range[0] <= i <= desired_range[1] for i in cfs_preds))):
                stop_cnt += 1
            else:
                stop_cnt = 0

            if stop_cnt >= 5:
                break

            previous_best_loss = current_best_loss
            population = np.unique(tuple(map(tuple, population)), axis=0)

            population_fitness = self.compute_loss(population, desired_range, desired_class)
            population_fitness = population_fitness[population_fitness[:, 1].argsort()]

            current_best_loss = population_fitness[0][1]
            to_pred = np.array([population[int(tup[0])] for tup in population_fitness[:self.total_CFs]])

            if self.total_CFs > 0:
                if self.model.model_type == ModelTypes.Classifier:
                    cfs_preds = self._predict_fn_custom(to_pred, desired_class)
                else:
                    cfs_preds = self.predict_fn(to_pred)

            top_members = self.total_CFs
            new_generation_1 = np.array([population[int(tup[0])] for tup in population_fitness[:top_members]])

            rest_members = self.population_size - top_members
            new_generation_2 = None
            if rest_members > 0:
                new_generation_2 = np.zeros((rest_members, self.data_interface.number_of_features))
                for new_gen_idx in range(rest_members):
                    top_half = int(len(population) / 2)
                    parent1_idx = random.randrange(top_half)
                    parent2_idx = random.randrange(top_half)
                    parent1 = population[parent1_idx]
                    parent2 = population[parent2_idx]
                    child = self.mate(parent1, parent2, features_to_vary, query_instance)
                    new_generation_2[new_gen_idx] = child

            if new_generation_2 is not None:
                if self.total_CFs > 0:
                    population = np.concatenate([new_generation_1, new_generation_2])
                else:
                    population = new_generation_2
            else:
                raise SystemError("The number of total_CFs is greater than the population size!")

            iterations += 1

        self.cfs_preds = []
        self.final_cfs = []
        i = 0
        while i < self.total_CFs:
            predictions = self.predict_fn_scores(population[i])[0]
            if self.is_cf_valid(predictions):
                self.final_cfs.append(population[i])
                if not isinstance(predictions, (np.floating, float)) and len(predictions) > 1:
                    self.cfs_preds.append(np.argmax(predictions))
                else:
                    self.cfs_preds.append(predictions)
            i += 1

        query_instance_df = self.label_decode(query_instance)
        query_instance_df[self.data_interface.outcome_name] = self.get_model_output_from_scores(self.test_pred)
        self.final_cfs_df = self.label_decode_cfs(self.final_cfs)
        self.final_cfs_df_sparse = copy.deepcopy(self.final_cfs_df)

        if self.final_cfs_df is not None:
            self.final_cfs_df[self.data_interface.outcome_name] = self.cfs_preds
            self.final_cfs_df_sparse[self.data_interface.outcome_name] = self.cfs_preds
            self.round_to_precision()

        query_instance_df, self.final_cfs_df, self.final_cfs_df_sparse = \
            self.decode_to_original_labels(query_instance_df, self.final_cfs_df, self.final_cfs_df_sparse)

        self.elapsed = timeit.default_timer() - self.start_time
        m, s = divmod(self.elapsed, 60)

        if verbose:
            if len(self.final_cfs) == self.total_CFs:
                print('Diverse Counterfactuals found! total time taken: %02d' % m,
                      'min %02d' % s, 'sec')
            else:
                print('Only %d (required %d) ' % (len(self.final_cfs), self.total_CFs),
                      'Diverse Counterfactuals found for the given configuration, perhaps ',
                      'change the query instance or the features to vary...',
                      '; total time taken: %02d' % m, 'min %02d' % s, 'sec')

        return query_instance_df

    # SAME AS DICE
    def label_encode(self, input_instance):
        for column in self.data_interface.categorical_feature_names:
            input_instance[column] = self.labelencoder[column].transform(input_instance[column])
        return input_instance

    # SAME AS DICE
    def label_decode(self, labelled_input):
        """Transforms label encoded data back to categorical values"""
        num_to_decode = 1
        if len(labelled_input.shape) > 1:
            num_to_decode = len(labelled_input)
        else:
            labelled_input = [labelled_input]

        input_instance = []

        for j in range(num_to_decode):
            temp = {}
            for i in range(len(labelled_input[j])):
                feat_name = self.data_interface.feature_names[i]
                if feat_name in self.data_interface.categorical_feature_names:
                    enc = self.labelencoder[feat_name]
                    val = enc.inverse_transform(np.array([labelled_input[j][i]], dtype=np.int32))
                    temp[feat_name] = val[0]
                else:
                    temp[feat_name] = labelled_input[j][i]
            input_instance.append(temp)

        input_instance_df = pd.DataFrame(input_instance, columns=self.data_interface.feature_names)
        return input_instance_df

    def label_decode_cfs(self, cfs_arr):
        ret_df = None
        if cfs_arr is None:
            return None
        for cf in cfs_arr:
            df = self.label_decode(cf)
            if ret_df is None:
                ret_df = df
            else:
                ret_df = pd.concat([ret_df, df], ignore_index=True)
        return ret_df

    def get_valid_feature_range(self, normalized=False):
        ret = self.data_interface.get_valid_feature_range(self.feature_range, normalized=normalized)
        for feat_name in self.data_interface.categorical_feature_names:
            ret[feat_name] = self.labelencoder[feat_name].transform(ret[feat_name])
        return ret
