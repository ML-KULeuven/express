import copy
from collections import defaultdict

import numpy as np
import pandas as pd
from cpmpy.expressions.variables import NegBoolView
from model.lg_problems import LGProblem
from utils.constants import *
from utils.utils import create_folders, create_dataframe_from_dicts, tune_lr, random_zero_cell, average_first_dicts
from utils.utils_lgp import generate_steps_lgps
from utils.utils_sudoku import generate_steps_sudoku, compute_norm_sudoku
import time
from tqdm import tqdm
import random

class PreferenceElicitationFramework:
    def __init__(self, logic_puzzles_set, oracle, no_oracle, initial_weights, instance_evaluation, df_steps_evaluation,
                 output_location, batch_size=1, max_data=100, time_eval=10, lr=0.1,
                 real_human=False, output_file_name=None, normalized=0, frozen_steps=None,
                 type_diversification='baseline', problem_type='sudoku',exploration_root=1,
                 type_steps=None,diver_value=2):
        """
        Initialize the preference elicitation framework.

        Args:
            logic_puzzles_set: List of puzzle instances for training.
            oracle: Oracle object for labelling (can be Human or simulated).
            no_oracle: User index.
            initial_weights: Dictionary of initial weights for objectives.
            instance_evaluation: Instance for evaluation.
            df_steps_evaluation: DataFrame of steps for evaluation.
            output_location: Path to save results.
            batch_size: Number of queries per iteration.
            max_data: Maximum number of labelled pairs.
            time_eval: Frequency of evaluation.
            lr: Learning rate.
            real_human: If True, disables automatic evaluation.
            output_file_name: Optional output file name.
            normalized: Normalization type (int), 0: None, 1: Nadir Norm., 2: Local Norm., 3: Cumulative Norm..
            frozen_steps: Predefined steps for training.
            type_diversification: Diversification strategy ('baseline', 'disjunction', 'w_disjunction', 'cpucb_disjunction').
            problem_type: 'sudoku' or 'lgp'.
            exploration_root: Exploration parameter.
            type_steps: Step selection strategy.
            diver_value: Diversification value, used for computing the exploration value for UCB (default is 2).
        """
        # Store all input parameters and initialize state variables
        self.logic_puzzles_set = logic_puzzles_set
        self.oracle = oracle
        self.indifference = 0
        self.labelling = 0
        self.problem_type = problem_type
        # Set objectives based on problem type
        if self.problem_type=='sudoku':
            self.objectives = OBJECTIVES_SUDOKU
        if self.problem_type == 'lgp':
            self.objectives = OBJECTIVES_LGP
        self.no_oracle = no_oracle
        self.max_data = max_data
        self.batch_size = batch_size
        self.initial_weights = initial_weights.copy()
        self.weights_learned = initial_weights
        self.no_iter = 1
        self.labelled = 0
        self.lr = lr
        self.previous_explanations = None
        self.all_explanations = []
        self.current_index = -1
        self.instance_evaluation = instance_evaluation
        self.df_steps_evaluation = df_steps_evaluation
        self.regret_list = []
        self.time_list = []
        self.output_location = output_location
        self.time_eval = time_eval
        self.real_human = real_human
        self.output_file_name = output_file_name
        self.df_difference = pd.DataFrame()
        self.normalized = normalized
        self.diver_value = diver_value
        # Set normalization values for objectives
        self.norm_list = []
        if self.normalized!=1:
            self.normalization_values = {}
            for obj in self.objectives:
                self.normalization_values[obj] = 1
        if self.normalized==1:
            self.normalization_values = {}
            if self.problem_type == 'sudoku':
                for obj in self.objectives:
                    self.normalization_values[obj] = OBJECTIVE_NORMALIZED_SUDOKU[obj]
            elif self.problem_type == 'lgp':
                for obj in self.objectives:
                    self.normalization_values[obj] = OBJECTIVES_NORMALIZED_LGP[obj]
        self.type_steps = type_steps
        self.frozen_steps = frozen_steps
        self.index_steps = 0
        self.type_diversification = type_diversification
        self.exploration_root = exploration_root
        self.trade_offs = {obj: 0 for obj in self.objectives}           # N of the UCB formulation in the paper 
        self.trade_offs_count = {obj: 0 for obj in self.objectives}     # Count number of pairs in which the solution picked has that sub-objective improved
        self.weight_list = []
        create_folders(self.output_location)

    def start(self):
        """
        Main loop for preference elicitation and learning.

        Output: Returns the learned weights after training.

        This function start the preference elicitation approach.
        It updates the weights, and periodically evaluates regret. It saves the dataset to CSV in the given output location.
        """
        dataset = []
        self.dataset_for_lr = []
        # Progress bar for labelling process
        with tqdm(total=self.max_data, desc="Labelling progress") as pbar:
            while self.labelled < self.max_data:
                dataset_for_training = []
                queries = self.acquisition_function()  # Select queries for labelling
                batch_no = 0
                for el in queries:
                    # Query the oracle for a label between two explanations
                    picked,missed,val_1,val_2 = self.oracle.label(el)
                    # label is defined and stored for data analysis
                    if picked==-1:  
                        label = 'Draw'
                    elif missed:
                        label = 'Miss'
                    else:
                        label = picked
                    self.labelled +=1
                    # Extract objective values for both explanations
                    obj_1 = [el[0][key] for key in self.objectives]
                    obj_2 = [el[1][key] for key in self.objectives]
                    #definition dataset containing data pairs, we also include if the user misslabel for data analysis
                    dataset.append([self.no_iter,batch_no,obj_1,val_1,obj_2,val_2,self.lr,label])

                    # Save dataset to CSV for analysis
                    df_dataset = pd.DataFrame(dataset,columns=['iter','no batch','features 1',
                                                               'obj value 1','features 2',
                                                               'obj value 2','lr selected','picked'])
                    df_dataset.to_csv(f'{self.output_location}' + f'/dataset.csv')
                    # If user express preference, update trade-off statistics by increasing the num,ber and training set
                    if picked >= 0:
                        for key, v1, v2 in zip(self.objectives, obj_1, obj_2): self.trade_offs[key] += v1 != v2 #increase the value of trade_off if key objective changes in the pair

                        for key, v1, v2 in zip(self.objectives, obj_1, obj_2):
                            if picked == 0:  # obj_1 is picked
                                if v1 < v2:
                                    self.trade_offs_count[key] += 1
                            elif picked == 1:  # obj_2 is picked
                                if v2 < v1:
                                    self.trade_offs_count[key] += 1

                        dataset_for_training.append([el[picked],el[1-picked]])
                        self.dataset_for_lr.append([el[picked],el[1-picked]])
                    batch_no += 1

                # Update weights using labelled data
                self.weights_learned = self.train(dataset_for_training, self.weights_learned, self.normalization_values)

                # Periodically evaluate regret and performance
                if self.batch_size < self.time_eval:
                    if self.no_iter % self.time_eval == 0:
                        self.evaluate()
                self.no_iter+=1
                pbar.update(1)
        return self.weights_learned

    def train(self,dataset,starting_weights,normalization_values):
        """
        Update weights using labelled pairs and normalization.

        Args:
            dataset: List of labelled pairs (dicts of features).
            starting_weights: Dictionary of current weights.
            normalization_values: Dictionary of normalization values for each objective.

        Returns:
            Updated weights dictionary.

        This function applies a simple gradient update to the weights based on the difference
        between paired explanations, normalized by the objective values.
        """
        start = starting_weights.copy()
        difference = defaultdict(int)
        if len(dataset)>0:
            # For each objective, compute the average difference and update the weight
            for key in self.objectives:
                for pair in dataset:
                    difference[key] = (pair[1][key] - pair[0][key])/normalization_values[key]
                difference[key] = difference[key]
                starting_weights[key] = max(1e-3,starting_weights[key] + self.lr*difference[key]) #the weight can not be negative, so we set a minimum value

        return starting_weights

    def evaluate(self):
        """
        Evaluate regret and prediction accuracy.

        Input: None (uses class attributes).
        Output: Saves regret and comparison results to CSV files.

        This function computes the regret (difference between predicted and ground truth explanations)
        using the current weights, and saves detailed results for analysis.
        """
        if not self.real_human:
            print('Start evaluation:')
            predicted = []
            if self.problem_type == 'sudoku':
                # For each evaluation step, generate an explanation using current weights
                for index, row in tqdm(self.df_steps_evaluation.iterrows(), total=len(self.df_steps_evaluation)):
                    sudoku_eval = row['grid'].replace(' ', ', ')
                    sudoku_eval = np.array(eval(sudoku_eval))

                    exp,_,_ = generate_steps_sudoku(to_return=True, grid=sudoku_eval,
                                                    feature_weights=self.weights_learned.copy(), no_user=1,
                                                    sequential_sudoku=-1, normalization=self.normalization_values,
                                                    fact_to_exp=None, is_tqdm=False)
                    predicted.append(exp)

                df_predicted_steps = pd.DataFrame(predicted)
                dict_features_weights = self.oracle.get_dict_feature_weights()

                # Compute obj value as weighted sum for predicted and ground truth explanations
                weighted_df_predicted = df_predicted_steps[dict_features_weights.keys()].mul(dict_features_weights)
                weighted_df_predicted = weighted_df_predicted.sum(axis=1)

                weighted_df_gt = self.df_steps_evaluation[dict_features_weights.keys()].mul(dict_features_weights)
                weighted_df_gt = weighted_df_gt.sum(axis=1)
                difference_df_abs = (weighted_df_predicted - weighted_df_gt)
                difference_df = (weighted_df_predicted - weighted_df_gt) / weighted_df_gt

                # Error handling for problematic rows   # Check if any values in difference_df are negative, small negative values can be due to numerical errors so are not considered
                try:
                    assert (difference_df >= 0).all().all(), "Some values in difference_df are not positive"
                except AssertionError as e:
                    create_folders('./errors')
                    error_filename = f"./errors/error_log_{self.lr}_{self.type_diversification}_{self.no_oracle}.txt"
                    problematic_rows = difference_df < 1e-3  # Boolean mask for problematic rows

                    with open(error_filename, "w") as f:
                        f.write(str(e))
                        f.write(
                            f"\nlr {self.lr} and diversification {self.type_diversification} and oracle {self.no_oracle}:\n")

                        f.write("\nDifference DF (only problematic rows):\n")
                        f.write(difference_df[problematic_rows].to_string())

                        f.write("\n\nPredicted Steps DF (only problematic rows):\n")
                        f.write(df_predicted_steps.loc[problematic_rows, dict_features_weights.keys()].to_string())

                        f.write("\n\nGround Truth Steps DF (only problematic rows):\n")
                        f.write(self.df_steps_evaluation.loc[problematic_rows, dict_features_weights.keys()].to_string())

                # Save regret for all steps, used for nal
                difference_df = ((weighted_df_predicted - weighted_df_gt) / weighted_df_gt).clip(lower=0)
                df_comparison = pd.DataFrame({
                    'predicted': weighted_df_predicted,
                    'ground_truth': weighted_df_gt,
                    'difference_df_relative': difference_df,
                    'difference_df_abs': difference_df_abs,
                })

                regret = difference_df.mean().mean()
                row = [self.labelled]
                row.append(str([float(value) for value in self.weights_learned.values()]))
                row.append(regret)
                self.regret_list.append(row)
                df = pd.DataFrame(self.regret_list, columns=['labelled data', 'weights', 'regret'])
                df.to_csv(f'{self.output_location}' + f'/regret.csv')
                df_comparison.to_csv(
                    f'{self.output_location}' + f'/regret_info_{int(self.no_iter / self.time_eval)}.csv')
                print(f'Regret: {regret}')
            if self.problem_type == 'lgp':
                # For logic grid problems, generate explanations and compute regret similarly
                facts = self.instance_evaluation[0][0].copy()
                constraints = self.instance_evaluation[0][1].copy()
                explainable_facts = self.instance_evaluation[0][2].copy()
                dict_constraint_type = self.instance_evaluation[0][3].copy()
                dict_adjacency = self.instance_evaluation[0][4].copy()


                for index, row in tqdm(self.df_steps_evaluation.iterrows(), total=len(self.df_steps_evaluation)):

                    exp,_,_ = generate_steps_lgps(facts.copy(), constraints, explainable_facts,
                                                  dict_constraint_type, dict_adjacency,
                                                  single_fact=None,
                                                  feature_weights=self.weights_learned, is_tqdm=False,
                                                  normalization = self.normalization_values, to_return=True,
                                                  counter=1)

                    if row['explained'].startswith("~"):
                        target_name = row['explained'][1:]
                    else:
                        target_name = row['explained']

                    for i, var in enumerate(explainable_facts):
                        if var.name == target_name:
                            if row['explained'].startswith("~"):
                                facts.append(~var)
                            else:
                                facts.append(var)
                            explainable_facts.remove(var)
                            break
                    predicted.append(exp)


                df_predicted_steps = pd.DataFrame(predicted)
                dict_features_weights = self.oracle.get_dict_feature_weights()

                weighted_df_predicted = df_predicted_steps[dict_features_weights.keys()].mul(dict_features_weights)
                weighted_df_predicted = weighted_df_predicted.sum(axis=1)

                weighted_df_gt = self.df_steps_evaluation[dict_features_weights.keys()].mul(dict_features_weights)
                weighted_df_gt =  weighted_df_gt.sum(axis=1)
                difference_df_abs = (weighted_df_predicted - weighted_df_gt)
                difference_df = (weighted_df_predicted - weighted_df_gt)/weighted_df_gt

                df_comparison = pd.DataFrame({
                    'predicted': weighted_df_predicted,
                    'ground_truth': weighted_df_gt,
                    'difference_df_relative': difference_df,
                    'difference_df_abs': difference_df_abs,
                })

                # save regret for all steps and average regret, used for analysis
                regret = difference_df.mean().mean()
                row = [self.labelled]
                row.append(str([float(value) for value in self.weights_learned.values()]))
                row.append(regret)
                self.regret_list.append(row)
                df = pd.DataFrame(self.regret_list,columns=['labelled data','weights','regret'])
                df.to_csv(f'{self.output_location}'+f'/regret.csv')
                df_comparison.to_csv(f'{self.output_location}' + f'/regret_info_{int(self.no_iter / self.time_eval)}.csv')
                print(f'Regret: {regret}')

    def acquisition_function(self):
        """
        Select queries for labelling (acquisition function).

        Input: None (uses class attributes).
        Output: List of query pairs for labelling.

        This function selects which explanations to present to the oracle for labelling.
        """
        batch = []
        data = []
        index_problem_before = self.current_index
        if self.frozen_steps is None: # If not SMUS generation
            while len(batch) < self.batch_size:
                self.current_index = self.problem_selection(index_problem_before)
                grid_problem_state = self.state_selection(sequential=(self.current_index==index_problem_before),
                                                          state=self.logic_puzzles_set[self.current_index])
                if self.type_steps == 'random': # Randomly select step
                    if self.problem_type=='sudoku':
                        row,col = random_zero_cell(grid_problem_state)
                        explanations = self.fact_and_exp_steps_selection(grid_problem_state,[row,col])
                    else:
                        state_plus_single_exp = grid_problem_state.copy()
                        choice = random.choice(list(grid_problem_state[2]))
                        state_plus_single_exp[2] = {choice}
                        self.emergency = grid_problem_state[2]
                        explanations = self.fact_and_exp_steps_selection(state_plus_single_exp)
                else:   # Dynamically select step based on the type of problem
                    explanations = self.fact_and_exp_steps_selection(grid_problem_state)
                batch.append([explanations[0],explanations[1]])
                self.previous_explanations = [explanations[0],explanations[1]]
        else:
            # If using frozen steps, select queries from predefined steps
            if self.problem_type == 'sudoku':
                self.current_index = self.problem_selection(index_problem_before)

                state,explained = self.frozen_steps[self.current_index][self.index_steps]
                explanations = self.fact_and_exp_steps_selection(state,fact_to_exp=explained)
                batch.append([explanations[0], explanations[1]])
                self.previous_explanations = [explanations[0], explanations[1]]
                self.index_steps +=1
            else:
                self.current_index = self.problem_selection(index_problem_before)
                grid_problem_state = self.state_selection(sequential=(self.current_index == index_problem_before),
                                                          state=self.logic_puzzles_set[self.current_index].copy())
                state_plus_single_exp = grid_problem_state.copy()
                expl_name = self.frozen_steps[self.index_steps].lstrip('~')

                match = next((e for e in grid_problem_state[2] if e.name == expl_name), None)
                state_plus_single_exp[2] = {match}

                self.emergency = grid_problem_state[2]

                explanations = self.fact_and_exp_steps_selection(state_plus_single_exp)
                batch.append([explanations[0], explanations[1]])
                self.previous_explanations = [explanations[0], explanations[1]]
                self.index_steps += 1
        data.extend(batch)
        return data

    def problem_selection(self,index_problem):
        """
        Select which problem to use next.

        Args:
            index_problem: Current problem index.

        Returns:
            Updated problem index.

        This function determines when to move to the next puzzle instance based on the state
        of the current puzzle (e.g., if only one cell remains to be explained). 
        LGPs not implemented because we never reach the end of the problem.
        """
        if self.previous_explanations is None:
            return 0
        if self.problem_type=='sudoku':
            grid = self.previous_explanations[0]['grid']
            to_expl = len(grid[grid == 0])
            if to_expl==1:
                index_problem +=1
                self.index_steps = 0
        return index_problem

    def state_selection(self, sequential, state=None):
        """
        Select the state for the next query.

        Args:
            sequential: Boolean, whether to use sequential selection.
            state: Current state (puzzle instance).

        Returns:
            Selected problem state.

        This function chooses the next puzzle state to use for generating explanations,
        updating the state based on previous explanations and user choices.
        """
        if self.problem_type=='sudoku':
            if self.frozen_steps is None:
                if not sequential:
                    # No update, use the previous state (First pair)
                    problem_state = state
                else:
                    # If sequential, update the grid with the last chosen explanation according to the learnt obj functions
                    exp_0 = self.evaluate_explanation(self.previous_explanations[0])
                    exp_1 = self.evaluate_explanation(self.previous_explanations[1])
                    problem_state = self.previous_explanations[1-(exp_0<exp_1)]['grid']
                    row,col,val = self.previous_explanations[1-(exp_0<exp_1)]['hint_derived'][1]
                    problem_state[row,col] = val
        else:
            if self.frozen_steps is None:
                if not sequential:
                    problem_state = state
                else:
                    # For LGP, update facts and explainable facts based on last explanation
                    exp_0 = self.evaluate_explanation(self.previous_explanations[0])
                    exp_1 = self.evaluate_explanation(self.previous_explanations[1])
                    explained = self.previous_explanations[1-(exp_0<exp_1)]['explained']
                    if isinstance(explained,NegBoolView):
                        explained = ~explained
                        bv_explained = (next((el for el in state[2] if el.name == explained.name), None))
                        state[0].append(~bv_explained)
                        state[2].remove(bv_explained)
                    else:
                        bv_explained = (next((el for el in state[2] if el.name == explained.name), None))
                        state[0].append(bv_explained)
                        state[2].remove(bv_explained)
                    problem_state = state
            else:
                if not sequential:
                    problem_state = state
                else:
                    # For frozen steps, update state based on predefined sequence
                    explained = self.frozen_steps[self.index_steps-1]
                    has_tilde = explained.startswith("~")
                    clean_expl = explained.lstrip("~")
                    bv_explained = (next((el for el in state[2] if el.name == clean_expl), None))
                    if has_tilde:
                        state[0].append(~bv_explained)
                    else:
                        state[0].append(bv_explained)
                    state[2].remove(bv_explained)
                    problem_state = state
        return problem_state

    def fact_and_exp_steps_selection(self, state, fact_to_exp=None):
        """
        Generate explanations for a given state and a (optionally) given fact to explain (in case of pre-defined sequence).

        Args:
            state: Current puzzle state.
            fact_to_exp: Optional, specific fact to explain.

        Returns:
            Tuple of (explanation_1, explanation_2).

        This function generates two explanations for the same puzzle state, possibly using
        different diversification strategies, and tracks timing and normalization information.
        """
        # Compute exploration parameter gamma for UCB-based diversification
        if self.exploration_root is None:
            gamma=None
        else:
            gamma = (1 / self.no_iter ** (1 / self.exploration_root))


        print(f'Gamma value: {gamma}')
        if self.problem_type=='sudoku':
            start_1 = time.time()
            # Generate first explanation (optimization of the learnd objective function)
            expl_1,constraints_1,features_1 = generate_steps_sudoku(grid=state.copy(), feature_weights=self.weights_learned, counter=1,
                                                                    is_tqdm=False, normalization=self.normalization_values,
                                                                    fact_to_exp=fact_to_exp)

            end_1 = time.time()

            start_2 = time.time()
            # Generate second explanation (trade-off learned obj function and diversification)
            expl_2,constraints_2,features_2  = generate_steps_sudoku(grid=state.copy(),
                                                                     feature_weights=self.weights_learned.copy(),
                                                                     counter=1, diversify=self.type_diversification,
                                                                     obj_values_gt=features_1, gamma=gamma,
                                                                     is_tqdm=False, normalization=self.normalization_values,
                                                                     fact_to_exp=fact_to_exp,
                                                                     trade_offs = [self.trade_offs, self.trade_offs_count],
                                                                     iterations=len(self.dataset_for_lr),
                                                                     diver_value=self.diver_value)

            end_2 = time.time()

            # Track timing and normalization for analysis
            self.time_list.append([self.no_iter, end_1 - start_1, end_2 - start_2])
            self.norm_list.append([self.no_iter, self.normalization_values.copy()])
            self.weight_list.append([self.no_iter, self.weights_learned.copy()])
            df = pd.DataFrame(self.time_list, columns=['iter', 'time exp 1', 'time exp 2'])
            df_norm = pd.DataFrame(self.norm_list, columns=['iter', 'norm'])
            df.to_csv(f'{self.output_location}' + f'/time.csv')
            df_weights = pd.DataFrame(self.weight_list, columns=['iter', 'weights'])
            df_weights.to_csv(f'{self.output_location}' + f'/weights.csv')
            df_norm.to_csv(f'{self.output_location}' + f'/norm.csv')
        elif self.problem_type=='lgp':
            facts = state[0]
            constraints = state[1]
            to_explain = state[2]
            dict_constraint_type = state[3]
            dict_adjacency =  state[4].copy()
            start_1 = time.time()
            # Generate first explanation (optimization of the learnd objective function)
            expl_1, constraints_1, features_1 = generate_steps_lgps(facts=facts, constraints=constraints, to_explain=to_explain.copy(),
                                                                    dict_constraint_type=dict_constraint_type,
                                                                    dict_adjacency=dict_adjacency, sequential_lgp=-1,
                                                                    feature_weights=self.weights_learned, is_tqdm=False,
                                                                    normalization=self.normalization_values, counter=1)
            end_1 = time.time()
            print(features_1)
            print(f'Generated first expl. step in {end_1 - start_1}')
            start_2 = time.time()
            try:
                 # Generate second explanation (trade-off learned obj function and diversification)
                expl_2, constraints_2, features_2 = generate_steps_lgps(facts=facts, constraints=constraints, to_explain=to_explain.copy(),
                                                                        dict_constraint_type=dict_constraint_type,
                                                                        dict_adjacency=dict_adjacency, sequential_lgp=-1,
                                                                        feature_weights=self.weights_learned, counter=1,
                                                                        diversify=self.type_diversification,
                                                                        obj_values_gt=features_1, gamma=gamma,
                                                                        is_tqdm=False, normalization=self.normalization_values,
                                                                        trade_offs=[self.trade_offs, self.trade_offs_count],
                                                                        iterations=self.no_iter,
                                                                        diver_value=self.diver_value)
            except Exception as e:
                # If an error occurs, generate a second explanation by allowing to choose another fact.
                expl_2, constraints_2, features_2 = generate_steps_lgps(facts=facts, constraints=constraints,
                                                                        to_explain=self.emergency.copy(),
                                                                        dict_constraint_type=dict_constraint_type,
                                                                        dict_adjacency=dict_adjacency,
                                                                        sequential_lgp=-1,
                                                                        feature_weights=self.weights_learned, counter=1,
                                                                        diversify=self.type_diversification,
                                                                        obj_values_gt=features_1, gamma=gamma,
                                                                        is_tqdm=False,
                                                                        normalization=self.normalization_values,
                                                                        trade_offs=[self.trade_offs,
                                                                                    self.trade_offs_count],
                                                                        iterations=self.no_iter,
                                                                        diver_value=self.diver_value)

            end_2 = time.time()
            print(f'Generated second expl. step in {end_2 - start_2}')

            # Track timing and normalization for analysis
            self.time_list.append([self.no_iter, end_1 - start_1, end_2 - start_2])
            self.norm_list.append([self.no_iter, self.normalization_values.copy()])
            self.weight_list.append([self.no_iter, self.weights_learned.copy()])
            df = pd.DataFrame(self.time_list, columns=['iter', 'time exp 1', 'time exp 2'])
            df.to_csv(f'{self.output_location}' + f'/time.csv')
            df_norm = pd.DataFrame(self.norm_list, columns=['iter', 'norm'])
            df.to_csv(f'{self.output_location}' + f'/time.csv')
            df_weights = pd.DataFrame(self.weight_list, columns=['iter', 'weights'])
            df_weights.to_csv(f'{self.output_location}' + f'/weights.csv')
            df_norm.to_csv(f'{self.output_location}' + f'/norm.csv')

        self.all_explanations.append([expl_1,expl_2])
        # Update normalization values if required
        if self.normalized != 0 and self.normalized != 1:
            self.update_norm(features_1,features_2)
        return expl_1,expl_2

    def evaluate_explanation(self,previous_explanations):
        """
        Compute utility of an explanation using current weights.

        Args:
            previous_explanations: Dictionary of feature values for an explanation.

        Returns:
            Utility value (float).

        This function calculates the weighted sum of features for an explanation,
        normalized by the current normalization values.
        """
        utility = 0
        for feature in self.objectives:
            utility += self.weights_learned[feature] * previous_explanations[feature] / self.normalization_values[feature]
        return utility

    def update_norm(self,feat_1,feat_2):
        """
        Update normalization values based on current features.

        Args:
            feat_1: Dictionary of feature values for explanation 1.
            feat_2: Dictionary of feature values for explanation 2.

        Returns:
            None (updates normalization_values in place).

        This function updates the normalization values for each objective, either locally
        (using the max of current features) or cumulatively (using the max seen so far).
        """
        if self.normalized == 2: #local
            for key in self.normalization_values:
                v1 = feat_1[key]
                v2 = feat_2[key]
                max_val = max(v1, v2, 1)
                self.normalization_values[key] = max_val
        if self.normalized == 3: #cumulative
            for key in self.normalization_values:
                v1 = feat_1[key]
                v2 = feat_2[key]
                v3 = self.normalization_values[key]
                max_val = max(v1, v2, v3, 1)
                self.normalization_values[key] = max_val

