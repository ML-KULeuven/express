# Import required libraries for core functionality
import ast
import copy
import json
import math
import os
import subprocess
from collections import defaultdict

# Import libraries for constraint programming and optimization
import cpmpy as cp
from time import time

# Import data manipulation and ML libraries
import numpy as np
import pandas as pd
from cpmpy.solvers.solver_interface import ExitStatus
from sklearn.model_selection import KFold
from tqdm import tqdm

# Import predefined normalization values for different puzzle types
from utils.constants import OBJECTIVES_NORMALIZED_LGP, OBJECTIVE_NORMALIZED_SUDOKU

# Global verbosity setting
verbose = 0

def create_folders(directory):
    """
    Create nested directories, making parent directories as needed.
    
    Args:
        directory: String path of directories to create
    """
    folders = directory.split(os.path.sep)
    current_path = ''
    if not folders[0]:
        current_path = os.path.sep  # Set the current path to root
        folders = folders[1:]
    for folder in folders:
        current_path = os.path.join(current_path, folder)
        if not os.path.exists(current_path):
            os.mkdir(current_path)

def create_dataframe_from_dicts(dicts):
    """
    Convert a dictionary of dictionaries to a pandas DataFrame.
    Handles empty dictionaries by filling with None values.
    
    Args:
        dicts: Dictionary containing nested dictionaries of features
        
    Returns:
        DataFrame with features as columns and original dict keys as a 'name' column
    """
    return pd.DataFrame([{
        **(d if d else {k: None for k in dicts[next(iter(dicts))]}),
        'name': name
    } for name, d in dicts.items()])

def leftover(time_limit, start):
    """
    Calculate remaining time before timeout, ensuring at least 1 second.
    
    Args:
        time_limit: Maximum allowed time in seconds
        start: Start time in seconds
        
    Returns:
        int: Number of seconds remaining, minimum 1
    """
    return max(1, time_limit - (time() - start))

def load_sudokus_from_json(filename="./data/sudoku_instances.json"):
    # Load all Sudoku puzzles from a JSON file as an array
    with open(filename, 'r') as file:
        sudokus_list = json.load(file)
    sudokus = [np.array(lst) for lst in sudokus_list]
    return sudokus

from cpmpy.expressions.core import Comparison, Operator
from cpmpy.expressions.variables import _BoolVarImpl, NegBoolView
from cpmpy.expressions.utils import is_num
from cpmpy.transformations.get_variables import get_variables

def _is_fact(expr):
    if isinstance(expr, _BoolVarImpl): return True # Boolean encoding
    return isinstance(expr, Comparison) and \
            expr.name == "==" and \
            isinstance(expr.args[0], _BoolVarImpl) and \
            is_num(expr.args[1]) and expr.args[1] == 1

def all_pos_weights(weights):
    if any(w <= 0 for w in weights):
        return False
    return True

def fact_to_cons_hints(dmap):
    """
        Help the hitting set solver by enforcing to take "useful" hitting sets.
        That is, when a given cell is selcted to be part of the explanation, there must also be at least one
            constraint be selected that overlaps with that cell.

        ONLY valid if all coefficients in the objective are positive
    """

    # bit of a hack but ok
    facts_map, cons_map = dict(), dict()
    for k,v in dmap.items():
        if _is_fact(v): facts_map[k] = v
        else:           cons_map[k] = v

    constraints = []
    for k,v in facts_map.items():
        fact_cons = []
        fact_vars = frozenset(get_variables(v))
        for k2, v2 in cons_map.items():
            if set(get_variables(v2)) & fact_vars: # cache this, or invert loops?
                fact_cons.append(k2)
        constraints.append(k.implies(cp.any(fact_cons)))

    return constraints

def all_overlapping_cons(dmap):

    var_cache = {k : frozenset(get_variables(v)) for k,v in dmap.items()}

    constraints = []
    for k1, v1 in dmap.items():
        k_cons = []
        for k2,v2 in dmap.items():
            if str(k1) == str(k2): continue
            if var_cache[k1] & var_cache[k2]:
                k_cons.append(k2)

        constraints.append(k1.implies(cp.any(k_cons)))

    return constraints


def split_ocus_assum(soft,             # List of soft constraints/assumptions 
                     dmap,             # Dictionary mapping assumptions to constraints
                     objectives=None,   # List of objective functions to optimize  
                     hard=[],          # List of hard constraints that must be satisfied
                     hard_hs = {},     # Dictionary mapping facts to hitting set constraints
                     oneof_idxes=[],   # Indices in soft containing facts to explain
                     solver="exact",    # Name of solver to use ("exact", etc)
                     hs_solver_name="gurobi",  # Hitting set solver name
                     time_limit=36000,  # Maximum time allowed in seconds
                     solver_params=dict()): # Additional solver parameters
    """
    Generate minimal satisfying subsets using OCUS algorithm with assumptions.
    For each fact to explain (specified by oneof_idxes), finds minimal sets of
    constraints that prove that fact while optimizing the given objectives.

    Args:
        soft: List of assumption variables for soft constraints 
        dmap: Maps assumption variables to their corresponding constraints
        objectives: List of objective functions to optimize (one per fact)
        hard: List of hard constraints that must be satisfied
        hard_hs: Maps facts to hitting set solver constraints
        oneof_idxes: Indices in soft list containing facts to explain
        solver: Name of main constraint solver to use
        hs_solver_name: Name of hitting set solver to use
        time_limit: Maximum solving time in seconds
        solver_params: Additional parameters for solvers

    Yields:
        Lists of constraints forming minimal explanations, or None if timeout
    """
    m = cp.Model(hard)
    start_time = time()
    s = cp.SolverLookup.get(solver, m)
    assert not s.solve(assumptions=soft), "MUS: model must be UNSAT"

    queue = []
    base_soft = set(a for i, a in enumerate(soft) if i not in oneof_idxes)

    # initialize hs solvers for each fact to explain
    for oneof_idx, objective in zip(oneof_idxes, objectives):
        to_expl = soft[oneof_idx]
        hs_solver = cp.SolverLookup.get(hs_solver_name)
        hs_solver += (to_expl == 1)
        if to_expl in hard_hs:
            hs_solver += hard_hs[to_expl]
        hs_solver.minimize(objective + to_expl)

        if hs_solver.solve(threads=8,time_limit=leftover(start_time, time_limit)):
            soft_subset = base_soft | set({to_expl})
            subset = [si for si in soft_subset if si.value()]
            queue.append((to_expl, hs_solver, hs_solver.objective_value(), subset))

    while leftover(start_time, time_limit) > 0:
        queue.sort(key=lambda tup: tup[2])  # find hs solver with lowest obj value
        to_expl, hs_solver, obj_val, subset = queue.pop(0)

        submap = {a: dmap[a] for i, a in enumerate(soft) if i not in oneof_idxes} | {to_expl: dmap[to_expl]}

        if not s.solve(assumptions=subset, **solver_params, time_limit=leftover(start_time, time_limit)):
            if leftover(start_time, time_limit) == 0:
                yield None
            else:
                yield [submap[a] for a in subset]
                hs_solver += ~cp.all(subset)
        else:
            for grown in corr_subsets(subset, submap, s, hard=hard, time_limit=leftover(start_time, time_limit)):
                hs_solver += cp.sum(grown) >= 1

        if hs_solver.solve(threads=8,time_limit=leftover(start_time, time_limit)):
            soft_subset = base_soft | set({to_expl})
            subset = [si for si in soft_subset if si.value()]
            queue.append((to_expl, hs_solver, hs_solver.objective_value(), subset))

    yield None


def corr_subsets(subset, dmap, solver, hard, time_limit=3600, solver_params=dict()):
    """If no change is the grown subset stop"""
    start_time = time()
    sat_subset = {s for s in subset}
    corr_subsets = []
    vars=  list(dmap.keys())
    solver.solution_hint(vars, [1]*len(vars))

    while solver.solve(assumptions=list(sat_subset), **solver_params):
        """
            Change the grow method here if wanted!
            MaxSAT grow will probably be slow but result in very small sets to hit (GOOD!)
            SAT-grow will probably be a little faster but greedily finds a small set to hit
            Greedy-grow will not do ANY solving and simply exploit the values in the current solution
        """
        corr_subset = greedy_grow(dmap)
        if len(corr_subset) == 0:
            return corr_subsets

        sat_subset |= corr_subset
        corr_subsets.append(corr_subset)
        solver.solution_hint(vars, [1] * len(vars))

    if solver.status().exitstatus == ExitStatus.UNKNOWN:
        raise TimeoutError("Correction subsets timed out during solve")
    return corr_subsets


def greedy_grow(dmap):
    """
        Very cheaply check the values of the decision variables and construct sat set from that
    """
    sat_subset = {assump for assump, cons in dmap.items() if assump.value() or cons.value()}
    return set(dmap.keys()) - sat_subset





def extract_dv_from_expression(constr,involved=[]):
    if isinstance(constr, Operator):
        for expr in constr.args:
            involved = extract_dv_from_expression(expr,involved)
    elif isinstance(constr, Comparison):
        for expr in constr.args:
            involved = extract_dv_from_expression(expr, involved)
    elif isinstance(constr, NegBoolView):
        involved.append(constr._bv)
        return involved
    elif isinstance(constr,_BoolVarImpl):
        involved.append(constr)
        return list(set(involved))
    return list(set(involved))



def tune_lr(dataset,objectives,initial_weights,learned_weights=None,current_lr=None):
    possible_lr = [0.1, 0.2, 0.5, 1, 2, 5, 10]
    if learned_weights is None and len(dataset)>=2:
        dict_ranking_loss = {lr: [] for lr in possible_lr}
        n_folds = min(5, len(dataset))
        kfold = KFold(n_splits=n_folds, shuffle=True)
        for train_index, test_index in kfold.split(dataset):
            for lr in possible_lr:
                train_data = [dataset[i] for i in train_index]
                test_data = [dataset[i] for i in test_index]
                w = train(train_data,objectives,initial_weights.copy(),lr)
                ranking_loss = evaluate(test_data,w,objectives)
                dict_ranking_loss[lr].append(ranking_loss)
        for lr in dict_ranking_loss.keys():
            dict_ranking_loss[lr] = sum(dict_ranking_loss[lr]) / len(dict_ranking_loss[lr])
        # Find all keys that have this minimum average ranking loss
        best_lr = max((k for k, v in dict_ranking_loss.items() if v == min(dict_ranking_loss.values())))
        return best_lr
    if learned_weights is not None and len(dataset)>=2:
        dict_ranking_loss = {lr: 0 for lr in possible_lr}
        evaluate_on = dataset[:-1]
        next_update = [dataset[-1]]
        for lr in possible_lr:
            w = train(next_update, objectives, learned_weights.copy(), lr)
            ranking_loss = evaluate(evaluate_on, w, objectives)
            dict_ranking_loss[lr] = ranking_loss
        best_lr = max([k for k, v in dict_ranking_loss.items() if v == min(dict_ranking_loss.values())])
        return best_lr
    else:
        return current_lr



def train(dataset,objectives,starting_weights,lr):
    difference = defaultdict(int)
    if len(dataset)>0:
        for key in objectives:
            for pair in dataset:
                difference[key] = (pair[1][key] - pair[0][key])
            difference[key] = difference[key]
            starting_weights[key] = max(1e-4,starting_weights[key] + lr*difference[key])
    return starting_weights

def evaluate(dataset,weights,objectives):
    ranking_loss = 0
    for pair in dataset:
        obj_value_0 = sum(pair[0][key] * weights[key] for key in objectives)
        obj_value_1 = sum(pair[1][key] * weights[key] for key in objectives)
        ranking_loss += (obj_value_0 >= obj_value_1)
    ranking_loss = ranking_loss/len(dataset)
    return ranking_loss


def rank_dictionary(d):
    previous_denominator = 1
    sorted_items = sorted(d.items(), key=lambda x: x[1])
    ranks, rank = {}, 1
    for i, (k, v) in enumerate(sorted_items):
        rank += i > 0 and v != sorted_items[i - 1][1]
        ranks[k] = rank

    grouped_by_rank = {}
    for k, rank in ranks.items():
        grouped_by_rank.setdefault(rank, []).append(k)

    for rank in sorted(grouped_by_rank,reverse=True):
        if rank == 1:
            next_den = 1
        else:
            next_den = 2*len(grouped_by_rank[rank])*previous_denominator
        for el in grouped_by_rank[rank]:
            ranks[el] = 1/next_den
        previous_denominator = next_den
    return ranks

def dict_normalize(d):
    max_val = max(d.values())
    return {k: (v) / (max_val) for k, v in d.items()}


def ucb_dictionary(d,learned_weights,iterations,ucb_val=2):
    ranks = {}
    # norm_preferred_count = dict_normalize(d[1])
    for obj in d[0]:
        if d[0][obj]==0:
            ranks[obj] = 1e3
        else:
            ranks[obj] = d[1][obj]/d[0][obj] + (ucb_val*math.log(iterations+1)/d[0][obj])**(1/2)
    # max_rank = max(ranks.values())
    # ranks = {k: v / max_rank for k, v in ranks.items()}
    return ranks



def compute_normalization(data,normalization,problem_type='sudoku'):
    objective_normalized = {}
    if normalization==0:
        if problem_type == 'sudoku':
            objective_normalized = {obj: 1 for obj in OBJECTIVE_NORMALIZED_SUDOKU}
        elif problem_type == 'lgp':
            objective_normalized = {obj: 1 for obj in OBJECTIVES_NORMALIZED_LGP}
    elif normalization==1:
        if problem_type=='sudoku':
            objective_normalized = {obj: OBJECTIVE_NORMALIZED_SUDOKU[obj] for obj in OBJECTIVE_NORMALIZED_SUDOKU}
        elif problem_type=='lgp':
            objective_normalized = {obj: OBJECTIVES_NORMALIZED_LGP[obj] for obj in OBJECTIVES_NORMALIZED_LGP}
    else:
        for subdict in data.values():
            for obj, values in subdict.items():
                objective_normalized[obj] = max(objective_normalized.get(obj, 0), len(values)) or 1
    return objective_normalized


def random_zero_cell(grid_problem_state):
    zero_positions = np.argwhere(grid_problem_state == 0)
    if zero_positions.size == 0:
        raise ValueError("No zero cells in the grid.")
    row, col = zero_positions[np.random.choice(len(zero_positions))]
    return row, col



def count_and_remove_matching_hints(df1, df2):
    indices_to_drop = []

    for idx, (row1, row2) in enumerate(zip(df1.itertuples(), df2.itertuples())):
        # Parse values (assumes strings; skip ast.literal_eval if already lists)
        facts1 = ast.literal_eval(row1.hint_facts)
        facts2 = ast.literal_eval(row2.hint_facts)

        constraints1 = ast.literal_eval(row1.hint_constraints)
        constraints2 = ast.literal_eval(row2.hint_constraints)

        facts_match = all(f in facts2 for f in facts1) and all(f in facts1 for f in facts2)
        constraints_match = all(c in constraints2 for c in constraints1) and all(c in constraints1 for c in constraints2)

        if facts_match and constraints_match:
            indices_to_drop.append(idx)

    # Drop matching rows from both DataFrames
    df1_cleaned = df1.drop(index=indices_to_drop).reset_index(drop=True)
    df2_cleaned = df2.drop(index=indices_to_drop).reset_index(drop=True)

    return len(indices_to_drop), df1_cleaned, df2_cleaned



def compute_normalization(data):
    objective_normalized = {}
    for subdict in data.values():
        for obj, values in subdict.items():
            objective_normalized[obj] = max(objective_normalized.get(obj, 0), len(values)) or 1
    return objective_normalized


def average_first_dicts(list_of_lists,objs):
    sum_dict = defaultdict(float)
    count = 0

    for pair in list_of_lists:
        first_dict = pair[0]
        for obj in objs:
            adjusted_value = 1 if first_dict[obj] == 0 else first_dict[obj]
            sum_dict[obj] += adjusted_value
        count += 1

    # Compute average
    avg_dict = {key: total / count for key, total in sum_dict.items()}
    return avg_dict

def cliffs_delta(a, b):
    a = np.array(a)
    b = np.array(b)
    n = len(a) * len(b)
    greater = sum(ai > bj for ai in a for bj in b)
    less = sum(ai < bj for ai in a for bj in b)
    return (greater - less) / n