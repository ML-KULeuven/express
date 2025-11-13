import math

import random
from enum import Enum

import numpy as np
import pandas as pd
from IPython.core.display_functions import clear_output
from cpmpy import boolvar
from tqdm import tqdm

from utils.constants import *
from utils.utils_graphs import plot_sudoku_explanations





class Oracle:
    def __init__(self,weights,problem,lbda_indif=1):
        self.lbda_indif = lbda_indif
        if problem== 'lgp':
            objectives = OBJECTIVES_LGP
            self.dict_feature_weights = dict(zip(objectives, weights))
        if problem=='sudoku':
            objectives = OBJECTIVES_SUDOKU
            self.dict_feature_weights = dict(zip(objectives, weights))

    def get_dict_feature_weights(self):
        return self.dict_feature_weights

    def label(self,feature_pairs):
        utility_pairs = []
        for MUS in feature_pairs:
            utility = 0
            for feature in self.dict_feature_weights:
                utility += self.dict_feature_weights[feature]*MUS[feature]
            utility_pairs.append(-utility)
        picked,missed = self.bt_pba(utility_pairs[0], utility_pairs[1],
                                    lbda_indif=self.lbda_indif)
        return picked,missed,utility_pairs[0],utility_pairs[1]

    def bt_pba(self, util1, util2, lbda_indif=1):
        """
        Given the utility of two solutions,
        return the proba the solution 1 is prefered and the proba of indifference
        based on Bradley-Terry model.
        """
        clipped_val_2 = np.clip(-lbda_indif * abs(util1 - util2), -700, 700)
        pba_indif = math.exp(clipped_val_2)
        print(f'Probabilities of missing {pba_indif} given {util1} and {util2}')
        if np.random.choice([0, 1], p=[pba_indif, 1 - pba_indif]) == 0:
            return -1, None
        else:
            if util1>util2:
                pba_pref = 0.90
            else:
                pba_pref = 0.10
            if np.random.choice([0, 1], p=[pba_pref, 1-pba_pref]) == 0:
                return 0, (pba_pref < 1 - pba_pref)
            else:
                return 1, (pba_pref > 1 - pba_pref)

class Human:
    def label(self,MUSes,evaluation=False):
        if not evaluation:
            original_indices = [0, 1]
            shuffled_indices = original_indices.copy()
            random.shuffle(shuffled_indices)

            # Shuffle MUSes according to shuffled_indices
            shuffled_MUSes = [MUSes[i] for i in shuffled_indices]
            plot_sudoku_explanations(shuffled_MUSes)

            selection = input("Enter 1 for left Image , 2 for right Image 1, 0 for neither of the two")
            selection = int(selection) - 1  # Convert to 0-based index
            if selection in [0, 1]:
                selected_original_index = shuffled_indices[selection]
            else:
                selected_original_index = - 1
            clear_output(wait=True)
            return selected_original_index, None, None, None
        else:
            results = {'SES':0,'Learnt':0,'Draw':0}
            MUSes[0]['generation'] = 'SES'
            MUSes[1]['generation'] = 'Learnt'
            for idx in tqdm(range(len(MUSes[0]))):
                row_hand = MUSes[0].iloc[idx]
                row_learnt = MUSes[1].iloc[idx]
                proposed = [row_hand,row_learnt]
                random.shuffle(proposed)
                plot_sudoku_explanations(proposed)
                selection = input("Enter 1 for left Image , 2 for right Image 1, 0 for neither of the two")
                clear_output(wait=True)
                selection = int(selection)
                selection = selection - 1
                if selection<0:
                    results['Draw'] += 1
                else:
                    results[proposed[selection]['generation']]+=1
            return results



class Relation(object):
    # rows, cols: list of names
    def __init__(self, rows, cols, name=""):
        self.cols = cols
        self.rows = rows
        self.name = name
        rel = boolvar(shape=(len(rows), len(cols)),name=name)
        self.df = pd.DataFrame(index=rows, columns=cols)
        for i,r in enumerate(rows):
            for j,c in enumerate(cols):
                self.df.loc[r,c] = rel[i,j]
    # use as: rel['a','b']
    def __getitem__(self, key):
        try:
            return self.df.loc[key]
        except KeyError:
            print(f"Warning: {self.name}{key} not in this relation")
            return False

class HintType(str, Enum):
    ROW     = "ROW"
    ATMOST_ROW = "ATMOST_ROW"
    ATLEAST_ROW = "ATLEAST_ROW"
    COL     = "COL"
    ATMOST_COL = "ATMOST_COL"
    ATLEAST_COL = "ATLEAST_COL"
    BLOCK   = "BLOCK"
    ATMOST_BLOCK = "ATMOST_BLOCK"
    ATLEAST_BLOCK = "ATLEAST_BLOCK"
    FACT    = "FACT"
    DOM_REDUCTION = "DOM_REDUCTION"
    DOM_REDUC_DERIVED = "DOM_REDUC_DERIVED"
    DERIVED = "DERIVED"