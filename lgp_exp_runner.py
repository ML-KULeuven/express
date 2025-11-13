import argparse
import ast
import json

import numpy as np
import pandas as pd


from model.lg_problems import LGProblem
from utils.constants import *
from utils.utils import load_sudokus_from_json
from utils.utils_classes import Oracle
from utils.utils_training import PreferenceElicitationFramework
from datetime import datetime

# -------------------------------
# Argument Parsing
# -------------------------------
# This script accepts the following command-line arguments:
# -u / --user:           (int) User index to select which user's weights to use. Default is 0.
# -n / --normalization:  (int) Type of Normalization. 0: none, 1: nadir, 2: local, 3: global. Default is 1.
# -s / --steps:          (str) Step selection strategy. Choices: 'SMUS', 'random', 'standard'. Default is 'standard'.
# -t / --type:           (str) Diversification strategy for learning. Choices: 'baseline', 'disjunction', 'coverage', 'w_disjunction', 'cpucb_disjunction'. Default is 'baseline'.
# -f / --seed:           (int) Random seed used to freeze the experiments. Default is 42.

parser = argparse.ArgumentParser(description='Runner LGPs')
parser.add_argument('-u', '--user', type=int, help='user', required=False, default=0)
parser.add_argument('-n', '--normalization', type=int, help='normalization', required=False, default=1)
parser.add_argument('-s', '--steps', type=str, choices=['SMUS','random','standard'],
                    help = 'steps', required=False, default='standard')
parser.add_argument('-t', '--type', type=str, choices=['baseline','disjunction','coverage',
                                                                    'w_disjunction','MACHOP'],
                    help = 'dispersion type', required=False, default='baseline')
parser.add_argument('-f', '--seed', type=int, help='random seed (frozen for reproducibility)',
                    required=False, default=42)

# Parse command-line arguments
args = parser.parse_args()
user = int(args.user)  # User index for selecting weights
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Timestamp for output folder
exploration_value = 1  # Exploration parameter for learning
normalization = int(args.normalization)  # Normalization type
steps = args.steps  # Step selection strategy
type_diversification = args.type  # Diversification strategy

random.seed(args.seed)
np.random.seed(args.seed)

# -------------------------------
# Problem Instance Preparation
# -------------------------------

range_considered_problems = 1  # Number of training problems to consider
instance_training = []         # List to hold training instances
instance_evaluation = []       # List to hold evaluation instances

# Prepare training problem for each problem in range_considered_problems
for range_problem in range(range_considered_problems):
    maker = LGProblem(type=range_problem)  # Create LGProblem instance
    # Unpack model components (facts, constraints, etc.)
    [], constraints, facts, explainable_facts, dict_constraint_type, dict_constraints_involved, rels_visualization, dict_constraints_clues = maker.make_model()
    # Store relevant components for training
    instance_training.append([facts,constraints,explainable_facts,dict_constraint_type,dict_constraints_involved])

# Prepare evaluation instance
maker = LGProblem(type=range_considered_problems)
[], constraints, facts, explainable_facts, dict_constraint_type, dict_constraints_involved, rels_visualization, dict_constraints_clues = maker.make_model()
instance_evaluation.append([facts,constraints,explainable_facts,dict_constraint_type,dict_constraints_involved])
df_steps_evaluation = pd.read_csv(f'data/gt_lgp/normalization_{0}/lgp_user_{user}_problem_{range_problem+1}.csv',index_col=False)

# -------------------------------
# Load User Weights
# -------------------------------

# Load precomputed user weights from JSON file
with open("data/weights/LGPs/weights.json", "r") as json_file:
    weights_user = json.load(json_file)

# -------------------------------
# Step Selection and Learning Rate Setup
# -------------------------------

# Select step strategy and learning rate based on arguments
if steps == 'SMUS':
    # Load steps from SMUS strategy CSV
    steps_lgps = pd.read_csv('data/gt_lgp/normalization_0/lgp_user_SMUS_problem_0.csv')['explained']
    type_steps = None
    lr = LR_NORM_LGP_SMUS
elif steps == 'random':
    # Use random step selection
    type_steps = 'random'
    steps_lgps = None
    lr = LR_NORM_LGP_RANDOM
else:
    # Use dynamic step selection
    type_steps = None
    steps_lgps = None
    lr = LR_NORM_LGP[normalization][type_diversification]


# Construct output directory path for results
output_location = f'results_AAAI/lgps_norm_{normalization}/tuner_{None}_exploration_{exploration_value}/lr_{lr}_diversification_{type_diversification}/lgps_steps_{steps}_user_{user}_id_{start_time}/'
#oracle definition
oracle = Oracle(weights=weights_user[user],problem='lgp')

# Set initial weights for objectives (all set to 1)
initial_weights = dict.fromkeys(OBJECTIVES_LGP, 1)
df_steps_evaluation = pd.read_csv(f'data/gt_lgp/normalization_{0}/lgp_user_{user}_problem_{range_problem+1}.csv',index_col=False)

pef = PreferenceElicitationFramework(logic_puzzles_set=instance_training, oracle=oracle, no_oracle=user,
                                     initial_weights=initial_weights, instance_evaluation=instance_evaluation,
                                     df_steps_evaluation=df_steps_evaluation, output_location=output_location,
                                     time_eval=100,
                                     normalized=normalization, batch_size=1, frozen_steps=steps_lgps,
                                     lr=lr, type_diversification=type_diversification,
                                     problem_type='lgp',exploration_root=exploration_value,
                                     type_steps=type_steps)
# Start the learning process and obtain learned weights
learned_weights  = pef.start()
