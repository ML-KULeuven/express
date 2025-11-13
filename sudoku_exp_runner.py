import argparse
import ast
import json

import numpy as np
import pandas as pd
from datetime import datetime
from utils.constants import *
from utils.utils import load_sudokus_from_json, create_folders
from utils.utils_classes import Oracle
from utils.utils_training import PreferenceElicitationFramework

# -------------------------------
# Argument Parsing
# -------------------------------
# This script accepts the following command-line arguments:
# -u / --user:           (int) User index to select which user's weights to use. Default is 0.
# -n / --normalization:  (int) Type of Normalization. 0: none, 1: nadir, 2: local, 3: global. Default is 1.
# -s / --steps:          (str) Step selection strategy. Choices: 'SMUS', 'random', 'standard'. Default is 'standard'.
# -t / --type:           (str) Diversification strategy for learning. Choices: 'baseline', 'disjunction', 'w_disjunction', 'cpucb_disjunction'. Default is 'baseline'.
# -f / --seed:           (int) Random seed used to freeze the experiments. Default is 42.

parser = argparse.ArgumentParser(description='Runner sudokus')
parser.add_argument('-u', '--user', type=int, help='user', required=False, default=0)
parser.add_argument('-n', '--normalization', type=int, help='normalization', required=False, default=1)
parser.add_argument('-s', '--steps', type=str, choices=['SMUS','random','standard'],
                    help = 'steps', required=False, default='standard')
parser.add_argument('-t', '--type', type=str, choices=['baseline','disjunction','coverage',
                                                                    'w_disjunction','MACHOP'],
                    help = 'dispersion type', required=False, default='baseline')
parser.add_argument('-f', '--seed', type=int, help='random seed',
                    required=False, default=42)

# Parse command-line arguments
args = parser.parse_args()
user = int(args.user)  # User index for selecting weights
normalization = int(args.normalization)  # Normalization type
steps = args.steps  # Step selection strategy
type_diversification = args.type  # Diversification strategy
exploration_value = 1  # Exploration parameter for learning
start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")  # Timestamp for output folder

random.seed(args.seed)
np.random.seed(args.seed)

# Load sudoku puzzles from JSON file
sudokus = load_sudokus_from_json()

# Load precomputed user weights from JSON file
with open("data/weights/sudoku/weights.json", "r") as json_file:
    weights_user = json.load(json_file)
steps_sudokus = None

# -------------------------------
# Step Selection and Learning Rate Setup
# -------------------------------

if steps == 'SMUS':
    # Use SMUS step selection strategy
    type_steps = None
    steps_sudokus = []
    for index in range(15):
        steps_single_sudoku = []
        # Load step data for each sudoku
        df_steps = pd.read_csv(f'data/gt_sudoku/SMUS/sudoku_user_{steps}_sudoku_{index}.csv',index_col=False)
        for index, row in df_steps.iterrows():
            # Parse grid and hint coordinates from CSV
            grid = np.fromstring(row['grid'].replace("[", "").replace("]", ""), sep=" ").astype(int).reshape(9, 9)
            r,c = ast.literal_eval(row['hint_derived'])[1][:2]
            steps_single_sudoku.append([grid,[r,c]])
        steps_sudokus.append(steps_single_sudoku)
    lr = LR_NORM_SUDOKU_SMUS
elif steps == 'random':
    # Use random step selection strategy
    type_steps = 'random'
    lr = LR_NORM_SUDOKU_RANDOM
else:
    # Use standard step selection strategy
    type_steps = None
    lr = LR_NORM_SUDOKU[normalization][type_diversification]

# -------------------------------
# Output Location Setup
# -------------------------------

# Construct output directory path for results
output_location = f'results_AAAI/sudoku_{normalization}/tuner_{None}_exploration_{exploration_value}/lr_{lr}_diversification_{type_diversification}/sudoku_steps_{steps}_user_{user}_id_{start_time}/'

#oracle definition
oracle = Oracle(weights=weights_user[user],problem='sudoku')

# Set initial weights for sudoku objectives (all set to 1)
initial_weights = dict.fromkeys(OBJECTIVES_SUDOKU, 1)

# Load evaluation steps from CSV for the selected user and sudoku
df_steps_evaluation = pd.read_csv(f'data/gt_sudoku/normalization_{0}/sudoku_user_{user}_sudoku_30.csv',index_col=False)

pef = PreferenceElicitationFramework(logic_puzzles_set=sudokus, oracle=oracle, no_oracle=user,
                                     initial_weights=initial_weights, instance_evaluation=sudokus[30],
                                     df_steps_evaluation=df_steps_evaluation, output_location=output_location,
                                     time_eval=100,
                                     normalized=normalization, batch_size=1, frozen_steps=steps_sudokus,
                                     lr=lr, type_diversification=type_diversification,
                                     exploration_root=exploration_value,type_steps=type_steps)
# Start the learning process and obtain learned weights
learned_weights  = pef.start()
