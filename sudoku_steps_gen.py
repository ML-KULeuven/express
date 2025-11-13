import argparse
import json

from utils.constants import OBJECTIVES_SUDOKU
from utils.utils_sudoku import generate_steps_sudoku, load_sudokus_from_json, generate_top_k_expl

# -------------------------------
# Argument Parsing
# -------------------------------
# This script accepts the following command-line argument:
# -u / --user: (str) User identifier. Can be an integer index, 'smus', or 'real'. Default is 0.

parser = argparse.ArgumentParser(description='Runner sudokus')
parser.add_argument('-u', '--user', type=str, help='user', required=False, default=0)
args = parser.parse_args()
no_users = args.user  # User identifier

# -------------------------------
# Load User Weights and Sudokus
# -------------------------------
with open("data/weights/sudoku/weights.json", "r") as json_file:
    weights = json.load(json_file)
sudokus = load_sudokus_from_json()

# -------------------------------
# Feature Weights Setup
# -------------------------------
# Select feature weights based on user argument
# - 'smus': use None (generation SES steps)
# - 'real': use hardcoded weights
# - otherwise: use weights from file for specified user index
if no_users=='smus':
    weights_user = None
elif no_users=='real':
    weights_user = {
        'number_facts': 1,
        'number_constraints': 20,
    }
else:
    no_users = int(no_users)
    weights_user = dict(zip(OBJECTIVES_SUDOKU, weights[no_users]))

# -------------------------------
# Normalization Setup
# -------------------------------
# Set normalization for each feature to 1
normalization= {el:1 for el in weights_user.keys()}

# -------------------------------
# Step Generation
# -------------------------------
# Generate steps for sudoku puzzle 31 using the selected weights and normalization
generate_steps_sudoku(grid=sudokus[31].copy(), feature_weights=weights_user, is_tqdm=True, normalization=normalization,
                      to_return=False,sequential_sudoku=31,no_user=no_users)
