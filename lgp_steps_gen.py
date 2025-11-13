import argparse
import json

from model.lg_problems import LGProblem
from utils.constants import OBJECTIVES_LGP
from utils.utils_lgp import generate_steps_lgps, new_generate_steps_lgps

# -------------------------------
# Argument Parsing
# -------------------------------
# This script accepts the following command-line arguments:
# -u / --user: (int) User index to select which user's weights to use. Default is 0.
#                Use -1 for SMUS strategy.
# -t / --type: (int) Type of logic grid problem to generate steps for (required).

parser = argparse.ArgumentParser(description='Runner lgps')
parser.add_argument('-u', '--user', type=int, help='user', required=False, default=0)
parser.add_argument('-t', '--type', type=int, help='type problem', required=True, default=0)
args = parser.parse_args()
no_users = args.user  # User index or SMUS flag
type_problem =  args.type  # Problem type

# -------------------------------
# Load User Weights
# -------------------------------
with open("data/weights/LGPs/weights.json", "r") as json_file:
    weights = json.load(json_file)

# -------------------------------
# Feature Weights Setup
# -------------------------------
if no_users==-1:
    # If user is -1, use SMUS (SES) strategy with generic weights
    no_users = 'SMUS'
    feature_weights = {'all':1}
else:
    # Otherwise, use weights for the specified user
    feature_weights = dict(zip(OBJECTIVES_LGP, weights[no_users]))
print(feature_weights)

# -------------------------------
# Normalization Setup
# -------------------------------
# Set normalization for each feature to 1
normalization= {el:1 for el in feature_weights.keys()}

# -------------------------------
# Problem Instance Preparation
# -------------------------------
# Create LGProblem instance and unpack model components
maker = LGProblem(type=type_problem)
[],constraints,facts,explainable_facts,dict_constraint_type,dict_constraints_involved,rels_visualization,dict_constraints_clues = maker.make_model()

# -------------------------------
# Step Generation
# -------------------------------
# Generate steps for the logic grid problem using the selected weights and normalization
generate_steps_lgps(facts,constraints,explainable_facts,feature_weights=feature_weights,
                    dict_constraint_type=dict_constraint_type,dict_adjacency=dict_constraints_involved,
                    normalization=normalization,sequential_lgp=type_problem,no_user=no_users)



