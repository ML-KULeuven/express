import ast
import json
import os
import pandas as pd
import cpmpy as cp
from cpmpy import intvar

from utils.constants import OBJECTIVE_NORMALIZED_SUDOKU, LR_NORM_LGP, LR_NORM_SUDOKU
from utils.utility_visualization import (compute_regret_hyperparam_last, extract_row_col_values, plot_RQ1,
                                         plot_lgp_sudoku_comparison, process_json_files,
                                         process_time_data, generate_RQ5_table,
                                         summarize_regrets, latex_regret_table_star, summarize_metrics,
                                         latex_compare_machop_baseline_table)
from utils.utils import cliffs_delta
from utils.utils_graphs import generate_time_table


# ## ------------------------RQ1-RQ2
ris_0 = compute_regret_hyperparam_last('results_oracle_AAAI/sudoku_0/tuner_None_exploration_1')
result_0 = extract_row_col_values(ris_0[1], ['baseline','disjunction'], LR_NORM_SUDOKU[0])
ris_1 = compute_regret_hyperparam_last('results_oracle_AAAI/sudoku_1/tuner_None_exploration_1')
result_1 = extract_row_col_values(ris_1[1], ['baseline','disjunction'], LR_NORM_SUDOKU[1])
ris_2 = compute_regret_hyperparam_last('results_oracle_AAAI/sudoku_3/tuner_None_exploration_1')
result_2 = extract_row_col_values(ris_2[1], ['baseline','disjunction'], LR_NORM_SUDOKU[3])
ris_3 = compute_regret_hyperparam_last('results_oracle_AAAI/sudoku_2/tuner_None_exploration_1')
result_3 = extract_row_col_values(ris_3[1], ['baseline','disjunction'], LR_NORM_SUDOKU[2])
for result in [result_0,result_1,result_2,result_3]:
    result['non-domination'] = result.pop('disjunction')

plot_RQ1([result_0,result_1,result_2,result_3], title='Relative Regret - Sudoku',y_label='Average Relative Regret',
         y_max=10,problem='Sudoku')



ris_0 = compute_regret_hyperparam_last('results_oracle_AAAI/lgps_norm_0/tuner_None_exploration_1')
result_0 = extract_row_col_values(ris_0[1], ['baseline','disjunction'], LR_NORM_LGP[0])
ris_1 = compute_regret_hyperparam_last('results_oracle_AAAI/lgps_norm_1/tuner_None_exploration_1')
result_1 = extract_row_col_values(ris_1[1], ['baseline','disjunction'], LR_NORM_LGP[1])
ris_2 = compute_regret_hyperparam_last('results_oracle_AAAI/lgps_norm_3/tuner_None_exploration_1')
result_2 = extract_row_col_values(ris_2[1], ['baseline','disjunction'], LR_NORM_LGP[3])
ris_3 = compute_regret_hyperparam_last('results_oracle_AAAI/lgps_norm_2/tuner_None_exploration_1')
result_3 = extract_row_col_values(ris_3[1], ['baseline','disjunction'], LR_NORM_LGP[2])
for result in [result_0,result_1,result_2,result_3]:
    result['non-domination'] = result.pop('disjunction')

plot_RQ1([result_0,result_1,result_2,result_3], title='Relative Regret - LGPs',y_label='Average Relative Regret',
         y_max=40,problem='LGPs')


##------------------------RQ3
ris_2 = compute_regret_hyperparam_last('results_oracle_AAAI/lgps_norm_2/tuner_None_exploration_1')
result_lgp = extract_row_col_values(ris_2[1], ['baseline','disjunction','w_disjunction','MACHOP'],LR_NORM_LGP[2])
for result in [result_lgp]:
    result.pop('baseline')
    result['L1'] = result.pop('disjunction')
    result['w_L1'] = result.pop('w_disjunction')
    result['cpucb_L1'] = result.pop('MACHOP')


ris_2 = compute_regret_hyperparam_last('results_oracle_AAAI/sudoku_2/tuner_None_exploration_1')
result_sudoku = extract_row_col_values(ris_2[1], ['baseline','disjunction','w_disjunction','MACHOP'],LR_NORM_SUDOKU[2])
for result in [result_sudoku]:
    result.pop('baseline')
    result['L1'] = result.pop('disjunction')
    result['w_L1'] = result.pop('w_disjunction')
    result['cpucb_L1'] = result.pop('MACHOP')


plot_lgp_sudoku_comparison(result_sudoku,result_lgp)


##------------------------RQ4-Reported in the table
ris = generate_time_table('./results_oracle_AAAI','sudoku')
print(ris)
ris = generate_time_table('./results_oracle_AAAI','lgps')
print(ris)


#------------------------RQ5-Reported in the table
base_directory = "results_real_users/sudoku"
aggregated_results,all_aggregated_results = process_json_files(base_directory)
time_results = process_time_data(base_directory)
latex_table = generate_RQ5_table(aggregated_results,time_results)
print(latex_table)


#------------------------Appendix - Regret Summary
roots = [
    "results_oracle_AAAI/lgps_norm_2/tuner_None_exploration_1/lr_0.1_diversification_baseline",
    "results_oracle_AAAI/lgps_norm_2/tuner_None_exploration_1/lr_0.1_diversification_w_disjunction",
    "results_oracle_AAAI/lgps_norm_2/tuner_None_exploration_1/lr_0.5_diversification_MACHOP",
    "results_oracle_AAAI/lgps_norm_2_steps_SMUS/tuner_None_exploration_1/lr_0.5_diversification_MACHOP"
]
labels = ["Choice Perceptron Online","Learned Weights Online","MACHOP Online", "MACHOP Offline – SES "]

summary = summarize_regrets(roots, labels)

latex_table = latex_regret_table_star(
    summary,
    caption="Oracle evaluation: Regret summary across users for Logic Grid Puzzles.",
    label="tab:regret_real_user"
)
print(latex_table)

roots = [
    "results_oracle_AAAI/sudoku_2/tuner_None_exploration_1/lr_0.5_diversification_baseline",
    "results_oracle_AAAI/sudoku_2/tuner_None_exploration_1/lr_5_diversification_w_disjunction",
    "results_oracle_AAAI/sudoku_2/tuner_None_exploration_1/lr_0.5_diversification_MACHOP",
    "results_oracle_AAAI/sudoku_2_steps_SMUS/tuner_None_exploration_1/lr_10_diversification_MACHOP"
]
labels = ["Choice Perceptron Online","Learned Weights Online","MACHOP Online", "MACHOP Offline – SES "]

summary = summarize_regrets(roots, labels)

latex_table = latex_regret_table_star(
    summary,
    caption="Oracle evaluation: Regret summary across users for Sudokus.",
    label="tab:regret_real_user"
)
print(latex_table)

##------------------------Appendix - Summary percentage of picked solutions proposed by MACHOP and Choice Perceptron
base_directory = "results_real_users/sudoku"  # Change this to your folder's path
aggregated_results,all_aggregated_results = process_json_files(base_directory)

ris10_m = summarize_metrics(all_aggregated_results['SMUS_machop_10'])
ris30_m = summarize_metrics(all_aggregated_results['SMUS_machop_30'])
ris50_m = summarize_metrics(all_aggregated_results['SMUS_machop_50'])

# BASELINE summaries
ris10_b = summarize_metrics(all_aggregated_results['SMUS_baseline_10'])
ris30_b = summarize_metrics(all_aggregated_results['SMUS_baseline_30'])
ris50_b = summarize_metrics(all_aggregated_results['SMUS_baseline_50'])

latex_table_compare = latex_compare_machop_baseline_table(
    [ris10_m, ris30_m, ris50_m],
    [ris10_b, ris30_b, ris50_b],
    ["SMUS_machop_10", "SMUS_machop_30", "SMUS_machop_50"]
)

print(latex_table_compare)


# Compute delta for each query level
for q in [10, 30, 50]:
    machop = all_aggregated_results[f'SMUS_machop_{q}']['Learnt']
    baseline = all_aggregated_results[f'SMUS_baseline_{q}']['Learnt']
    delta = cliffs_delta(machop, baseline)
    print(f"Query {q}: Cliff's delta = {delta:.3f}")
