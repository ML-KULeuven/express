import ast
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, patches
from matplotlib.table import Table


def plot_sudoku_explanations(all_rows, col_title=None, additional_titles=None,
                             col_grid="grid", col_hint_facts="hint_facts", col_hint_constraints="hint_constraints",
                             col_hint_derived="hint_derived"):
    if isinstance(all_rows, pd.DataFrame):
        all_rows = [row for _, row in all_rows.iterrows()]
    sudoku_grids = []
    explained_cells = []
    explained_vals = []
    used_rows = []
    used_cols = []
    used_blocks = []
    used_cells = []
    # Number of subplots: One row, multiple columns
    for _,row_df in enumerate(all_rows):

        # Parse the sudoku grid
        if isinstance(row_df[col_grid], np.ndarray):
            sudoku_grid = row_df[col_grid]
        else:
            data_str = row_df[col_grid].replace("[", "").replace("]", "").strip()
            rows = data_str.split("\n")
            data = [list(map(int, row.split())) for row in rows]
            sudoku_grid = np.array(data)

        # Create the table for displaying the grid
        sudoku_grids.append(sudoku_grid)

        if isinstance(row_df[col_hint_facts], str):
            hint_facts = ast.literal_eval(row_df[col_hint_facts])
            hint_constraints = ast.literal_eval(row_df[col_hint_constraints])
            hint_derived = ast.literal_eval(row_df[col_hint_derived])
        else:
            # Extract the facts, constraints, and derived values
            hint_facts = list_to_tuple(row_df[col_hint_facts])
            hint_constraints = list_to_tuple(row_df[col_hint_constraints])
            hint_derived = to_tuple(row_df[col_hint_derived])

        all_facts = {(row, col): val for fact_type, (row, col, val) in hint_facts}
        all_row_cons = [hint for hint_type, hint in hint_constraints if hint_type == "ROW"]
        all_col_cons = [hint for hint_type, hint in hint_constraints if hint_type == "COL"]
        all_block_cons = [list(hint[0]) for hint_type, hint in hint_constraints if hint_type == "BLOCK" ]

        _, (ex_x, ex_y, derived_val) = hint_derived
        explained_cells.append([ex_x, ex_y])
        explained_vals.append(derived_val)
        used_cells.append(all_facts)
        used_rows.append(all_row_cons)
        used_cols.append(all_col_cons)
        used_blocks.append(all_block_cons)

    input1 = {
        'grid': sudoku_grids[0],
        'explained_cell': explained_cells[0],
        'explained_val': explained_vals[0],
        'used_rows': used_rows[0],
        'used_cols': used_cols[0],
        'used_blocks': used_blocks[0],
        'used_cells': used_cells[0]
    }

    input2 = {
        'grid': sudoku_grids[1],
        'explained_cell': explained_cells[1],
        'explained_val': explained_vals[1],
        'used_rows': used_rows[1],
        'used_cols': used_cols[1],
        'used_blocks': used_blocks[1],
        'used_cells': used_cells[1]
    }
    draw_two_sudoku_explanations(input1,input2)

def draw_two_sudoku_explanations(data1, data2):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))  # Side-by-side plots

    draw_single_sudoku(axs[0], **data1)
    draw_single_sudoku(axs[1], **data2)

    plt.tight_layout()
    plt.show()

def draw_single_sudoku(ax, grid, explained_cell, explained_val, used_rows, used_cols, used_blocks, used_cells):
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 9)
    ax.set_xticks(np.arange(0, 10))
    ax.set_yticks(np.arange(0, 10))
    ax.grid(which='major', color='black', linewidth=1)
    ax.invert_yaxis()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params('both', length=0, width=0, which='major')

    eps = 0.2

    # Draw the grid values
    for i in range(9):
        for j in range(9):
            val = grid[i, j]
            if val != 0:
                ax.text(j + 0.5, i + 0.5, str(val), va='center', ha='center', fontsize=14)

    # Highlight explained cell
    ex_x, ex_y = explained_cell
    ax.add_patch(patches.Rectangle((ex_y, ex_x), 1, 1, facecolor="lightgreen"))
    ax.text(ex_y + 0.5, ex_x + 0.5, str(explained_val), va='center', ha='center', fontsize=14)

    # Highlight referenced cells
    for cx, cy in used_cells:
        ax.add_patch(patches.Rectangle((cy, cx), 1, 1, facecolor='#ffe165', linewidth=2))

    # Bold 3x3 block boundaries
    for i in range(0, 10, 3):
        ax.axhline(i, color='black', linewidth=2)
        ax.axvline(i, color='black', linewidth=2)

    # Highlight used rows
    for r in used_rows:
        ax.add_patch(
            patches.Rectangle((0 + eps, r + eps), 9 - 2 * eps, 1 - 2 * eps, edgecolor='blue', fill=False, linewidth=2))

    # Highlight used columns
    for c in used_cols:
        ax.add_patch(
            patches.Rectangle((c + eps, 0 + eps), 1 - 2 * eps, 9 - 2 * eps, edgecolor='blue', fill=False, linewidth=2))

    # Highlight used blocks (3x3)
    for b in used_blocks:
        block_row, block_col = b
        ax.add_patch(
            patches.Rectangle((block_col + eps, block_row + eps), 3 - 2 * eps, 3 - 2 * eps,
                              edgecolor='blue', fill=False, linewidth=2))
    ax.set_aspect('equal')

def list_to_tuple(lst):

    return [to_tuple(elem) for elem in lst]

def to_tuple(lst):

    return tuple(to_tuple(x) if isinstance(x, list) else x for x in lst)




import os
import pandas as pd

import os
import re
import pandas as pd


def escape_latex(text: str) -> str:
    return text.replace("_", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")


def simplify_label(label: str, type_prefix: str) -> str:
    # Mapping normalization names
    mapping = {
        f"{type_prefix}_0": "no normalization",
        f"{type_prefix}_1": "nadir",
        f"{type_prefix}_2": "local",
        f"{type_prefix}_3": "cumulative"
    }

    # Replace based on prefix match
    for raw, pretty in mapping.items():
        if label.startswith(raw):
            label = label.replace(raw, pretty)
            break

    # Remove lr_ values
    label = re.sub(r"lr_[0-9.]+", "", label)

    # Remove "diversification"
    label = label.replace("diversification", "")

    # Replace remaining underscores with space
    label = label.replace("_", " ").strip()

    return escape_latex(label)


def escape_latex(text: str) -> str:
    return text.replace("_", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")

def simplify_label(label: str, type_prefix: str) -> str:
    mapping = {
        f"{type_prefix}_0": "no normalization",
        f"{type_prefix}_1": "nadir",
        f"{type_prefix}_2": "local",
        f"{type_prefix}_3": "cumulative"
    }

    for raw, pretty in mapping.items():
        if label.startswith(raw):
            label = label.replace(raw, pretty)
            break

    label = re.sub(r"lr_[0-9.]+", "", label)
    label = label.replace("diversification", "")
    label = label.replace("cpucb", "machop")
    label = label.replace("disjunction", "disj.")
    label = label.replace("_", " ").strip()
    label = re.sub(r"\s+", " ", label)  # Replace multiple spaces with one

    return escape_latex(label)





def escape_latex(text: str) -> str:
    return text.replace("_", " ").replace("&", "\\&").replace("%", "\\%").replace("$", "\\$")

def simplify_label(label: str, type_prefix: str) -> str:
    mapping = {
        f"{type_prefix}_norm_0": "no normalization",
        f"{type_prefix}_norm_1": "nadir",
        f"{type_prefix}_norm_2": "local",
        f"{type_prefix}_norm_3": "cumulative"
    }

    for raw, pretty in mapping.items():
        if label.startswith(raw):
            label = label.replace(raw, pretty)
            break

    label = re.sub(r"lr_[0-9.]+", "", label)
    label = label.replace("diversification", "")
    label = label.replace("cpucb", "machop")
    label = label.replace("disjunction", "disj.")
    label = label.replace("_", " ").strip()
    label = re.sub(r"\s+", " ", label)  # Normalize spacing

    return escape_latex(label)

def generate_time_table(root_folder: str, type_prefix: str) -> str:
    rows = []

    for type_folder in os.listdir(root_folder):
        if not type_folder.startswith(type_prefix) or '2' not in type_folder :
            continue

        type_path = os.path.join(root_folder, type_folder)
        if not os.path.isdir(type_path):
            continue

        intermediate_path = next((os.path.join(type_path, d) for d in os.listdir(type_path)
                                  if os.path.isdir(os.path.join(type_path, d))), None)
        if intermediate_path is None:
            continue

        for run_folder in os.listdir(intermediate_path):
            run_path = os.path.join(intermediate_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            sum_means, sum_stds = [], []

            for inner_folder_name in os.listdir(run_path):
                inner_folder = os.path.join(run_path, inner_folder_name)
                dataset_file = os.path.join(inner_folder, "time.csv")

                if os.path.isdir(inner_folder) and os.path.isfile(dataset_file):
                    df = pd.read_csv(dataset_file)
                    if "time exp 1" in df.columns and "time exp 2" in df.columns:
                        time_sum = df[["time exp 1", "time exp 2"]].dropna().sum(axis=1)
                        if not time_sum.empty:
                            sum_means.append(time_sum.mean())
                            sum_stds.append(time_sum.std())

            if sum_means:
                overall_mean = pd.Series(sum_means).mean()
                overall_std = pd.Series(sum_means).std()
                label_raw = f"{type_folder}_{run_folder}"
                label = simplify_label(label_raw, type_prefix)
                rows.append((label, overall_mean, overall_std))

    # Build LaTeX table
    latex_table = "\\begin{tabular}{p{4cm}cc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Method & Avg. Total Time & Std Total Time \\\\\n"
    latex_table += "\\midrule\n"

    for row in rows:
        latex_table += f"{row[0]} & {row[1]:.2f} & {row[2]:.2f} \\\\\n"

    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}"

    return latex_table
