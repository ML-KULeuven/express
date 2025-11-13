import glob
import json
import os

import os
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def extract_lr_and_diversification(folder_name):
    match = re.match(r'lr_(.*?)_diversification_(.*)', folder_name)
    if match:
        return match.groups()
    return None, None


def extract_user_number(subfolder_name):
    match = re.search(r'user_(\d+)', subfolder_name)
    return int(match.group(1)) if match else None


def find_lowest_regret(file_path):
    df = pd.read_csv(file_path)
    last_col = df.columns[-1]  # Get the last column name
    min_val = df[last_col].min()
    min_row = df[last_col].idxmin() + 1  # Convert to 1-based index
    return min_val, min_row

def find_last_regret(file_path):
    df = pd.read_csv(file_path)
    last_col = df.columns[-1]  # Get the last column name
    min_val = df.iloc[-1][last_col]
    return min_val


def find_cumulative_regret(file_path):
    df = pd.read_csv(file_path)
    last_col = df.columns[-1]  # Get the last column name
    min_val = sum(df[last_col])/len(df[last_col])
    return min_val

def find_draw(file_path):
    df = pd.read_csv(file_path)
    count_draws = 100 * df['picked'].eq('Draw').sum() / len(df['picked'])
    return count_draws

def compute_regret_hyperparam(results_folder):
    tables = {}
    aggregated_results = {}

    for folder in os.listdir(results_folder):
        lr_value, diversification = extract_lr_and_diversification(folder)
        if lr_value is None or diversification is None:
            continue

        folder_path = os.path.join(results_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            user_number = extract_user_number(subfolder)
            if user_number is None:
                continue

            regret_file = os.path.join(folder_path, subfolder, 'regret.csv')
            if not os.path.exists(regret_file):
                continue

            min_val, min_row = find_lowest_regret(regret_file)
            formatted_value = f"{min_val:.3f} ({(min_row+1)*20})"

            # Store per-user results
            if user_number not in tables:
                tables[user_number] = {}

            if diversification not in tables[user_number]:
                tables[user_number][diversification] = {}

            tables[user_number][diversification][lr_value] = (min_val, formatted_value)

            # Store values for aggregation
            if diversification not in aggregated_results:
                aggregated_results[diversification] = {}

            if lr_value not in aggregated_results[diversification]:
                aggregated_results[diversification][lr_value] = []

            aggregated_results[diversification][lr_value].append(min_val)

    # Display per-user tables and plots
    for user_number, table_data in tables.items():
        df = pd.DataFrame.from_dict(table_data, orient='index').sort_index()
        df = df[sorted(df.columns, key=float)]  # Sort columns numerically

        df_numeric = df.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        df_labels = df.applymap(lambda x: x[1] if isinstance(x, tuple) else "")

        print(f"User {user_number} Results:")
        print(df_labels)
        print("\n")

        # Plot the table
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_numeric, annot=df_labels, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

        plt.title(f"Relative Regret for User {user_number}")
        plt.xlabel("Learning Rate (lr)")
        plt.ylabel("Query Selection Strategy")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Adjust layout
        plt.subplots_adjust(left=0.3)

        # Save images
        plt.savefig(f"./graphs/regret_user_{user_number}_results.png", bbox_inches='tight')
        plt.show()

    # Compute aggregated results (mean and standard deviation)
    avg_table = {
        div: {lr: (np.mean(vals), np.std(vals)) for lr, vals in lr_dict.items()}
        for div, lr_dict in aggregated_results.items()
    }

    df_avg = pd.DataFrame.from_dict(avg_table, orient='index').sort_index()
    df_avg = df_avg[sorted(df_avg.columns, key=float)]  # Sort columns numerically

    # Format as mean (std)
    df_labels_avg = df_avg.applymap(lambda x: f"{x[0]:.2f}±{x[1]:.2f}" if not pd.isna(x) else "")

    print("Average Regret Across Users (Mean ± Std):")
    print(df_labels_avg)
    print("\n")

    # Extract mean regret values
    df_means = df_avg.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)

    # Get indices of the top-5 lowest mean regrets
    min_indices = np.dstack(np.unravel_index(np.argsort(df_means.values, axis=None)[:5], df_means.shape))[0]

    # Plot the aggregated table with highlighted top-5 lowest regrets
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(df_means, annot=df_labels_avg, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Highlight the top-5 lowest regret cells
    for idx in min_indices:
        ax.add_patch(plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.title("Average Relative Regret Across Users")
    plt.xlabel("Learning Rate (lr)")
    plt.ylabel("Query Selection Strategy")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.subplots_adjust(left=0.3)

    # Save images
    plt.savefig("./graphs/avg_regret_results.png", bbox_inches='tight')
    plt.show()

    return tables, df_avg

def compute_regret_hyperparam_last(results_folder):
    tables = {}
    aggregated_results = {}

    for folder in os.listdir(results_folder):
        lr_value, diversification = extract_lr_and_diversification(folder)
        if lr_value is None or diversification is None:
            continue

        folder_path = os.path.join(results_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            user_number = extract_user_number(subfolder)
            if user_number is None:
                continue

            regret_file = os.path.join(folder_path, subfolder, 'regret.csv')
            if not os.path.exists(regret_file):
                continue

            min_val = find_last_regret(regret_file)
            formatted_value = f"{min_val:.3f}"

            # Store per-user results
            if user_number not in tables:
                tables[user_number] = {}

            if diversification not in tables[user_number]:
                tables[user_number][diversification] = {}

            tables[user_number][diversification][lr_value] = (min_val, formatted_value)

            # Store values for aggregation
            if diversification not in aggregated_results:
                aggregated_results[diversification] = {}

            if lr_value not in aggregated_results[diversification]:
                aggregated_results[diversification][lr_value] = []

            aggregated_results[diversification][lr_value].append(min_val)

    # Compute aggregated results (mean and standard deviation)
    avg_table = {
        div: {lr: (np.mean(vals), np.std(vals)) for lr, vals in lr_dict.items()}
        for div, lr_dict in aggregated_results.items()
    }

    df_avg = pd.DataFrame.from_dict(avg_table, orient='index').sort_index()
    df_avg = df_avg[sorted(df_avg.columns, key=float)]  # Sort columns numerically

    # Format as mean (std)
    df_labels_avg = df_avg.applymap(lambda x: f"{x[0]:.2f}±{x[1]:.2f}" if not pd.isna(x) else "")

    print("Average Regret Across Users (Mean ± Std):")
    print(df_labels_avg)
    print("\n")

    return tables, df_avg



def compute_time_hyperparam_last(results_folder):
    tables = {}
    aggregated_results = {}

    for folder in os.listdir(results_folder):
        lr_value, diversification = extract_lr_and_diversification(folder)
        if lr_value is None or diversification is None:
            continue

        folder_path = os.path.join(results_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            user_number = extract_user_number(subfolder)
            if user_number is None:
                continue

            regret_file = os.path.join(folder_path, subfolder, 'time.csv')
            if not os.path.exists(regret_file):
                continue

            min_val = find_average_time(regret_file)
            formatted_value = f"{min_val:.3f}"

            # Store per-user results
            if user_number not in tables:
                tables[user_number] = {}

            if diversification not in tables[user_number]:
                tables[user_number][diversification] = {}

            tables[user_number][diversification][lr_value] = (min_val, formatted_value)

            # Store values for aggregation
            if diversification not in aggregated_results:
                aggregated_results[diversification] = {}

            if lr_value not in aggregated_results[diversification]:
                aggregated_results[diversification][lr_value] = []

            aggregated_results[diversification][lr_value].append(min_val)

    # Display per-user tables and plots
    for user_number, table_data in tables.items():
        df = pd.DataFrame.from_dict(table_data, orient='index').sort_index()
        df = df[sorted(df.columns, key=float)]  # Sort columns numerically

        df_numeric = df.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        df_labels = df.applymap(lambda x: x[1] if isinstance(x, tuple) else "")

        print(f"User {user_number} Results:")
        print(df_labels)
        print("\n")

        # Plot the table
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(df_numeric, annot=df_labels, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

        plt.title(f"Relative Regret for User {user_number}")
        plt.xlabel("Learning Rate (lr)")
        plt.ylabel("Query Selection Strategy")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Adjust layout
        plt.subplots_adjust(left=0.3)

        # Save images
        # plt.savefig(f"regret_user_{user_number}_results.png", bbox_inches='tight')
        # plt.show()

    # Compute aggregated results (mean and standard deviation)
    avg_table = {
        div: {lr: (np.mean(vals), np.std(vals)) for lr, vals in lr_dict.items()}
        for div, lr_dict in aggregated_results.items()
    }

    df_avg = pd.DataFrame.from_dict(avg_table, orient='index').sort_index()
    df_avg = df_avg[sorted(df_avg.columns, key=float)]  # Sort columns numerically

    # Format as mean (std)
    df_labels_avg = df_avg.applymap(lambda x: f"{x[0]:.2f}±{x[1]:.2f}" if not pd.isna(x) else "")

    print("Average Regret Across Users (Mean ± Std):")
    print(df_labels_avg)
    print("\n")

    # Extract mean regret values
    df_means = df_avg.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)

    # Get indices of the top-5 lowest mean regrets
    min_indices = np.dstack(np.unravel_index(np.argsort(df_means.values, axis=None)[:5], df_means.shape))[0]

    # Plot the aggregated table with highlighted top-5 lowest regrets
    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(df_means, annot=df_labels_avg, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Highlight the top-5 lowest regret cells
    for idx in min_indices:
        ax.add_patch(plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.title("Average Relative Regret Across Users")
    plt.xlabel("Learning Rate (lr)")
    plt.ylabel("Query Selection Strategy")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.subplots_adjust(left=0.3)

    # Save images
    plt.savefig("avg_regret_results_last.png", bbox_inches='tight')
    plt.show()

    return tables, df_avg




def compute_cumulative_regret_hyperparam(results_folder):
    tables = {}
    aggregated_results = {}

    for folder in os.listdir(results_folder):
        lr_value, diversification = extract_lr_and_diversification(folder)
        if lr_value is None or diversification is None:
            continue

        folder_path = os.path.join(results_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            user_number = extract_user_number(subfolder)
            if user_number is None:
                continue

            regret_file = os.path.join(folder_path, subfolder, 'regret.csv')
            if not os.path.exists(regret_file):
                continue

            min_val = find_cumulative_regret(regret_file)
            formatted_value = f"{min_val:.3f}"

            # Store per-user results
            if user_number not in tables:
                tables[user_number] = {}

            if diversification not in tables[user_number]:
                tables[user_number][diversification] = {}

            tables[user_number][diversification][lr_value] = (min_val, formatted_value)

            # Store values for aggregation
            if diversification not in aggregated_results:
                aggregated_results[diversification] = {}

            if lr_value not in aggregated_results[diversification]:
                aggregated_results[diversification][lr_value] = []

            aggregated_results[diversification][lr_value].append(min_val)

    # Display per-user tables and plots
    for user_number, table_data in tables.items():
        df = pd.DataFrame.from_dict(table_data, orient='index').sort_index()
        df = df[sorted(df.columns, key=float)]  # Sort columns numerically

        df_numeric = df.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        df_labels = df.applymap(lambda x: x[1] if isinstance(x, tuple) else "")

        print(f"User {user_number} Results:")
        print(df_labels)
        print("\n")

        # Plot the table
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_numeric, annot=df_labels, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

        plt.title(f"Cumulative Regret for User {user_number}")
        plt.xlabel("Learning Rate (lr)")
        plt.ylabel("Query Selection Strategy")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Adjust layout
        plt.subplots_adjust(left=0.3)

        # Save images
        plt.savefig(f"cumulative_regret_user_{user_number}_results.png", bbox_inches='tight')
        plt.show()

    # Compute aggregated results (mean and standard deviation)
    avg_table = {
        div: {lr: (np.mean(vals), np.std(vals)) for lr, vals in lr_dict.items()}
        for div, lr_dict in aggregated_results.items()
    }

    df_avg = pd.DataFrame.from_dict(avg_table, orient='index').sort_index()
    df_avg = df_avg[sorted(df_avg.columns, key=float)]  # Sort columns numerically

    # Format as mean (std)
    df_labels_avg = df_avg.applymap(lambda x: f"{x[0]:.2f}±{x[1]:.2f}" if not pd.isna(x) else "")

    print("Cumulative Regret Across Users (Mean ± Std):")
    print(df_labels_avg)
    print("\n")

    # Extract mean regret values
    df_means = df_avg.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)

    # Get indices of the top-5 lowest mean regrets
    min_indices = np.dstack(np.unravel_index(np.argsort(df_means.values, axis=None)[:5], df_means.shape))[0]

    # Plot the aggregated table with highlighted top-5 lowest regrets
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(df_means, annot=df_labels_avg, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Highlight the top-5 lowest regret cells
    for idx in min_indices:
        ax.add_patch(plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.title("Average Cumulative Relative Regret Across Users")
    plt.xlabel("Learning Rate (lr)")
    plt.ylabel("Query Selection Strategy")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.subplots_adjust(left=0.3)

    # Save images
    plt.savefig("avg_cumualtive_regret_results.png", bbox_inches='tight')
    plt.show()

    return tables, df_avg

def compute_draws_hyperparam(results_folder):
    tables = {}
    aggregated_results = {}

    for folder in os.listdir(results_folder):
        lr_value, diversification = extract_lr_and_diversification(folder)
        if lr_value is None or diversification is None:
            continue

        folder_path = os.path.join(results_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            user_number = extract_user_number(subfolder)
            if user_number is None:
                continue

            regret_file = os.path.join(folder_path, subfolder, 'dataset.csv')
            if not os.path.exists(regret_file):
                continue

            min_val = find_draw(regret_file)
            formatted_value = f"{min_val:.3f}"

            # Store per-user results
            if user_number not in tables:
                tables[user_number] = {}

            if diversification not in tables[user_number]:
                tables[user_number][diversification] = {}

            tables[user_number][diversification][lr_value] = (min_val, formatted_value)

            # Store values for aggregation
            if diversification not in aggregated_results:
                aggregated_results[diversification] = {}

            if lr_value not in aggregated_results[diversification]:
                aggregated_results[diversification][lr_value] = []

            aggregated_results[diversification][lr_value].append(min_val)

    # Display per-user tables and plots
    for user_number, table_data in tables.items():
        df = pd.DataFrame.from_dict(table_data, orient='index').sort_index()
        df = df[sorted(df.columns, key=float)]  # Sort columns numerically

        df_numeric = df.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        df_labels = df.applymap(lambda x: x[1] if isinstance(x, tuple) else "")

        print(f"User {user_number} Results:")
        print(df_labels)
        print("\n")

        # Plot the table
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.heatmap(df_numeric, annot=df_labels, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

        plt.title(f"% Draws for User {user_number}")
        plt.xlabel("Learning Rate (lr)")
        plt.ylabel("Query Selection Strategy")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        # Adjust layout
        plt.subplots_adjust(left=0.3)

        # Save images
        plt.savefig(f"draws_user_{user_number}_results.png", bbox_inches='tight')
        plt.show()

    # Compute aggregated results (mean and standard deviation)
    avg_table = {
        div: {lr: (np.mean(vals), np.std(vals)) for lr, vals in lr_dict.items()}
        for div, lr_dict in aggregated_results.items()
    }

    df_avg = pd.DataFrame.from_dict(avg_table, orient='index').sort_index()
    df_avg = df_avg[sorted(df_avg.columns, key=float)]  # Sort columns numerically

    # Format as mean (std)
    df_labels_avg = df_avg.applymap(lambda x: f"{x[0]:.2f}±{x[1]:.2f}" if not pd.isna(x) else "")

    print("% Draws Across Users (Mean ± Std):")
    print(df_labels_avg)
    print("\n")

    # Extract mean regret values
    df_means = df_avg.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)

    # Get indices of the top-5 lowest mean regrets
    min_indices = np.dstack(np.unravel_index(np.argsort(df_means.values, axis=None)[:5], df_means.shape))[0]

    # Plot the aggregated table with highlighted top-5 lowest regrets
    fig, ax = plt.subplots(figsize=(18, 6))
    sns.heatmap(df_means, annot=df_labels_avg, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

    # Highlight the top-5 lowest regret cells
    for idx in min_indices:
        ax.add_patch(plt.Rectangle((idx[1], idx[0]), 1, 1, fill=False, edgecolor='black', lw=3))

    plt.title("% Draws Across Users Users")
    plt.xlabel("Learning Rate (lr)")
    plt.ylabel("Query Selection Strategy")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Adjust layout
    plt.subplots_adjust(left=0.3)

    # Save images
    plt.savefig("avg_draws_results.png", bbox_inches='tight')
    plt.show()

    return tables, df_avg



def plot_regret_from_folder(root_folder):
    # Extract lr value and diversification name from the root folder name
    match = re.search(r'lr_(.*?)_diversification_(.*)', os.path.basename(root_folder))
    if not match:
        raise ValueError("Folder name does not match the expected format 'lr_{value}_diversification_{name}'")

    lr_value, diversification_name = match.groups()

    # Iterate through subfolders
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path) and subfolder.startswith("sudoku_steps_standard_user_"):
            # Extract user number
            user_match = re.search(r'sudoku_steps_standard_user_(\d+)', subfolder)
            if not user_match:
                continue
            user_number = user_match.group(1)

            # Path to regret.csv
            regret_file = os.path.join(subfolder_path, "regret.csv")
            if os.path.exists(regret_file):
                # Read data
                df = pd.read_csv(regret_file)

                if 'labelled data' not in df.columns or 'regret' not in df.columns:
                    print(f"Skipping {regret_file}: Missing required columns.")
                    continue

                # Plot
                plt.figure(figsize=(8, 5))
                plt.plot(df['labelled data'], df['regret'], marker='o', linestyle='-')
                plt.xlabel("Labelled Data")
                plt.ylabel("Regret")
                plt.title(
                    f"Relative regret for user {user_number} with lr {lr_value} and diversification {diversification_name}")
                plt.grid(True)

                # Save figure in the subfolder
                output_path = os.path.join(subfolder_path, "regret_plot.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Saved plot: {output_path}")


diversification_types = {"baseline", "disjunction", "w_disjunction", "coverage", "w_coverage", "coverage_sum",
                         "w_coverage_sum"}




def find_lowest_regret(file_path):
    df = pd.read_csv(file_path)
    min_idx = df['regret'].idxmin()
    return df.loc[min_idx, 'regret'], min_idx


def compute_regret_tuner(results_folder):
    tables = {}
    aggregated_results = {}
    dict_meaning_tuner = {'False':'Baseline','True':'Custom','None':'Tuned'}

    for tuner_folder in ["tuner_False", "tuner_None", "tuner_True"]:
        tuner_path = os.path.join(results_folder, tuner_folder)
        if not os.path.isdir(tuner_path):
            continue

        tuner_label = tuner_folder.split("_")[1]  # Extract None, False, or True
        tuner_label = dict_meaning_tuner[tuner_label]
        for subfolder in os.listdir(tuner_path):
            _,diversification = extract_lr_and_diversification(subfolder)
            diversification = "_".join(diversification.split("_")[:-2]) if "_tuner_" in diversification else diversification
            if diversification is None:
                continue

            div_path = os.path.join(tuner_path, subfolder)
            if not os.path.isdir(div_path):
                continue

            for user_folder in os.listdir(div_path):
                regret_file = os.path.join(div_path, user_folder, 'regret.csv')
                if not os.path.exists(regret_file):
                    continue

                min_val, _ = find_lowest_regret(regret_file)

                if user_folder not in tables:
                    tables[user_folder] = {}
                if diversification not in tables[user_folder]:
                    tables[user_folder][diversification] = {}

                tables[user_folder][diversification][tuner_label] = min_val

                if diversification not in aggregated_results:
                    aggregated_results[diversification] = {}
                if tuner_label not in aggregated_results[diversification]:
                    aggregated_results[diversification][tuner_label] = []

                aggregated_results[diversification][tuner_label].append(min_val)

    # Compute mean and std
    avg_table = {
        div: {tuner: (np.mean(vals), np.std(vals)) for tuner, vals in tuner_dict.items()}
        for div, tuner_dict in aggregated_results.items()
    }

    df_avg = pd.DataFrame.from_dict(avg_table, orient='index').sort_index()
    df_avg = df_avg[["Baseline", "Tuned", "Custom"]]  # Ensure correct column order

    # Format as mean ± std
    df_labels_avg = df_avg.applymap(lambda x: f"{x[0]:.2f}±{x[1]:.2f}" if not pd.isna(x) else "")

    print("Average Regret Across Users (Mean ± Std):")
    print(df_labels_avg)
    print("\n")

    # Plot heatmap
    df_means = df_avg.applymap(lambda x: x[0] if isinstance(x, tuple) else np.nan)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_means, annot=df_labels_avg, fmt="", cmap="coolwarm", linewidths=0.5, ax=ax)

    plt.title("Average Relative Regret Across Users")
    plt.xlabel("Tuner")
    plt.ylabel("Query Selection Strategy")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.subplots_adjust(left=0.3)
    plt.savefig("avg_regret_results.png", bbox_inches='tight')
    plt.show()

    return tables, df_avg


def find_average_time(time_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(time_file)

    # Compute the average for both columns 'time exp 1' and 'time exp 2'
    avg_time_exp1 = (df['time exp 1'] + df['time exp 2']).mean()

    return avg_time_exp1

def compute_time_hyperparam(results_folder):
    aggregated_results = {}

    for folder in os.listdir(results_folder):
        lr_value, diversification = extract_lr_and_diversification(folder)
        if lr_value is None or diversification is None:
            continue

        folder_path = os.path.join(results_folder, folder)
        if not os.path.isdir(folder_path):
            continue

        for subfolder in os.listdir(folder_path):
            user_number = extract_user_number(subfolder)
            if user_number is None:
                continue

            time_file = os.path.join(folder_path, subfolder, 'time.csv')
            if not os.path.exists(time_file):
                continue

            avg_time_exp1, avg_time_exp2 = find_average_time(time_file)  # Get averages for both columns

            # Store values for aggregation
            if diversification not in aggregated_results:
                aggregated_results[diversification] = {}

            if lr_value not in aggregated_results[diversification]:
                aggregated_results[diversification][lr_value] = {"exp1": [], "exp2": []}

            # Append the average times for both experiments
            aggregated_results[diversification][lr_value]["exp1"].append(avg_time_exp1)
            aggregated_results[diversification][lr_value]["exp2"].append(avg_time_exp2)

    # Compute aggregated results (mean ± std) for both experiments
    avg_table_exp1 = {
        div: {
            lr: f"{np.mean(vals['exp1']):.2f} ± {np.std(vals['exp1']):.2f}"
            for lr, vals in lr_dict.items()
        }
        for div, lr_dict in aggregated_results.items()
    }

    avg_table_exp2 = {
        div: {
            lr: f"{np.mean(vals['exp2']):.2f} ± {np.std(vals['exp2']):.2f}"
            for lr, vals in lr_dict.items()
        }
        for div, lr_dict in aggregated_results.items()
    }

    # Create DataFrames for each experiment
    df_avg_exp1 = pd.DataFrame.from_dict(avg_table_exp1, orient='index').sort_index()

    df_avg_exp2 = pd.DataFrame.from_dict(avg_table_exp2, orient='index').sort_index()

    # Replace NaN with an empty string
    df_avg_exp1 = df_avg_exp1.applymap(lambda x: "" if pd.isna(x) else x)
    df_avg_exp2 = df_avg_exp2.applymap(lambda x: "" if pd.isna(x) else x)

    return df_avg_exp1, df_avg_exp2




def format_table(dfs, column_names):
    """Formats multiple pre-processed DataFrames into a LaTeX table."""
    assert len(dfs) == len(column_names), "Mismatch between number of DataFrames and column names"

    # Extract all unique methods (row indices)
    all_methods = sorted(set().union(*[df.index for df in dfs]))

    # Initialize LaTeX table
    latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l " + "c " * len(column_names) + "}\n\\hline\n"
    latex_table += "Method & " + " & ".join(column_names) + " \\\\\n\\hline\n"

    for method in all_methods:
        row_values = []
        latex_method = method.replace("_", "\\_")  # Escape underscores for LaTeX

        for df in dfs:
            if method in df.index:
                # Extract the single non-empty value in the row
                value = next((val for val in df.loc[method] if val), "")
                value = value.replace("±", " \\pm ")  # Replace ± with \pm
            else:
                value = ""
            row_values.append(f"${value}$")

        latex_table += f"{latex_method} & {' & '.join(row_values)} \\\\\n"

    latex_table += "\\hline\n\\end{tabular}\n\\caption{Regret for Sudoku (avg over 5 users, 5 seeds)}\n"
    latex_table += "\\label{tab:regret_results}\n\\end{table}"

    return latex_table


def extract_row_col_values(df, rows_to_consider, row_to_col_map):
    result = {}
    for row in rows_to_consider:
        if row in row_to_col_map:
            col = row_to_col_map[row]
            cell = df.at[row, str(col)]
            # Ensure it's a tuple of two np.float64
            if isinstance(cell, tuple) and len(cell) == 2 and all(isinstance(x, np.float64) for x in cell):
                result[row] = [float(cell[0]), float(cell[1])]
    return result


def plot_RQ1(dicts, title, y_label, y_max=40, problem = 'sudoku'):
    dict_labels = ['Default normal.', 'No normal.', 'Cumul. normal.', 'Local normal.']

    all_keys = sorted(set().union(*[d.keys() for d in dicts]))

    vibrant_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    key_to_color = {key: vibrant_colors[i % len(vibrant_colors)] for i, key in enumerate(all_keys)}

    # ✅ Also swapped dicts[0] and dicts[1] to match new order
    dicts = [dicts[1], dicts[0], dicts[2], dicts[3]]

    x = np.arange(len(dicts))
    width = 0.8 / len(all_keys)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, key in enumerate(all_keys):
        values = [d.get(key, [0, 0])[0] for d in dicts]
        stds = [d.get(key, [0, 0])[1] for d in dicts]
        positions = x + (i - (len(all_keys) - 1) / 2) * width

        bars = ax.bar(
            positions, values, width, yerr=stds,
            label=key, color=key_to_color[key],
            capsize=5, alpha=0.85,
            edgecolor='black', linewidth=1.2
        )

        for bar, val in zip(bars, values):
            bar_height = bar.get_height()
            label_y = bar.get_y() + bar_height * 0.9
            if bar_height < 1:
                label_y = bar.get_y() + bar_height - 0.05

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{val:.2f}', ha='center', va='top', fontsize=12,
                color='white', fontweight='bold'
            )

    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(dict_labels, fontsize=16, fontweight='bold')

    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(0, y_max)
    ax.legend(fontsize=16, title_fontsize=16, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f'./graphs/RQ1_{problem}.png', dpi=300)
    plt.show()

def plot_lgp_sudoku_comparison(lgp_data, sudoku_data,
                               y_label="Average Relative Regret",
                               title="Relative Regret",
                               y_max=None,
                               save_path='relative_regret_rq2.png'):
    # --- Settings ---
    renamed_methods = {
        'L1': 'L₁',
        'w_L1': 'learned weights',
        'cpucb_L1': 'UCB weights'
    }

    vibrant_colors = ['#ff7f0e', '#2ca02c', '#1f77b4']
    method_to_color = {
        'L1': vibrant_colors[0],
        'w_L1': vibrant_colors[1],
        'cpucb_L1': vibrant_colors[2]
    }

    methods = list(lgp_data.keys())

    # Extract values and errors
    lgp_values = [lgp_data[m][0] for m in methods]
    lgp_errors = [lgp_data[m][1] for m in methods]
    sudoku_values = [sudoku_data[m][0] for m in methods]
    sudoku_errors = [sudoku_data[m][1] for m in methods]

    x = np.arange(2)  # LGP and Sudoku groups
    width = 0.2

    if y_max is None:
        y_max = max(max(lgp_values), max(sudoku_values)) + max(max(lgp_errors), max(sudoku_errors)) + 0.5

    group_labels = ['LGP', 'Sudoku']

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        values = [lgp_values[i], sudoku_values[i]]
        errors = [lgp_errors[i], sudoku_errors[i]]
        position = x + i * width - width
        color = method_to_color.get(method, '#7f7f7f')

        label = renamed_methods.get(method, method)  # plain text, no italics

        bar = ax.bar(
            position, values, width, yerr=errors,
            label=label,
            color=color,
            capsize=5, alpha=0.85,
            edgecolor='black', linewidth=1.2
        )

        # Value labels on bars
        for j, b in enumerate(bar):
            bar_height = b.get_height()
            label_y = b.get_y() + bar_height * 0.9
            if bar_height < 1:
                label_y = b.get_y() + bar_height - 0.05

            ax.text(
                b.get_x() + b.get_width() / 2,
                label_y,
                f'{values[j]:.2f}', ha='center', va='top', fontsize=10,
                color='white', fontweight='bold'
            )

    # Axis and styling
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=16, fontweight='bold')

    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(0, y_max)
    ax.legend(fontsize=16, title_fontsize=16, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()





def plot_explanation_times(data):


    labels = [row[0] for row in data]
    mean1 = [row[1] for row in data]
    std1 = [row[2] for row in data]
    mean2 = [row[3] for row in data]
    std2 = [row[4] for row in data]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    bars1 = ax.bar(x - width/2, mean1, width, label='first expl.',
                   yerr=std1, capsize=5, color='royalblue', edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, mean2, width, label='second expl.',
                   yerr=std2, capsize=5, color='orange', edgecolor='black', linewidth=1)

    ax.set_ylabel('Time (s)', fontsize=14)
    ax.set_title('Time Sudoku with $\mathbf{MACHOP}$', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight='bold')
    ax.tick_params(axis='y', labelsize=12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=14)

    # Add horizontal grid lines
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.6)

    # Add white value labels inside each bar
    for bar_group, values in zip([bars1, bars2], [mean1, mean2]):
        for bar, value in zip(bar_group, values):
            bar_height = bar.get_height()
            label_y = bar.get_y() + bar_height * 0.9
            if bar_height < 1:
                label_y = bar.get_y() + bar_height - 0.05

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f'{value:.2f}', ha='center', va='top', fontsize=10,
                color='white', fontweight='bold'
            )
    plt.tight_layout()
    plt.savefig('explanation_times.png', dpi=300)
    plt.show()


def plot_lgp_sudoku_comparison(sudoku_data, lgp_data,
                               y_label="Average Relative Regret",
                               title="Relative Regret",
                               y_max=None,
                               save_path='./graphs/RQ3.png'):
    # --- Settings ---
    renamed_methods = {
        'L1': 'L1',
        'w_L1': 'learned weights',
        'cpucb_L1': 'UCB weights'
    }

    vibrant_colors = ['#ff7f0e', '#2ca02c', '#ff0000']
    method_to_color = {
        'L1': vibrant_colors[0],
        'w_L1': vibrant_colors[1],
        'cpucb_L1': vibrant_colors[2]
    }

    methods = list(sudoku_data.keys())

    # Extract values and errors
    lgp_values = [lgp_data[m][0] for m in methods]
    lgp_errors = [lgp_data[m][1] for m in methods]
    sudoku_values = [sudoku_data[m][0] for m in methods]
    sudoku_errors = [sudoku_data[m][1] for m in methods]

    x = np.arange(2)  # LGP and Sudoku groups
    width = 0.2

    if y_max is None:
        y_max = max(max(lgp_values), max(sudoku_values)) + max(max(lgp_errors), max(sudoku_errors)) + 0.5

    group_labels = ['Sudoku','LGP']

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, method in enumerate(methods):
        values = [sudoku_values[i], lgp_values[i]]
        errors = [sudoku_errors[i], lgp_errors[i]]
        position = x + i * width - width
        color = method_to_color.get(method, '#7f7f7f')

        label = renamed_methods.get(method, method)  # plain text, no italics

        bar = ax.bar(
            position, values, width, yerr=errors,
            label=label,
            color=color,
            capsize=5, alpha=0.85,
            edgecolor='black', linewidth=1.2
        )

        # Value labels on bars
        for j, b in enumerate(bar):
            bar_height = b.get_height()
            label_y = b.get_y() + bar_height * 0.9
            if bar_height < 1:
                label_y = b.get_y() + bar_height - 0.05

            ax.text(
                b.get_x() + b.get_width() / 2,
                label_y,
                f'{values[j]:.2f}', ha='center', va='top', fontsize=10,
                color='white', fontweight='bold'
            )

    # Axis and styling
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=16, fontweight='bold')

    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.set_ylim(0, y_max)
    ax.legend(fontsize=16, title_fontsize=16, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def generate_RQ5_table(aggregated_results, time_aggregation):
    # Define the queries
    queries = [10, 30, 50]

    # Start the LaTeX table
    table = r"""
            \begin{table*}[bt!]
            \centering
            \caption{User study: generation time and \% picked solutions from SES with MACHOP and Choice Perceptron (ChoPerc).}
            \label{tab:RQ5}
            \renewcommand{\arraystretch}{0.9} % reduces row height
            \setlength{\tabcolsep}{4pt}      % reduces column spacing
            \begin{tabular}{@{}c cc ccc ccc@{}}
            \toprule
             & \multicolumn{2}{c}{\textbf{Query Generation Time (s)}} & \multicolumn{3}{c}{\textbf{Evaluation: SES vs MACHOP}} & \multicolumn{3}{c}{\textbf{Evaluation: SES vs ChoPerc}} \\
            \cmidrule(lr){2-3} \cmidrule(lr){4-6} \cmidrule(lr){7-9}
            \textbf{Query} & MACHOP & ChoPerc & SES (\%) & MACHOP (\%) & Indiff. (\%) & SES (\%) & ChoPerc (\%) & Indiff. (\%) \\
            \midrule
            """

    # Loop through the queries and extract the relevant values from the aggregated results and time data
    for query in queries:
        # Extract the data for MACHOP and ChoPerc
        machop_data = aggregated_results.get(f"SMUS_machop_{query}", {})
        choperc_data = aggregated_results.get(f"SMUS_baseline_{query}", {})  # baseline is used for ChoPerc

        # Extract values for SES, MACHOP, Indiff for each query
        ses_machop_mean, ses_machop_std = machop_data.get('SMUS', {}).get('mean', 0), machop_data.get('SMUS', {}).get('std_dev', 0)
        machop_mean, machop_std = machop_data.get('Learnt', {}).get('mean', 0), machop_data.get('Learnt', {}).get('std_dev', 0)
        indiff_machop_mean, indiff_machop_std = machop_data.get('Indifferent', {}).get('mean', 0), machop_data.get('Indifferent', {}).get('std_dev', 0)

        ses_choperc_mean, ses_choperc_std = choperc_data.get('SMUS', {}).get('mean', 0), choperc_data.get('SMUS', {}).get('std_dev', 0)
        choperc_mean, choperc_std = choperc_data.get('Learnt', {}).get('mean', 0), choperc_data.get('Learnt', {}).get('std_dev', 0)
        indiff_choperc_mean, indiff_choperc_std = choperc_data.get('Indifferent', {}).get('mean', 0), choperc_data.get('Indifferent', {}).get('std_dev', 0)

        # Multiply everything by 100 to convert to percentages
        ses_machop_mean, ses_machop_std = ses_machop_mean * 100, ses_machop_std * 100
        machop_mean, machop_std = machop_mean * 100, machop_std * 100
        indiff_machop_mean, indiff_machop_std = indiff_machop_mean * 100, indiff_machop_std * 100

        ses_choperc_mean, ses_choperc_std = ses_choperc_mean * 100, ses_choperc_std * 100
        choperc_mean, choperc_std = choperc_mean * 100, choperc_std * 100
        indiff_choperc_mean, indiff_choperc_std = indiff_choperc_mean * 100, indiff_choperc_std * 100

        # Retrieve the generation time for MACHOP and baseline (ChoPerc) from time_aggregation
        machop_time_mean = time_aggregation["MACHOP"].get(query, {}).get("mean", 0)
        machop_time_std = time_aggregation["MACHOP"].get(query, {}).get("std_dev", 0)
        choperc_time_mean = time_aggregation["baseline"].get(query, {}).get("mean", 0)
        choperc_time_std = time_aggregation["baseline"].get(query, {}).get("std_dev", 0)

        # Format the data into LaTeX format, adding the time values for each method
        table += f"{query} & {machop_time_mean:.1f} ± {machop_time_std:.1f} & {choperc_time_mean:.1f} ± {choperc_time_std:.1f} & {ses_machop_mean:.1f} ± {ses_machop_std:.1f} & {machop_mean:.1f} ± {machop_std:.1f} & {indiff_machop_mean:.1f} ± {indiff_machop_std:.1f} & {ses_choperc_mean:.1f} ± {ses_choperc_std:.1f} & {choperc_mean:.1f} ± {choperc_std:.1f} & {indiff_choperc_mean:.1f} ± {indiff_choperc_std:.1f} \\\\ \n"

    # End the table
    table += r"\bottomrule\end{tabular\end{table*}"

    return table


def process_time_data(base_dir):
    # Initialize dictionaries to store aggregated time data
    time_aggregation = {
        "baseline": {10: [], 30: [], 50: []},
        "MACHOP": {10: [], 30: [], 50: []}
    }

    # Traverse through all subfolders
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        if os.path.isdir(folder_path) and '_baseline' in folder or '_MACHOP' in folder:
            # Process both the baseline and MACHOP subfolders
            extracted_method = folder_path.split('_')[-1]
            time_file_path = os.path.join(folder_path, 'time.csv')

            if os.path.exists(time_file_path):
                # Read the time.csv file
                df = pd.read_csv(time_file_path)

                # Extract the relevant columns
                time_exp1 = df['time exp 1'].values
                time_exp2 = df['time exp 2'].values

                # Sum the values of the two columns for each of the first 10, 30, and 50 rows
                for rows in [10, 30, 50]:
                    # Sum the values of time_exp1 and time_exp2 element-wise
                    total_time = time_exp1[:rows] + time_exp2[:rows]  # Element-wise sum

                    # Now, sum the combined time for the specified rows
                    avg_time = np.sum(total_time)/rows

                    # Add the summed value to the aggregation dictionary
                    time_aggregation[extracted_method][rows].append(avg_time)

    # Compute average and standard deviation for each time group
    for method in time_aggregation:
        for rows in time_aggregation[method]:
            values = time_aggregation[method][rows]
            mean = np.mean(values)
            std_dev = np.std(values)
            time_aggregation[method][rows] = {"mean": mean, "std_dev": std_dev}

    return time_aggregation


def process_json_files(base_dir):
    # Initialize a dictionary to hold the sum of all JSON values by file name
    file_aggregation = {}
    file_aggregation_all = {}

    # Traverse through all subfolders
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Skip folders like 'results', '_baseline', or '_MACHOP'
        if os.path.isdir(folder_path) and (
                'results' not in folder and '_baseline' not in folder and '_MACHOP' not in folder):
            # Loop through the six JSON files
            for json_file in os.listdir(folder_path):
                if json_file.endswith(".json"):
                    file_path = os.path.join(folder_path, json_file)

                    # Read the JSON content
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Add the Indifference key by summing Draw and identical
                    data["Indifferent"] = data.get("Draw", 0) + data.get("identical", 0)
                    data.pop("Draw", None)
                    data.pop("identical", None)

                    # Normalize the values by their total sum
                    total_sum = sum(data.values())  # Sum all values (SMUS, Learnt, Indifferent)
                    normalized_data = {key: value / total_sum for key, value in data.items()}

                    # Extract the file name without the path to use for aggregation
                    file_name = json_file.split(".json")[0]

                    # Initialize the aggregation dictionary for each file name if it doesn't exist
                    if file_name not in file_aggregation:
                        file_aggregation[file_name] = {key: [] for key in normalized_data}
                        file_aggregation_all[file_name] = {key: [] for key in normalized_data}

                    # Add the normalized values to the aggregation dictionary
                    for key, value in normalized_data.items():
                        file_aggregation[file_name][key].append(value)
                        file_aggregation_all[file_name][key].append(value)

    # Calculate the average for each file by dividing by the number of instances
    for file_name in file_aggregation:
        for key in file_aggregation[file_name]:
            values = file_aggregation[file_name][key]  # List of values for each key

            # Calculate mean and standard deviation using numpy
            mean = np.mean(values)
            std_dev = np.std(values)

            # Store the results as a dictionary
            file_aggregation[file_name][key] = {"mean": mean, "std_dev": std_dev}

    return file_aggregation,file_aggregation_all



def process_json_files(base_dir):
    # Initialize a dictionary to hold the sum of all JSON values by file name
    file_aggregation = {}
    file_aggregation_all = {}

    # Traverse through all subfolders
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)

        # Skip folders like 'results', '_baseline', or '_MACHOP'
        if os.path.isdir(folder_path) and (
                'results' not in folder and '_baseline' not in folder and '_MACHOP' not in folder):
            # Loop through the six JSON files
            for json_file in os.listdir(folder_path):
                if json_file.endswith(".json"):
                    file_path = os.path.join(folder_path, json_file)

                    # Read the JSON content
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    # Add the Indifference key by summing Draw and identical
                    data["Indifferent"] = data.get("Draw", 0) + data.get("identical", 0)
                    data.pop("Draw", None)
                    data.pop("identical", None)

                    # Normalize the values by their total sum
                    total_sum = sum(data.values())  # Sum all values (SMUS, Learnt, Indifferent)
                    normalized_data = {key: value / total_sum for key, value in data.items()}

                    # Extract the file name without the path to use for aggregation
                    file_name = json_file.split(".json")[0]

                    # Initialize the aggregation dictionary for each file name if it doesn't exist
                    if file_name not in file_aggregation:
                        file_aggregation[file_name] = {key: [] for key in normalized_data}
                        file_aggregation_all[file_name] = {key: [] for key in normalized_data}

                    # Add the normalized values to the aggregation dictionary
                    for key, value in normalized_data.items():
                        file_aggregation[file_name][key].append(value)
                        file_aggregation_all[file_name][key].append(value)

    # Calculate the average for each file by dividing by the number of instances
    for file_name in file_aggregation:
        for key in file_aggregation[file_name]:
            values = file_aggregation[file_name][key]  # List of values for each key

            # Calculate mean and standard deviation using numpy
            mean = np.mean(values)
            std_dev = np.std(values)

            # Store the results as a dictionary
            file_aggregation[file_name][key] = {"mean": mean, "std_dev": std_dev}

    return file_aggregation,file_aggregation_all


def compute_regret_from_root(root: str) -> pd.DataFrame:
    """
    Read all runs under a ROOT directory and compute regrets for each user/run.
    Matches any folder of the form:
      sudoku_steps_<anystring>_user_*_id_*/regret.csv
    Returns DataFrame with columns: [user, run_id, regret].
    """
    pattern = os.path.join(root, "*_steps_*_user_*_id_*", "regret.csv")
    rows = []

    for fp in glob.glob(pattern):
        m = re.search(r"user_(\d+)_id_([^/]+)", fp)
        if not m:
            continue
        user = int(m.group(1))
        run_id = m.group(2)

        try:
            df = pd.read_csv(fp)
            regret = float(df["regret"].iloc[0])
            rows.append({"user": user, "run_id": run_id, "regret": regret})
        except Exception as e:
            print(f"Skipping {fp}: {e}")
            continue

    return pd.DataFrame(rows)


def summarize_regrets(roots, labels=None):
    """
    Compute summary stats (mean, median, percentiles, min–max) for regrets
    across multiple ROOT directories.
    """
    summaries = []
    if labels is None:
        labels = [os.path.basename(r.rstrip("/")) for r in roots]

    for root, label in zip(roots, labels):
        df = compute_regret_from_root(root)
        if df.empty:
            print(f"No data found for {label}")
            continue

        regrets = df["regret"].values
        summaries.append({
            "label": label,
            "mean": np.mean(regrets),
            "median": np.median(regrets),
            "p25": np.percentile(regrets, 25),
            "p75": np.percentile(regrets, 75),
            "min": np.min(regrets),
            "max": np.max(regrets)
        })

    return pd.DataFrame(summaries)


def latex_regret_table_star(summary_df: pd.DataFrame,
                            caption="",
                            label="",
                            decimals: int = 3) -> str:
    """
    Generate a wide LaTeX table* summarizing mean, median, percentiles, and range (no ICC, no n).
    """
    def fmt(x):
        return f"{x:.{decimals}f}" if pd.notnull(x) else "--"

    rows = []
    for _, r in summary_df.iterrows():
        rows.append(
            f"{r['label']} & "
            f"{fmt(r['mean'])} & "
            f"{fmt(r['median'])} & "
            f"{fmt(r['p25'])} & {fmt(r['p75'])} & "
            f"{fmt(r['min'])} & {fmt(r['max'])} \\\\"
        )

    table = r"""
    \begin{table*}[bt!]
    \centering
    \caption{""" + caption + r"""}
    \label{""" + label + r"""}
    \renewcommand{\arraystretch}{0.9}
    \setlength{\tabcolsep}{6pt}
    \begin{tabular}{@{}l c c cc cc@{}}
    \toprule
    & & & \multicolumn{2}{c}{\textbf{Percentile}} & \multicolumn{2}{c}{\textbf{Range}} \\
    \cmidrule(lr){4-5} \cmidrule(lr){6-7}
    \textbf{Setting} & \textbf{Mean} & \textbf{Median} & \textbf{25th} & \textbf{75th} & \textbf{Min} & \textbf{Max} \\
    \midrule
    """ + "\n".join(rows) + r"""
    \bottomrule
    \end{tabular}
    \end{table*}
    """
    return table


def summarize_metrics(metrics_dict):
    """
    Compute summary statistics (mean, median, 25th–75th percentile, min–max)
    for each key in a dictionary of metric lists.

    Parameters
    ----------
    metrics_dict : dict
        A dictionary where keys are metric names (e.g. 'Learnt', 'SMUS', 'Indifferent')
        and values are lists of numeric values (e.g. per-user percentages).

    Returns
    -------
    dict
        A dictionary of summary statistics for each metric.
    """
    summary = {}

    for key, values in metrics_dict.items():
        values = np.array(values)
        summary[key] = {
            "mean": np.mean(values),
            "median": np.median(values),
            "25th": np.percentile(values, 25),
            "75th": np.percentile(values, 75),
            "min": np.min(values),
            "max": np.max(values)
        }

    return summary


def latex_compare_machop_baseline_table(results_machop, results_baseline, labels):
    """
    Create a LaTeX table* with two horizontally aligned tables:
    one for MACHOP results and one for Baseline results.
    """

    def build_rows(results, labels):
        rows = []
        for label, result in zip(labels, results):
            stats = result['Learnt']
            query = label.split("_")[-1]
            line = (
                f"{query} & "
                f"{stats['mean']:.3f} & "
                f"{stats['median']:.3f} & "
                f"{stats['25th']:.3f} & "
                f"{stats['75th']:.3f} & "
                f"{stats['min']:.3f} & "
                f"{stats['max']:.3f} \\\\"
            )
            rows.append(line)
        return "\n".join(rows)

    # Build rows for each method
    machop_rows = build_rows(results_machop, labels)
    baseline_rows = build_rows(results_baseline, labels)

    # Full LaTeX combined figure
    latex = r"""
    \begin{table*}[bt!]
    \centering
    \caption{Comparison of summary statistics (mean, median, percentile, and range) for learnt solution preference across query counts.}
    \label{tab:machop_vs_baseline}
    \renewcommand{\arraystretch}{0.9}
    \setlength{\tabcolsep}{5pt}
    
    \begin{minipage}{0.48\textwidth}
    \centering
    \textbf{MACHOP}\\[3pt]
    \begin{tabular}{@{}c cccccc@{}}
    \toprule
    \textbf{Query} & \textbf{Mean} & \textbf{Median} &
    \multicolumn{2}{c}{\textbf{Percentile}} &
    \multicolumn{2}{c}{\textbf{Range}} \\
    \cmidrule(lr){4-5} \cmidrule(lr){6-7}
    & & & 25th & 75th & Min & Max \\
    \midrule
    """ + machop_rows + r"""
    \bottomrule
    \end{tabular}
    \end{minipage}
    \hfill
    \begin{minipage}{0.48\textwidth}
    \centering
    \textbf{Baseline}\\[3pt]
    \begin{tabular}{@{}c cccccc@{}}
    \toprule
    \textbf{Query} & \textbf{Mean} & \textbf{Median} &
    \multicolumn{2}{c}{\textbf{Percentile}} &
    \multicolumn{2}{c}{\textbf{Range}} \\
    \cmidrule(lr){4-5} \cmidrule(lr){6-7}
    & & & 25th & 75th & Min & Max \\
    \midrule
    """ + baseline_rows + r"""
    \bottomrule
    \end{tabular}
    \end{minipage}
    
    \end{table*}
    """
    return latex


