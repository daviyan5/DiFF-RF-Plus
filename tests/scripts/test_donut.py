#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diff-RF Donut script.

Workflow:
  1. Create/load a donut dataset (saved in 'pkl' folder).
  2. Build a results directory (under 'results/<test_name>_<date>/') where all PNGs and CSV will be stored.
  3. Plot cluster figures, compute anomaly scores using diff_rf and Isolation Forest,
     and save heat maps (as PNG) and CSV results.
"""

import os
import time
import pickle
import pathlib
import csv

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest

from diffrfplus.model import TreeEnsemble

plt.switch_backend('agg')
# Adjust plot settings
plt.gcf().subplots_adjust(bottom=0.15)
matplotlib.rcParams.update({'font.size': 22})

# Base directory for this script (assumed to be inside the tests folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKL_FILENAME = 'donut_data_problem.pkl'

def get_or_create_dir(subfolder):
    """Return a subdirectory path under BASE_DIR and create it if necessary."""
    directory = os.path.join(BASE_DIR, subfolder)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory


def get_results_dir(test_name="donut"):
    """Create and return a unique results directory for the current test run.
    
    The folder will be created under 'results/<test_name>_<date>/'.
    """
    date_str = time.strftime("%Y%m%d_%H%M%S")
    results_parent = get_or_create_dir("results")
    results_dir = os.path.join(results_parent, f"{test_name}_{date_str}")
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)
    return results_dir


def normalize_array(arr):
    """Normalize a numpy array to the range [0, 1]."""
    return (arr - arr.min()) / (arr.max() - arr.min())


def save_heatmap(data, score, title, filename, out_dir):
    """Create and save a heat map scatter plot as a PNG file."""
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], marker='o', c=score, cmap='viridis')
    plt.colorbar()
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.title(title)
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()


def gen_tore_vecs(dims, number, rmin, rmax):
    """Generate vectors with magnitudes distributed between rmin and rmax."""
    vecs = np.random.uniform(low=-1, size=(number, dims))
    radius = rmin + np.random.sample(number) * (rmax - rmin)
    mags = np.sqrt((vecs * vecs).sum(axis=-1))
    for i in range(number):
        vecs[i, :] = vecs[i, :] / mags[i] * radius[i]
    return vecs[:, 0], vecs[:, 1]


def create_donut_data():
    """Generate the donut dataset and save it to a pickle file in 'pkl' folder."""
    print('Building Donut data...')
    num_objs = 1000
    xn, yn = gen_tore_vecs(2, num_objs, 1.5, 4)
    x_n = np.array([xn, yn]).T

    num_objs_b = 1000
    mean = [0, 0]
    cov = [[0.5, 0], [0, 0.5]]
    xb, yb = np.random.multivariate_normal(mean, cov, num_objs_b).T
    x_b = np.array([xb, yb]).T

    num_objs_t = 1000
    xnt, ynt = gen_tore_vecs(2, num_objs_t, 1.5, 4)
    x_nt = np.array([xnt, ynt]).T

    mean = [3., 3.]
    cov = [[0.25, 0], [0, 0.25]]
    num_objs_a = 1000
    xa, ya = np.random.multivariate_normal(mean, cov, num_objs_a).T
    x_a = np.array([xa, ya]).T

    x_ab = np.concatenate([x_a, x_b])

    # Save dataset in the lowercase 'pkl' folder.
    pkl_dir = get_or_create_dir("pkl")
    pkl_file = os.path.join(pkl_dir, 'donut_data_problem.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump([x_n, x_nt, x_a, x_b, x_ab], f)


def load_donut_data():
    """Load and return the donut dataset from the pickle file."""
    pkl_dir = os.path.join(BASE_DIR, "pkl")
    pkl_file = os.path.join(pkl_dir, PKL_FILENAME)
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)  # returns (x_n, x_nt, x_a, x_b, x_ab)


def plot_clusters(x_n, x_a, x_b, out_dir):
    """Plot the cluster figures and save them as PNG files."""
    # Plot 1: Normal data
    plt.figure()
    plt.plot(x_n[:, 0], x_n[:, 1], 'bo', markersize=10)
    plt.savefig(os.path.join(out_dir, 'clusters_donnuts0.png'))
    plt.close()

    # Plot 2: Normal data and anomalous cluster (partial)
    nn = len(x_a)
    plt.figure()
    plt.plot(x_n[:, 0], x_n[:, 1], 'bo', x_a[:nn, 0], x_a[:nn, 1], 'rs')
    plt.savefig(os.path.join(out_dir, 'clusters_donnuts1.png'))
    plt.close()

    # Plot 3: Normal data, anomalous cluster, and background
    plt.figure()
    plt.plot(x_n[:, 0], x_n[:, 1], 'bo', x_a[:nn, 0], x_a[:nn, 1], 'rs', x_b[:nn, 0], x_b[:nn, 1], 'gd')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig(os.path.join(out_dir, 'clusters_donnuts2.png'))
    plt.close()


def compute_diff_rf_results(x_n, x_nt, x_ab, n_trees, sample_size_ratio, alpha_0, out_dir):
    """
    Build the diff_rf model, compute anomaly scores, plot heat maps,
    and return the test data along with a dictionary of anomaly score arrays.
    """
    sample_size = int(sample_size_ratio * len(x_n)) if sample_size_ratio <= 1 else sample_size_ratio

    diff_rf = TreeEnsemble(sample_size=sample_size, n_trees=n_trees)
    fit_start = time.time()
    diff_rf.fit(x_n, n_jobs=8)
    fit_time = time.time() - fit_start
    print(f"Fit time {fit_time:3.2f}s")
    n_nodes = sum(t.n_nodes for t in diff_rf.trees)
    print(f"{n_nodes} total nodes in {n_trees} trees")

    x_t = np.concatenate([x_nt, x_ab])
    scores = diff_rf.anomaly_score(x_t, alpha=alpha_0)
    sc_diff_rf = normalize_array(np.array(scores['collective']))
    sc_ff = normalize_array(np.array(scores['frequency']))
    sc_di = normalize_array(np.array(scores['pointwise']))

    save_heatmap(x_t, sc_ff, 'diff_rf (visiting frequency score) heat map',
                 'heatmap_diff_rf_freq_score.png', out_dir)
    save_heatmap(x_t, sc_diff_rf, 'diff_rf (collective anomaly score) heat map',
                 'heatmap_diff_rf_collective_score.png', out_dir)
    save_heatmap(x_t, sc_di, 'diff_rf (point-wise anomaly score) heat map',
                 'heatmap_diff_rf_point_wise_score.png', out_dir)

    cif = IsolationForest(n_estimators=n_trees, max_samples=sample_size, bootstrap=False, n_jobs=12)
    cif.fit(x_n)
    sc_if = normalize_array(-cif.decision_function(x_t))
    save_heatmap(x_t, sc_if, 'isolation forest heat map', 'heatmap_if.png', out_dir)

    return x_t, {
        'Isolation Forest': sc_if,
        'diff_rf (point-wise anomaly score)': sc_di,
        'diff_rf (frequency of visit scoring only)': sc_ff,
        'diff_rf (collective anomaly score)': sc_diff_rf
    }


def compute_auc_scores(y_true, scores_dict):
    """Compute and return AUC scores given the ground truth and a dictionary of scores."""
    auc_results = {}
    for method, score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        auc_results[method] = auc(fpr, tpr)
    return auc_results


def save_csv_results(auc_results, out_dir):
    """Write the AUC results to a CSV file in the given directory."""
    csv_file = os.path.join(out_dir, 'results.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Method', 'AUC'])
        for method, auc_val in auc_results.items():
            writer.writerow([method, auc_val])

def generate_readme(results_dir):
    """
    Generate a README.md file inside the results_dir.
    The README displays all PNG images side by side and the CSV content as a Markdown table.
    """

    # List PNG images in the results directory
    png_files = sorted([f for f in os.listdir(results_dir) if f.lower().endswith('.png')])

    # Build HTML for images side by side
    images_md = ""
    for png in png_files:
        images_md += f'<img src="{png}" width="300" style="margin-right: 10px;" />\n'

    # Read CSV and build a Markdown table
    csv_path = os.path.join(results_dir, "results.csv")
    csv_md = ""
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                # Create header
                header = rows[0]
                csv_md += "| " + " | ".join(header) + " |\n"
                csv_md += "| " + " | ".join(["---"] * len(header)) + " |\n"
                # Add rows
                for row in rows[1:]:
                    csv_md += "| " + " | ".join(row) + " |\n"

    # Combine everything into the README content
    readme_content = (
        "# Test Results\n\n"
        "## Generated Images\n\n"
        f"{images_md}\n\n"
        "## CSV Results\n\n"
        f"{csv_md}\n"
    )

    # Write the README.md file
    readme_path = os.path.join(results_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

def main(test_name="donut", n_trees=256, sample_size_ratio=0.25, alpha_0=1):
    # Create dataset
    if not os.path.exists(os.path.join(BASE_DIR, "pkl", PKL_FILENAME)):
        create_donut_data()
    x_n, x_nt, x_a, x_b, x_ab = load_donut_data()

    # Create unique results directory for this test run
    results_dir = get_results_dir(test_name)

    # Plot clusters and save results
    plot_clusters(x_n, x_a, x_b, results_dir)

    # Compute diff_rf anomaly scores, plot heat maps, and get scores dictionary
    _, scores = compute_diff_rf_results(x_n, x_nt, x_ab, n_trees, sample_size_ratio, alpha_0, results_dir)

    # Compute and print AUC scores
    y_true = np.array([-1] * len(x_nt) + [1] * len(x_ab))
    auc_results = compute_auc_scores(y_true, scores)
    for method, auc_val in auc_results.items():
        print(f"{method} AUC = {auc_val}")

    # Save AUC results to CSV
    save_csv_results(auc_results, results_dir)
    generate_readme(results_dir)


if __name__ == '__main__':
    main()
