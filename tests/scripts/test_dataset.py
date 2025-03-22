import os
import sys
import argparse
import pickle
import csv
import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, roc_curve

from diffrfplus.model import TreeEnsemble, calculate_hyperparameters

DEBUG = True
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIRS = os.path.join(BASE_DIR, "data", "data_preprocessed")
sys.path.insert(0, BASE_DIR)

from model import DiFF_RF  # pylint: disable=import-error,wrong-import-position,wrong-import-order

def calculate_alpha_df(data, n_trees, sample_size, n_iter=5):

    possible_values = [1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1, 2, 5, 10, 100]
    r_alpha = {alpha: 0.0 for alpha in possible_values}

    num_parts = max(1, len(data) // sample_size)
    partitions = [data.loc[idx] for idx in np.array_split(data.index, num_parts)]

    # For each iteration
    for _ in range(n_iter):
        for i, p_i in enumerate(partitions):
            x_i = pd.concat([partitions[j] for j in range(len(partitions)) if j != i], ignore_index=True)

            for alpha in possible_values:
                print(f"Calculating alpha: {alpha} for partition {i+1}/{len(partitions)}")
                diff_rf = DiFF_RF.DiFF_TreeEnsemble(sample_size=sample_size, n_trees=n_trees)
                diff_rf.fit(x_i.values, n_jobs=8)

                scores_x = np.array(diff_rf.anomaly_score(x_i.values, alpha=alpha)[2])
                scores_p = np.array(diff_rf.anomaly_score(p_i.values, alpha=alpha)[2])

                delta = 0
                for perc in [95, 96, 97, 98, 99]:
                    quantile_value = np.percentile(scores_x, perc)
                    count = np.sum(scores_p > quantile_value)
                    delta += count * (100 - perc)
                r_alpha[alpha] += delta

    total_count = n_iter * len(partitions)
    for alpha in possible_values:
        r_alpha[alpha] /= total_count

    best_alpha = min(r_alpha, key=r_alpha.get)
    return best_alpha

def calculate_hyperparameters_df(data):
    n_trees = 128
    sample_size_ratio = 0.25
    sample_size = int(len(data) * sample_size_ratio)
    alpha = 0.1#calculate_alpha_df(data, n_trees, sample_size)
    kwargs = {
        "sample_size": sample_size,
        "n_trees": n_trees,
        "alpha": alpha
    }
    return kwargs

def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    precision = tp/(tp+fp)
    f1 = (2*tpr*precision)/(tpr+precision)
    return {'acc':acc,'tpr':tpr,'fpr':fpr,'precision':precision,'f1-score':f1}

def plot_confusion_matrix(y_true, y_pred, model_name, results_dir):
    cm = confusion_matrix(y_true, y_pred)
    group_counts = [f'{value:.0f}' for value in cm.ravel()]
    group_percentages = [f'{value*100:.2f}%' for value in cm.ravel() / np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.array(labels).reshape(cm.shape)
    ax = sns.heatmap(cm, annot=labels, cmap='Oranges',
                     xticklabels=['Predicted Benign', 'Predicted Malicious'],
                     yticklabels=['Actual Benign', 'Actual Malicious'], fmt='')
    # Save the heatmap as a PNG file named "<model_name>_cm.png" in results_dir
    filepath = os.path.join(results_dir, f"{model_name}_cm.png")
    ax.get_figure().savefig(filepath, bbox_inches='tight')
    plt.close(ax.get_figure())


def save_csv_results(results, out_dir):
    csv_file = os.path.join(out_dir, 'results.csv')
    with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['model', 'accuracy', 'true positive rate', 'false positive rate', 'precision', 'f1-score'])
        for model, metrics in results.items():
            row = [model]
            for _, value in metrics.items():
                row.append(value)
            writer.writerow(row)

def get_results_dir(test_name="dataset"):
    date_str = time.strftime("%Y%m%d_%H%M%S")
    results_parent = os.path.join(BASE_DIR, "results")
    os.makedirs(results_parent, exist_ok=True)

    results_dir = os.path.join(results_parent, f"{test_name}_{date_str}")
    os.makedirs(results_dir, exist_ok=True)

    return results_dir

def get_best_threshold(y, anomaly_scores):
    fpr, tpr, thresholds = roc_curve(y, anomaly_scores)
    df_val_roc = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
    df_val_roc['youden-index'] = df_val_roc['tpr'] - df_val_roc['fpr']
    df_val_roc.sort_values('youden-index', ascending=False).drop_duplicates('fpr').query('fpr < 0.03')
    # Remove np.inf, np.-inf and NaN values
    df_val_roc = df_val_roc[~df_val_roc['thresholds'].isin([np.inf, -np.inf, np.nan])]
    best_threshold = df_val_roc.iloc[0]['thresholds']
    return best_threshold

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Test script options")

    # Define the command-line arguments
    parser.add_argument("-d", "--data_set", required=True, help="Pass data_set to be used (obrigatório)")
    parser.add_argument("-hy", "--hyperparameters", action="store_true", help="Recalculate hyperparameters (optional)")
    # Parse the arguments
    args = parser.parse_args()

    # Accessing the values from the arguments
    data_set = args.data_set
    recalculate_hyperparameters = args.hyperparameters

    # Print or use the parsed arguments as needed
    print(f"Data set: {data_set}")
    hyperparameters = {
        "DiFF-RF": {},
        "DiFF-RF-Plus": {},
    }
    # Load the dataset
    x_train = pd.read_csv(os.path.join(DATASET_DIRS, data_set, "X_train.csv"))
    x_test = pd.read_csv(os.path.join(DATASET_DIRS, data_set, "X_test.csv"))
    x_val = pd.read_csv(os.path.join(DATASET_DIRS, data_set, "X_val.csv"))
    y_test = pd.read_csv(os.path.join(DATASET_DIRS, data_set, "y_test.csv"))
    y_val = pd.read_csv(os.path.join(DATASET_DIRS, data_set, "y_val.csv"))

    # Drop headers
    if DEBUG:
        x_train = x_train.sample(1000).reset_index(drop=True)
        x_test = x_test.sample(1000).reset_index(drop=True)
        x_val = x_val.sample(1000).reset_index(drop=True)
        y_test = y_test.sample(1000).reset_index(drop=True)
        y_val = y_val.sample(1000).reset_index(drop=True)


    if recalculate_hyperparameters:
        print("Recalculating hyperparameters...")
        hyperparameters["DiFF-RF"] = calculate_hyperparameters_df(x_train)
        hyperparameters["DiFF-RF-Plus"] = calculate_hyperparameters(x_train)

    else:
        hyperparameters_file = os.path.join(DATASET_DIRS, f"{data_set}_hyp.pkl")
        if os.path.exists(hyperparameters_file):
            with open(hyperparameters_file, "rb") as f:
                hyperparameters = pickle.load(f)
        else:
            print(f"Hyperparameters file not found for {data_set}.")
            hyperparameters["DiFF-RF"] = calculate_hyperparameters_df(x_train)
            hyperparameters["DiFF-RF-Plus"] = calculate_hyperparameters(x_train)
            with open(hyperparameters_file, "wb") as f:
                pickle.dump(hyperparameters, f)
    print(f"Hyperparameters: {hyperparameters}")


    models = {
        "DiFF-RF": DiFF_RF.DiFF_TreeEnsemble(n_trees=hyperparameters["DiFF-RF"]["n_trees"],
                                             sample_size=hyperparameters["DiFF-RF"]["sample_size"]),
        "DiFF-RF-Plus": TreeEnsemble(n_trees=hyperparameters["DiFF-RF-Plus"]["n_trees"], 
                                     sample_size=hyperparameters["DiFF-RF-Plus"]["sample_size"]),
    }

    results = {
        "DiFF-RF": {},
        "DiFF-RF-Plus": {},
    }
    results_dir = get_results_dir(data_set)

    for model_name, model in models.items():
        print(f"Training {model_name}...")

        model.fit(np.array(x_train), n_jobs=8)

        index_pointwise = 0 if model_name == "DiFF-RF" else "pointwise"
        index_collective = 2 if model_name == "DiFF-RF" else "collective"

        y_pred_val_pointwise = model.anomaly_score(np.array(x_val), alpha=hyperparameters[model_name]["alpha"])[index_pointwise]
        y_pred_val_collective = model.anomaly_score(np.array(x_val), alpha=hyperparameters[model_name]["alpha"])[index_collective]

        y_pred_test_pointwise = model.anomaly_score(np.array(x_test), alpha=hyperparameters[model_name]["alpha"])[index_pointwise]
        y_pred_test_collective = model.anomaly_score(np.array(x_test), alpha=hyperparameters[model_name]["alpha"])[index_collective]

        threshold_pointwise = get_best_threshold(y_val, y_pred_val_pointwise)
        threshold_collective = get_best_threshold(y_val, y_pred_val_collective)

        y_pred_val_pointwise = (y_pred_val_pointwise > threshold_pointwise).astype(int)
        y_pred_val_collective = (y_pred_val_collective > threshold_collective).astype(int)
        y_pred_test_pointwise = (y_pred_test_pointwise > threshold_pointwise).astype(int)
        y_pred_test_collective = (y_pred_test_collective > threshold_collective).astype(int)

        results[model_name]["threshold_pointwise"] = threshold_pointwise
        results[model_name]["threshold_collective"] = threshold_collective

        def populate_results(key, name, metrics):
            results[key][f"{name}_acc"] = metrics['acc']
            results[key][f"{name}_tpr"] = metrics['tpr']
            results[key][f"{name}_fpr"] = metrics['fpr']
            results[key][f"{name}_precision"] = metrics['precision']
            results[key][f"{name}_f1-score"] = metrics['f1-score']

        metrics_val_pointwise = get_overall_metrics(y_val, y_pred_val_pointwise)
        populate_results(model_name, "val_pointwise", metrics_val_pointwise)

        metrics_val_collective = get_overall_metrics(y_val, y_pred_val_collective)
        populate_results(model_name, "val_collective", metrics_val_collective)

        metrics_test_pointwise = get_overall_metrics(y_test, y_pred_test_pointwise)
        results[model_name]["test_pointwise"] = metrics_test_pointwise

        metrics_test_collective = get_overall_metrics(y_test, y_pred_test_collective)
        results[model_name]["test_collective"] = metrics_test_collective

        print(f"Results for {model_name} on validation set:")
        print(f"Pointwise: {metrics_val_pointwise}")
        print(f"Collective: {metrics_val_collective}")

        # Plot confusion matrix into results directory
        plot_confusion_matrix(y_val, y_pred_val_pointwise, model_name + "_pointwise", results_dir)
        plot_confusion_matrix(y_val, y_pred_val_collective, model_name + "_collective", results_dir)

    save_csv_results(results, results_dir)

if __name__ == "__main__":
    main()
