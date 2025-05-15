import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from bayes_opt import BayesianOptimization
from brokenaxes import brokenaxes
from scipy.interpolate import griddata

def load_scores(file_path):
    data = pd.read_csv(file_path, sep='\t', usecols=['Key', 'S_bandwidth', 'S_DOS'])
    data['S_DOS'] = data['S_DOS'].replace('N/A', np.nan).astype(float)
    data['S_bandwidth'] = data['S_bandwidth'].replace('N/A', np.nan).astype(float)
    return data

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_s_total_with_hdbscan(X, lambda_val, beta, alpha, min_cluster_size):
    mask_non_minus_one = X[:, 0] != -1
    X_valid = X[mask_non_minus_one]

    X_transformed = X_valid

    clusterer = hdbscan.HDBSCAN(min_cluster_size=int(min_cluster_size), min_samples=5, cluster_selection_method='eom')
    labels = clusterer.fit_predict(X_transformed)

    high_density_cluster = -1
    max_score = -float('inf')
    unique_labels = np.unique(labels[labels != -1]) 
    if len(unique_labels) > 0:
        for label in unique_labels:
            cluster_points = X_valid[labels == label]
            cluster_center = np.mean(cluster_points, axis=0)
            score = cluster_center[0] + cluster_center[1]  
            if score > max_score:
                max_score = score
                high_density_cluster = label

    s_total = np.zeros(len(X))
    s_total[~mask_non_minus_one] = -1 

    offset_1 = 0.2  #  [0.5, 1.5]
    offset_2 = 1.2  #  [0.1, 1.0]
    s_total[mask_non_minus_one] = (
            sigmoid(lambda_val * (X_valid[:, 0] + X_valid[:, 1] - offset_1)) *
            sigmoid(beta * (X_valid[:, 0] * X_valid[:, 1])  - offset_2)
    )

    return s_total, labels, high_density_cluster, mask_non_minus_one


def compute_target_function(lambda_val, beta, alpha, min_cluster_size, X):
   
    s_total, labels, high_density_cluster, mask_non_minus_one = compute_s_total_with_hdbscan(
        X, lambda_val, beta, alpha, min_cluster_size
    )

    s_total_valid = s_total[mask_non_minus_one]
    X_valid = X[mask_non_minus_one]
    labels_valid = labels

    high_density_mask = labels_valid == high_density_cluster

    bandwidth_threshold_low = np.percentile(X_valid[:, 0], 10)
    dos_threshold_low = np.percentile(X_valid[:, 1], 10)
    low_score_mask = (X_valid[:, 0] <= bandwidth_threshold_low) | (X_valid[:, 1] <= dos_threshold_low)

    high_density_s_total = s_total_valid[high_density_mask]
    high_density_mean = np.mean(high_density_s_total) if len(high_density_s_total) > 0 else 0

    low_score_s_total = s_total_valid[low_score_mask]
    low_score_mean = np.mean(low_score_s_total) if len(low_score_s_total) > 0 else 0

    threshold = np.percentile(s_total_valid, 90)
    high_s_total_count = np.sum(s_total_valid >= threshold)
    high_s_total_ratio = high_s_total_count / len(s_total_valid) if len(s_total_valid) > 0 else 0

    w1, w2, w3 = 0.4, 0.3, 0.3
    target = w1 * high_density_mean - w2 * low_score_mean - w3 * high_s_total_ratio

    return target, high_density_mean, low_score_mean, high_s_total_ratio, s_total, labels, high_density_cluster, mask_non_minus_one


def save_results(scores_data, output_file):
    scores_data.to_csv(output_file, sep='\t', index=False)
    print(f"Results saved to {output_file}")


def calculate_s_total(scores_data, lambda_val, beta, alpha, min_cluster_size):

    scores_data_valid = scores_data[['S_bandwidth', 'S_DOS']].dropna()
    X = scores_data_valid.values
    s_total, _, _, _ = compute_s_total_with_hdbscan(X, lambda_val, beta, alpha, min_cluster_size)

    mask_non_minus_one = X[:, 0] != -1
    s_total_valid = s_total[mask_non_minus_one]
    if len(s_total_valid) > 0:
        min_val = np.min(s_total_valid)
        max_val = np.max(s_total_valid)
        if max_val > min_val:  
            s_total[mask_non_minus_one] = (s_total_valid - min_val) / (max_val - min_val)

    s_total[~mask_non_minus_one] = 0

    scores_data['S_total'] = np.nan  
    valid_indices = scores_data.index[scores_data[['S_bandwidth', 'S_DOS']].notna().all(axis=1)]
    scores_data.loc[valid_indices, 'S_total'] = s_total

    return scores_data, s_total

def optimize_parameters_with_bayesian(scores_data):
    X = scores_data[['S_bandwidth', 'S_DOS']].dropna().values
    if len(X) < 2:
        print("Not enough valid data points for optimization.")
        return None, None, None, None, None, None, None, None

    def target_function(lambda_val, beta, alpha, min_cluster_size):
        target, _, _, _, _, _, _, _ = compute_target_function(
            lambda_val, beta, alpha, min_cluster_size, X
        )
        return target

    pbounds = {
        'lambda_val': (0, 10),
        'beta': (0, 50),
        'alpha': (2, 5),
        'min_cluster_size': (5, 50)
    }

    optimizer = BayesianOptimization(
        f=target_function,
        pbounds=pbounds,
        random_state=1,
        verbose=2
    )

    optimizer.maximize(
        init_points=5,
        n_iter=50
    )

    best_params = optimizer.max['params']
    best_lambda_val = best_params['lambda_val']
    best_beta = best_params['beta']
    best_alpha = best_params['alpha']
    best_min_cluster_size = best_params['min_cluster_size']

    _, best_high_density_mean, best_low_score_mean, best_high_s_total_ratio, best_s_total, best_labels, best_high_density_cluster, best_mask_non_minus_one = compute_target_function(
        best_lambda_val, best_beta, best_alpha, best_min_cluster_size, X
    )

    mask_non_minus_one = X[:, 0] != -1
    s_total_valid = best_s_total[mask_non_minus_one]
    if len(s_total_valid) > 0:
        min_val = np.min(s_total_valid)
        max_val = np.max(s_total_valid)
        if max_val > min_val:
            best_s_total[mask_non_minus_one] = (s_total_valid - min_val) / (max_val - min_val)
    best_s_total[~mask_non_minus_one] = 0

    lambda_values = []
    beta_values = []
    target_values = []
    high_density_means = []
    low_score_means = []
    high_s_total_ratios = []
    s_total_means = []
    s_total_stds = []

    for res in optimizer.res:
        lambda_val = res['params']['lambda_val']
        beta = res['params']['beta']
        alpha = res['params']['alpha']
        min_cluster_size = res['params']['min_cluster_size']
        target, high_density_mean, low_score_mean, high_s_total_ratio, s_total, _, _, _ = compute_target_function(
            lambda_val, beta, alpha, min_cluster_size, X
        )

        s_total_valid = s_total[mask_non_minus_one]
        if len(s_total_valid) > 0:
            min_val = np.min(s_total_valid)
            max_val = np.max(s_total_valid)
            if max_val > min_val:
                s_total[mask_non_minus_one] = (s_total_valid - min_val) / (max_val - min_val)
        s_total[~mask_non_minus_one] = 0

        s_total_mean = np.mean(s_total_valid) if len(s_total_valid) > 0 else 0
        s_total_std = np.std(s_total_valid) if len(s_total_valid) > 0 else 0

        lambda_values.append(lambda_val)
        beta_values.append(beta)
        target_values.append(target)
        high_density_means.append(high_density_mean)
        low_score_means.append(low_score_mean)
        high_s_total_ratios.append(high_s_total_ratio)
        s_total_means.append(s_total_mean)
        s_total_stds.append(s_total_std)

    return (
    lambda_values, beta_values, target_values, high_density_means, low_score_means, high_s_total_ratios, s_total_means,
    s_total_stds,
    best_lambda_val, best_beta, best_alpha, best_min_cluster_size, best_s_total, best_labels, best_high_density_cluster,
    best_mask_non_minus_one, X)


def scale_values(values, old_min=0.1, old_max=1.0, new_min=0.0, new_max=1.0):
    scaled = new_min + (values - old_min) * (new_max - new_min) / (old_max - old_min)
    return np.clip(scaled, new_min, new_max)


def main():
    input_file = "./data/scores.txt"
    output_file = "./data/scores_with_stotal.txt"

    scores_data = load_scores(input_file)
    print("Loaded data:")
    print(scores_data.head())
    print(
        f"S_bandwidth stats: mean={scores_data['S_bandwidth'].mean():.4f}, max={scores_data['S_bandwidth'].max():.4f}, min={scores_data['S_bandwidth'].min():.4f}")
    print(
        f"S_DOS stats: mean={scores_data['S_DOS'].mean():.4f}, max={scores_data['S_DOS'].max():.4f}, min={scores_data['S_DOS'].min():.4f}")

    result = optimize_parameters_with_bayesian(scores_data)
    if result is None:
        print("Failed to compute parameters due to insufficient data.")
        return

    (lambda_values, beta_values, target_values, high_density_means, low_score_means, high_s_total_ratios, s_total_means,
     s_total_stds,
     best_lambda_val, best_beta, best_alpha, best_min_cluster_size, best_s_total, best_labels,
     best_high_density_cluster, best_mask_non_minus_one, X) = result

    scores_data_with_stotal,best_s_total = calculate_s_total(scores_data, best_lambda_val, best_beta, best_alpha,
                                                best_min_cluster_size)

    best_s_total = scale_values(best_s_total)


    save_results(scores_data_with_stotal, output_file)


if __name__ == "__main__":
    main()