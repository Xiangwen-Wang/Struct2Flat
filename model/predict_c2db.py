import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pickle
import os
import numpy as np
import dgl
from torch.utils.data import DataLoader
from configs import get_cfg_defaults
from model import Main_model
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def collate_fn(batch):
    graphs, keys = zip(*batch)
    return (
        dgl.batch([g[0] for g in graphs]),
        dgl.batch([g[1] for g in graphs]),
        keys,
    )


class DGLDataset(Dataset):
    def __init__(self, data_dict):
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, value = self.data[index]
        structure_graph = value["structure_graph"]
        return structure_graph, key


def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_keys = []

    with torch.no_grad():
        for batch in data_loader:
            graph1, graph2, keys = batch
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            predicted_score = model(graph1, graph2, keys, device, mode="infer")
            all_predictions.extend(predicted_score.cpu().numpy().reshape(-1))
            all_keys.extend(keys)

    return all_keys, all_predictions


def predict_with_mc_dropout(model, data_loader, device, n_samples=30):
    model.train()  # 启用dropout
    all_keys = []
    all_predictions = []

    for batch in data_loader:
        graph1, graph2, keys = batch
        graph1 = graph1.to(device)
        graph2 = graph2.to(device)

        mc_predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(graph1, graph2, keys, device, mode="infer")
                mc_predictions.append(pred.cpu().numpy().reshape(-1))

        mc_predictions = np.stack(mc_predictions, axis=0)
        all_predictions.append(mc_predictions)
        all_keys.extend(keys)

    all_predictions = np.concatenate(all_predictions, axis=1)
    return all_keys, all_predictions


def plot_flatness_distribution(predictions, output_dir):
    plt.figure(figsize=(10, 6))
    plt.hist(predictions, bins=50, density=True, alpha=0.7)
    plt.xlabel('Predicted Flatness Score')
    plt.ylabel('Density')
    plt.title('Distribution of Predicted Flatness Scores')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, 'flatness_distribution.png')
    plt.savefig(output_path)
    plt.close()
    print(f"Flatness distribution plot saved to {output_path}")


def plot_overall_uncertainty(keys, mc_predictions, output_dir):
    mean_preds = np.mean(mc_predictions, axis=0)
    std_preds = np.std(mc_predictions, axis=0)

    # 按均值排序
    sort_indices = np.argsort(mean_preds)
    sorted_means = mean_preds[sort_indices]
    sorted_stds = std_preds[sort_indices]

    # 图1：按预测值排序的整体不确定性
    plt.figure(figsize=(9, 8))
    plt.plot(sorted_means, 'b-', label='Mean Prediction')
    plt.fill_between(range(len(sorted_means)),
                     sorted_means - 1.96 * sorted_stds,
                     sorted_means + 1.96 * sorted_stds,
                     alpha=0.3, color='blue',
                     label='95% Confidence Interval')
    plt.xlabel('Samples (sorted by prediction)',fontsize=18)
    plt.ylabel('Predicted Flatness Score',fontsize=18)
    plt.title('Overall Prediction Uncertainty',fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=18)
    output_path = os.path.join(output_dir, 'overall_uncertainty_sorted.png')
    plt.savefig(output_path, dpi=800)
    plt.show()
    plt.close()
    print(f"Overall uncertainty plot saved to {output_path}")

    # 图2：不确定性分布的直方图
    plt.figure(figsize=(10, 6))
    plt.hist(std_preds, bins=50, density=True, alpha=0.7, color='purple')
    plt.xlabel('Prediction Uncertainty (Standard Deviation)')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Uncertainty')
    plt.grid(True, alpha=0.3)
    output_path = os.path.join(output_dir, 'uncertainty_distribution.png')
    plt.savefig(output_path)

    plt.close()
    print(f"Uncertainty distribution plot saved to {output_path}")


def main():
    cfg = get_cfg_defaults()

    best_model_path = cfg["DIR"]["SAVEMODEL"] + '/best_model.pth'
    prediction_output_file = cfg["DIR"]["C2DBoutput"] + '/predictions.txt'
    output_dir = cfg["DIR"]["C2DBoutput"]

    os.makedirs(output_dir, exist_ok=True)

    with open(cfg["DIR"]["C2DBfile"], 'rb') as f:
        dataset_dic = pickle.load(f)

    dataset_full = dataset_dic
    val_dataset = DGLDataset(dataset_full)
    val_loader = DataLoader(val_dataset, batch_size=cfg["MODEL"]["BATCH_SIZE"],
                            shuffle=False, collate_fn=collate_fn)

    model = Main_model(**cfg).to(device)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    keys, predictions = predict(model, val_loader, device)
    mc_keys, mc_predictions = predict_with_mc_dropout(model, val_loader, device)

    with open(prediction_output_file, 'w') as f:
        f.write("Key\tPredicted Flatness Score\n")
        for key, pred in zip(keys, predictions):
            f.write(f"{key}\t{pred:.6f}\n")

    plot_flatness_distribution(predictions, output_dir)
    plot_overall_uncertainty(mc_keys, mc_predictions, output_dir)

    print(f"Predictions saved to {prediction_output_file}")


if __name__ == '__main__':
    main()