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

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def collate_fn(batch):
    graphs, texts, flatness_score = zip(*batch)
    flatness_score = torch.tensor(flatness_score, dtype=torch.float32).unsqueeze(-1)
    return (
        dgl.batch([g[0] for g in graphs]),
        dgl.batch([g[1] for g in graphs]),
        texts,
        flatness_score,
    )

class DGLDataset(Dataset):
    def __init__(self, data_dict):
        self.data = list(data_dict.items())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        key, value = self.data[index]
        structure_graph = value["structure_graph"]
        flatness_score = value["flatness_score"]
        text_string = value["text_string"]
        return structure_graph, text_string, flatness_score

def predict(model, data_loader, device):
    model.eval()
    all_predictions = []
    all_true_values = []
    all_keys = []

    with torch.no_grad():
        for batch in data_loader:
            graph1, graph2, text_input, flatness_score = batch
            graph1 = graph1.to(device)
            graph2 = graph2.to(device)
            flatness_score = flatness_score.to(device)

            predicted_score = model(graph1, graph2, text_input, device, mode="infer")

            all_predictions.extend(predicted_score.cpu().numpy().reshape(-1))
            all_true_values.extend(flatness_score.cpu().numpy().reshape(-1))
            all_keys.extend(text_input)

    return all_keys, all_predictions, all_true_values

def preprocess_band_structure(dataset_dic, flatness_dic):
    common_keys = set(dataset_dic.keys()) & set(flatness_dic.keys())
    dataset_full = {key: dataset_dic[key].copy() for key in common_keys}
    for key in common_keys:
        score = flatness_dic[key]
        dataset_full[key]['flatness_score'] = score
    return dataset_full

def load_existing_scores(output_file):
    scores_dict = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line.strip():
                    key, score = line.strip().split('\t')
                    scores_dict[key] = float(score)
    return scores_dict

def main():

    cfg = get_cfg_defaults()

    best_model_path = cfg["DIR"]["SAVEMODEL"] + '/best_model.pth'
    prediction_output_file = cfg["DIR"]["SAVEMODEL"] + '/predictions.txt'

    with open(cfg["DIR"]["picklefile"], 'rb') as f:
        dataset_dic = pickle.load(f)

    scores_dict = load_existing_scores(cfg["DIR"]["bandfile"])
    dataset_full = preprocess_band_structure(dataset_dic, scores_dict)

    flatness_scores = [v["flatness_score"] for v in dataset_full.values()]
    print(f"Flatness score mean: {np.mean(flatness_scores)}, std: {np.std(flatness_scores)}")

    val_data = dataset_full

    val_dataset = DGLDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=cfg["MODEL"]["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn)

    model = Main_model(**cfg).to(device)
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"Loaded best model from {best_model_path}")
    else:
        raise FileNotFoundError(f"Best model not found at {best_model_path}")

    keys, predictions, true_values = predict(model, val_loader, device)

    with open(prediction_output_file, 'w') as f:
        f.write("Key\tPredicted\tTrue\n")
        for key, pred, true in zip(keys, predictions, true_values):
            f.write(f"{key}\t{pred:.6f}\t{true:.6f}\n")

    print(f"Predictions saved to {prediction_output_file}")

if __name__ == '__main__':

    main()
