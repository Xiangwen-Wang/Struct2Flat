import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from graphencoder import GraphEncoder
from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
import torch.nn as nn

class TextEncoder(nn.Module):
    def __init__(self, bert_position_embeddings, bert_num_hidden_layers, bert_num_attention_heads, bert_hidden_size, out_dim):
        super(TextEncoder, self).__init__()
        bert_config = RobertaConfig.from_pretrained('roberta-base')
        bert_config.max_position_embeddings = bert_position_embeddings
        bert_config.num_hidden_layers = bert_num_hidden_layers
        bert_config.num_attention_heads = bert_num_attention_heads
        bert_config.hidden_size = bert_hidden_size

        self.base = RobertaModel.from_pretrained("roberta-base", config=bert_config, ignore_mismatched_sizes=True)
        self.head = nn.Sequential(
            nn.Linear(bert_hidden_size, bert_hidden_size),
            nn.ReLU(),
            nn.Linear(bert_hidden_size, out_dim)
        )

    def forward(self, input_ids, attention_mask):
        roberta_output = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = roberta_output.pooler_output
        return self.head(pooled_output)



class Main_model(nn.Module):
    def __init__(self, **config):
        super(Main_model, self).__init__()

        # Graph Encoder
        self.graph_gnn = GraphEncoder(
            config["GNN"]["classification"],
            config["GNN"]["atom_input_features"],
            config["GNN"]["hidden_features"],
            config["GNN"]["edge_input_features"],
            config["GNN"]["embedding_features"],
            config["GNN"]["triplet_input_features"],
            config["GNN"]["alignn_layers"],
            config["GNN"]["gcn_layers"],
            config["GNN"]["output_features"]
        )

        # Text Encoder
        self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        self.textencoder = TextEncoder(
            config["BERT"]["position_embeddings"],
            config["BERT"]["num_hidden_layers"],
            config["BERT"]["num_attention_heads"],
            config["BERT"]["hidden_size"],
            config["BAND"]["d_model"]
        )

        # Projection layer for graph embedding
        self.graph_end = nn.Linear(config["GNN"]["output_features"], config["BAND"]["d_model"])

        # Bilinear Interaction parameters
        self.bilinear_weight = nn.Parameter(
            torch.randn(config["BAND"]["d_model"], config["BAND"]["d_model"], config["BAND"]["d_model"])
        )

        # MLP for regression, 输入维度与 d_model 匹配
        self.mlp = nn.Sequential(
            nn.Linear(config["BAND"]["d_model"], 256),  # 输入维度改为 d_model
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, graph_input1, graph_input2, text_input, device, mode="train"):
        # Graph Encoding
        graph_embed = self.graph_gnn(graph_input1, graph_input2)  # [batch_size, output_features]
        graph_embed = self.graph_end(graph_embed)  # [batch_size, d_model]
        print(graph_embed.size())

        # Text Encoding
        encoded_inputs = self.tokenizer(
            text_input, return_tensors="pt", padding=True, truncation=True, padding_side="right"
        ).to(device)
        input_ids = encoded_inputs['input_ids']  # [batch_size, seq_len]
        attention_mask = encoded_inputs['attention_mask']  # [batch_size, seq_len]
        text_embed = self.textencoder(input_ids=input_ids, attention_mask=attention_mask)  # [batch_size, hidden_size]


        bilinear_out = torch.einsum('bi,ijk,bk->bj', graph_embed, self.bilinear_weight,
                                    text_embed)  # [batch_size, d_model]


        predict_score = self.mlp(bilinear_out)  # [batch_size, 1]

        return predict_score
