from yacs.config import CfgNode as CN

_C = CN()

_C.GNN = CN()
_C.GNN.alignn_layers = 4
_C.GNN.gcn_layers = 2
_C.GNN.atom_input_features = 92
_C.GNN.edge_input_features = 80
_C.GNN.triplet_input_features = 40
_C.GNN.embedding_features = 64
_C.GNN.hidden_features = 256
    # fc_layers: int = 1
    # fc_features: int = 64
_C.GNN.output_features = 256
    #if used to classify structures
_C.GNN.classification = False
_C.GNN.num_classes = 5

# ENCODER DECODER
_C.MODEL = CN()
_C.MODEL.MLP_LAYER = 3
_C.MODEL.DROPOUT = 0.1
_C.MODEL.lr_decay = 0.5
_C.MODEL.decay_interval = 10
_C.MODEL.NUM_EPOCHS = 60
_C.MODEL.BATCH_SIZE = 8
_C.MODEL.LR = 0.0001
_C.MODEL.WEIGHT_DECAY = 1e-4
_C.MODEL.SEED = 2048

_C.BERT = CN()
_C.BERT.position_embeddings = 514
_C.BERT.num_hidden_layers = 12
_C.BERT.num_attention_heads = 12
_C.BERT.hidden_size = 768

# BAND

_C.BAND = CN()
_C.BAND.num_layers = 6
_C.BAND.n_heads = 4
_C.BAND.d_model = 128
_C.BAND.num_bands = 6
_C.BAND.n_k = 64
_C.BAND.recovery_window = 16



# DIR
_C.DIR = CN()
_C.DIR.OUTPUT_DIR = "/Users/user/Desktop/AImaterials/results/flatness"
_C.DIR.picklefile = '/Users/user/Desktop/AImaterials/dataset/2dmatpedia/graph_text_2dmat.pkl'
_C.DIR.bandfile = '/Users/user/Desktop/AImaterials/dataset/2dmatpedia/score_with_rightspin/alldos_03/flatness_scores.txt'
_C.DIR.SAVEMODEL = '/Users/user/Desktop/AImaterials/results/flatness/03alldos'
_C.DIR.C2DBoutput = '/Users/user/Desktop/AImaterials/results/flatness/03alldos/c2db'
_C.DIR.C2DBfile = '/Users/user/Desktop/AImaterials/dataset/c2db/graph_text_2dmat.pkl'
def get_cfg_defaults():
    return _C.clone()