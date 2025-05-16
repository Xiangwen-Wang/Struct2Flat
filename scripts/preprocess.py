import json
from torch.utils.data import Dataset
from GraphGen import main as GraphGen
import pickle
import numpy as np
import torch
import dgl
from pymatgen.core.structure import Structure
import pandas as pd

def generate_material_info_dict(filename):
    result = {}
    bandgap_dic = {}
    cbm_dic = {}
    vbm_dic = {}
    with (open(filename, 'r') as file):
        for line in file:
            try:
                item = json.loads(line)
                material_id = item.get("material_id")
                bandstructure = item.get("bandstructure", {})
                thermo = item.get("thermo", {})
                structure = item.get("structure", {})
                bandgap = bandstructure.get('bandgap')
                cbm = bandstructure.get('cbm')
                vbm = bandstructure.get('vbm')
                species = []
                frac_coords = []
                magmoms = []
                for atom in structure.get('sites',{}):
                    species.append(atom.get('label'))
                    frac_coords.append(atom.get('abc'))
                    magmoms.append(atom.get('properties', {}).get('magmom'))
                lattice_params = {'a': structure.get('lattice',{}).get('a'),
                                  'b': structure.get('lattice',{}).get('b'),
                                  'c': structure.get('lattice',{}).get('c'),
                                  'beta': structure.get('lattice',{}).get('beta'),
                                  'alpha': structure.get('lattice', {}).get('alpha'),
                                  'gamma': structure.get('lattice',{}).get('gamma')}

                info_str = (
                    f"formula_anonymous: {item.get('formula_anonymous')}, "
                    f"formula_pretty: {item.get('formula_pretty')}, "
                    # f"total_magnetization: {item.get('total_magnetization')}, "
                    f"sg_symbol: {item.get('sg_symbol')}, "
                    f"point_group: {item.get('point_group')}, "
                    f"crystal_system: {item.get('crystal_system')}, "
                    f"lattice_mat_2d: {structure.get('lattice').get('matrix')}, "
                    f"lattice_params: {lattice_params}, "
                    f"species: {species}, "
                    f"fractional_coords: {frac_coords}, "
                    f"lattice_volume: {structure.get('lattice', {}).get('volume')}"
                )

                result[material_id] = info_str
                bandgap_dic[material_id]  = bandgap
                cbm_dic[material_id] = cbm
                vbm_dic[material_id] = vbm

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return result, bandgap_dic,cbm_dic,vbm_dic

def load_structure_from_json(json_line):
    data = json.loads(json_line)
    structure_data = data['structure']

    lattice_data = structure_data['lattice']['matrix']
    species = [site['species'][0]['element'] for site in structure_data['sites']]
    coords = [site['abc'] for site in structure_data['sites']]

    structure = Structure(lattice_data, species, coords)
    return structure

class CrystalGraphDataset(Dataset):
    def __init__(self, json_file, graph_gen):

        self.graph_dict = {}
        self.graph_gen = graph_gen  #

        with open(json_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    material_id = data['material_id']
                    structure = load_structure_from_json(line)
                    g = self.graph_gen.get_graph_data(structure)


                    self.graph_dict[material_id] = g
                except KeyError as e:
                    print(f"Error at line {line_num}: Missing key {e}. Skipping entry.")
                except ValueError as ve:
                    print(f"Error at line {line_num}: {ve}. Skipping structure.")
                except json.JSONDecodeError as je:
                    print(f"JSON decoding error at line {line_num}: {je}")

    def __len__(self):
        return len(self.graph_dict)

    def __getitem__(self, material_id):
        if material_id in self.graph_dict:
            return self.graph_dict[material_id]
        else:
            raise KeyError(f"Graph with material_id {material_id} not found.")


database_file = './data/db.json'
output_file = './data/graph_text_2dmat.pkl'

graph_gen = GraphGen(database_file)
cgcnn_feature_path = "./atom_init.json"
graph_gen = GraphGen(cgcnn_feature_path)
graph_dic = CrystalGraphDataset(database_file, graph_gen)
string_dic, bandgap_dic, cbm_dic, vbm_dic = generate_material_info_dict(database_file)

merged_dict = {}
for key, graph in graph_dic.graph_dict.items():
    merged_dict[key] = {
        'structure_graph': graph,  
        'text_string': string_dic.get(key, None),  
        'bandgap': bandgap_dic.get(key, None),
        'cbm': cbm_dic.get(key, None),
        'vbm': vbm_dic.get(key, None)
    }


print(f"Total merged entries: {len(merged_dict)}")


with open(output_file, 'wb') as f:
    pickle.dump(merged_dict, f)

print("Dictionary successfully saved.")