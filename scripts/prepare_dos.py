import os
import json
import numpy as np
from pymatgen.electronic_structure.dos import Dos, Spin
from monty.json import MontyDecoder
import pickle

def load_dos_files(directory):

    dos_dic = {}

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            filepath = os.path.join(directory, filename)
            try:

                with open(filepath, 'r') as f:
                    data = json.load(f, cls=MontyDecoder)

                if isinstance(data, Dos):
                    dos_object = data
                elif isinstance(data, dict) and "densities" in data and "@class" in data and data["@class"] == "Dos":
                    densities = data["densities"]
                    energies = data.get("energies")
                    if energies is None:
                        dos_1_len = len(densities.get("1", []))
                        dos_minus1_len = len(densities.get("-1", []))
                        max_len = max(dos_1_len, dos_minus1_len)
                        if max_len > 0:
                            energies = np.linspace(-14.0328, 0.0768, max_len)
                            print(f"No 'energies' found in {filename}, using default grid with {max_len} points")
                        else:
                            raise ValueError("No valid DOS data found")
                    efermi = data.get("efermi", 0.0)
                    dos_object = Dos(efermi=efermi, energies=energies, densities=densities)
                else:
                    print(f"Skipping {filename}: Not a valid Dos object")
                    continue

                dos_data = {
                    "energies": dos_object.energies.tolist(),  
                    "efermi": dos_object.efermi
                }

                if Spin.up in dos_object.densities:
                    dos_up = dos_object.densities[Spin.up]
                    if len(dos_object.energies) != len(dos_up):
                        print(
                            f"Warning: Length mismatch in {filename} - energies: {len(dos_object.energies)}, dos_up: {len(dos_up)}")
                        min_len = min(len(dos_object.energies), len(dos_up))
                        dos_data["energies"] = dos_object.energies[:min_len].tolist()
                        dos_data["dos_up"] = dos_up[:min_len].tolist()
                    else:
                        dos_data["dos_up"] = dos_up.tolist()
                else:
                    dos_data["dos_up"] = None

                if Spin.down in dos_object.densities:
                    dos_down = dos_object.densities[Spin.down]
                    if len(dos_object.energies) != len(dos_down):
                        print(
                            f"Warning: Length mismatch in {filename} - energies: {len(dos_object.energies)}, dos_down: {len(dos_down)}")
                        min_len = min(len(dos_object.energies), len(dos_down))
                        dos_data["energies"] = dos_object.energies[:min_len].tolist()
                        dos_data["dos_down"] = dos_down[:min_len].tolist()
                    else:
                        dos_data["dos_down"] = dos_down.tolist()
                else:
                    dos_data["dos_down"] = None

                key = os.path.splitext(filename)[0]
                dos_dic[key] = dos_data

                print(f"Loaded: {filename}")

            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

    return dos_dic


if __name__ == "__main__":
    directory = "./dos" # get from 2dmatpedia database upon request
    dos_dic = load_dos_files(directory)
    output_file = './data/dos_file.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(dos_dic, f)
