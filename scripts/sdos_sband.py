import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_dos_and_scores_surrodings(bs_info, dos_data, omega_max, alpha=0.2, delta=10.0, mu=2.0):
    spin = bs_info.get('spin')
    flattest_energies = bs_info.get('flattest_curve_energies')
    if flattest_energies is None:
        print("No flattest curve energies available.")
        return None, None

    E_min = np.min(flattest_energies)
    E_max = np.max(flattest_energies)
    E_mid = (E_min + E_max) / 2

    E_lower = E_mid - omega_max / 2
    E_upper = E_mid + omega_max / 2
    E_prev_lower = E_mid - 3 * omega_max / 2
    E_prev_upper = E_lower
    E_next_lower = E_upper
    E_next_upper = E_mid + 3 * omega_max / 2

    energies = np.array(dos_data.get('energies'))
    dos_up = np.array(dos_data.get('dos_up')) if dos_data.get('dos_up') is not None else None
    dos_down = np.array(dos_data.get('dos_down')) if dos_data.get('dos_down') is not None else None
    Efermi = dos_data.get('efermi')

    if spin == '1' and dos_up is not None:
        dos_total = dos_up
    elif spin == '-1' and dos_down is not None:
        dos_total = dos_down
    else:
        print(f"No DOS data available for spin {spin}.")
        return None, None

    def calculate_avg_dos(energies, dos, e_lower, e_upper):
        mask = (energies >= e_lower) & (energies <= e_upper)
        if not np.any(mask):
            return 0.0
        e_range = energies[mask]
        dos_range = dos[mask]
        if len(e_range) < 2:  
            return np.mean(dos_range) if len(dos_range) > 0 else 0.0
        integral = np.trapz(dos_range, e_range)
        range_width = e_upper - e_lower
        return integral / range_width if range_width > 0 else 0.0

    dos_avg = calculate_avg_dos(energies, dos_total, E_lower, E_upper)
    dos_prev = calculate_avg_dos(energies, dos_total, E_prev_lower, E_prev_upper)
    dos_next = calculate_avg_dos(energies, dos_total, E_next_lower, E_next_upper)

    if dos_avg == 0:
        print("Zero DOS in target energy range.")
        return None, None

    dos_surrounding = (dos_prev + dos_next) / 2 if (dos_prev + dos_next) > 0 else 1e-6

    peak_contrast = dos_avg / dos_surrounding if dos_surrounding > 0 else 1e6

    S_dos = peak_contrast / (1 + peak_contrast)  # Range [0, 1]

    return S_dos, peak_contrast


def calculate_dos_and_scores(bs_info, dos_data, omega_max, alpha=0.2, delta=10.0, mu=2.0):
    flattest_energies = bs_info.get('flattest_curve_energies')
    if flattest_energies is None:
        print("No flattest curve energies available.")
        return None, None, None, None

    E_min = np.min(flattest_energies)
    E_max = np.max(flattest_energies)
    E_mid = (E_min + E_max) / 2

    E_lower = E_mid - omega_max / 2
    E_upper = E_mid + omega_max / 2

    energies = np.array(dos_data['energies'])
    dos_up = np.array(dos_data['dos_up']) if dos_data['dos_up'] is not None else None
    dos_down = np.array(dos_data['dos_down']) if dos_data['dos_down'] is not None else None
    Efermi = dos_data['efermi']

    spin = bs_info.get('spin')
    if spin == '1' and dos_up is not None:
        dos_total = dos_up
    elif spin == '-1' and dos_down is not None:
        dos_total = dos_down
    else:
        print("No DOS data available for either spin channel.")
        return None, None

    mask = (energies >= E_lower) & (energies <= E_upper)
    if not np.any(mask):
        print("No DOS data within the target energy range.")
        return None, None
    energies_in_range = energies[mask]
    dos_in_range = dos_total[mask]
    num_points = len(energies_in_range)
    if num_points < 1:
        print("Insufficient DOS data points in the target energy range.")
        return None, None
    dos_avg = np.sum(dos_in_range) / num_points

    # Calculate DOS_avg for the -5，5 range
    mask_all = (energies >= int(-5)) & (energies <= int(5))
    dos_all = 0.0
    if np.any(mask_all):
        energies_all = energies[mask_all]
        dos_all_range = dos_total[mask_all]
        dos_all = np.sum(dos_all_range) / len(energies_all) if len(energies_all) > 0 else 0.0

    peak_contrast = dos_avg / dos_all

    S_dos = peak_contrast / (1 + peak_contrast)  # Range [0, 1]

    return S_dos,peak_contrast  

def calculate_s_bandwidth(bs_info, omega_max):
    bandwidth = bs_info.get('flattest_curve_bandwidth', float('inf'))
    if bandwidth >= omega_max:
        S_bandwidth = int(-1)
    else:
        S_bandwidth = 0.5 * (np.cos(np.pi * bandwidth / omega_max) + 1)
    return S_bandwidth

def calculate_scores_from_saved_data(bs_file, dos_file, omega_max, alpha=0.2, delta=10.0, mu=2.0):

    with open(bs_file, 'rb') as f:
        bs_label = pickle.load(f)

    with open(dos_file, 'rb') as f:
        dos_dic = pickle.load(f)

    common_keys = set(bs_label.keys()) & set(dos_dic.keys())

    scores_dict = {}

    for key in common_keys:
        print(f"Processing {key}...")
        bs_info = bs_label.get(key)
        dos_data = dos_dic.get(key, {'efermi': 0.0})  # Fallback if DOS data is missing

        S_bandwidth = calculate_s_bandwidth(bs_info, omega_max)

        S_dos,peak_contrast = calculate_dos_and_scores(bs_info, dos_data, omega_max, alpha, delta, mu)

        scores_dict[key] = {
            'S_bandwidth': S_bandwidth,
            'S_DOS': S_dos,
            'peak_contrast': peak_contrast,
            'spin': bs_info['spin'],
        }

        print(f"\nResults for key: {key}")
        print(f"Spin: {bs_info['spin']}")
        print(f"Efermi: {bs_info['efermi']:.4f} eV")
        print(f"S_bandwidth: {S_bandwidth:.4f}")
        if S_dos is not None:
            print(f"S_DOS: {S_dos:.4f}")
        else:
            print("S_DOS: Not available")

    return scores_dict

def save_scores(scores_dict, output_file):
    with open(output_file, 'w') as f:
        f.write("Key\tS_bandwidth\tpeak_contrast\tS_DOS\tSpin\n")
        for key, scores in scores_dict.items():
            f.write(f"{key}\t{scores['S_bandwidth']:.4f}\t"
                    f"{scores['peak_contrast'] if scores['peak_contrast'] is not None else 'N/A'}\t"
                    f"{scores['S_DOS'] if scores['S_DOS'] is not None else 'N/A'}\t{scores['spin']}\n")


if __name__ == "__main__":
    bs_file = "./data/bs_file.pkl"
    dos_file = "./data/dos_file.pkl"
    output_file = "./data/scores.txt"

    scores_dict = calculate_scores_from_saved_data(bs_file, dos_file, omega_max=0.5)

    save_scores(scores_dict, output_file)

    print(f"\nScores saved to {output_file}")