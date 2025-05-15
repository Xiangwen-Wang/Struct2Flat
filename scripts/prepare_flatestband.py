import os
import json
import numpy as np
from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.symmetry.bandstructure import HighSymmKpath
from monty.json import MontyDecoder
import pickle

def preprocess_band_structure(bandstructure_dict, bands_ef, num_bands, if_spin_polarized=False):
    bands_dict = bandstructure_dict["bands"]
    Efermi = float(bands_ef)
    selected_bands_dict = {}
    original_indices_dict = {}  # Store original band indices
    spins = ['1', '-1'] if if_spin_polarized else ['1']
    for spin in spins:
        if spin not in bands_dict:
            continue
        bands = np.array(bands_dict[spin])
        avg_energies = np.mean(bands, axis=1)
        avg_distances = np.abs(avg_energies - Efermi)
        sorted_indices = np.argsort(avg_distances)
        num_bands_total = min(num_bands, len(sorted_indices))
        selected_indices = sorted_indices[:num_bands_total].tolist()
        selected_bands = bands[selected_indices, :]
        selected_bands_dict[spin] = selected_bands
        original_indices_dict[spin] = selected_indices  # Map selected bands to original indices
    return selected_bands_dict, Efermi, original_indices_dict

def load_and_process_bandstructure(file_path, num_bands=6):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    bands_py = BandStructureSymmLine.from_dict(data)
    if_spin_polarized = bands_py.is_spin_polarized
    try:
        continuous_bands = HighSymmKpath.get_continuous_path(bands_py)
        bandstructure_dict = continuous_bands.as_dict()
        kp_original = np.array([kp.cart_coords for kp in continuous_bands.kpoints])
    except:
        bandstructure_dict = bands_py.as_dict()
        kp_original = np.array([kp.cart_coords for kp in bands_py.kpoints])
    bands_ef = bands_py.efermi
    eigenv, efermi, original_indices = preprocess_band_structure(
        bandstructure_dict, bands_ef, num_bands, if_spin_polarized=if_spin_polarized
    )
    return kp_original, eigenv, efermi, if_spin_polarized, original_indices

def find_intersections_and_segments(bands, original_indices, epsilon=0.015):
    num_bands, num_kpoints = bands.shape
    potential_intersections = []
    for x in range(num_kpoints):
        energies = bands[:, x]
        for i in range(num_bands):
            for j in range(i + 1, num_bands):
                if abs(energies[i] - energies[j]) < epsilon:
                    if (x, i, j) not in potential_intersections:
                        potential_intersections.append((x, i, j))
                if x > 0:
                    prev_energies = bands[:, x - 1]
                    if abs(energies[i] - prev_energies[j]) < epsilon:
                        if (x, i, j) not in potential_intersections:
                            potential_intersections.append((x, i, j))
                if x < num_kpoints - 1:
                    next_energies = bands[:, x + 1]
                    if abs(energies[i] - next_energies[j]) < epsilon:
                        if (x, i, j) not in potential_intersections:
                            potential_intersections.append((x, i, j))
    print(f"Number of potential intersections: {len(potential_intersections)}")
    potential_intersections = list(dict.fromkeys(potential_intersections))

    potential_intersections.sort(key=lambda x: (x[1], x[2], x[0]))
    merged_intersections = []

    i = 0
    while i < len(potential_intersections):
        start_x, band_i, band_j = potential_intersections[i]
        current_x = start_x
        current_band_i = band_i
        current_band_j = band_j

        j = i + 1
        while j < len(potential_intersections):
            next_x, next_band_i, next_band_j = potential_intersections[j]
            if (next_band_i != current_band_i or next_band_j != current_band_j) or (next_x != current_x + 1):
                break
            current_x = next_x
            j += 1

        merged_intersections.append((start_x, current_band_i, current_band_j))
        if j > i + 1:
            merged_intersections.append((current_x, current_band_i, current_band_j))

        i = j

    merged_intersections.sort(key=lambda x: x[0])

    band_segments = [[] for _ in range(num_bands)]
    intersection_points = {}
    for x, band_i, band_j in merged_intersections:
        if band_i not in intersection_points:
            intersection_points[band_i] = set()
        if band_j not in intersection_points:
            intersection_points[band_j] = set()
        intersection_points[band_i].add(x)
        intersection_points[band_j].add(x)
    for band_idx in range(num_bands):
        original_band_idx = original_indices[band_idx]
        if band_idx not in intersection_points:
            band_segments[band_idx].append((0, num_kpoints, original_band_idx, band_idx + 1))
        else:
            points = sorted(list(intersection_points[band_idx]))
            if not points:
                band_segments[band_idx].append((0, num_kpoints, original_band_idx, band_idx + 1))
            else:
                start = 0
                for x in points:
                    if start < x:
                        band_segments[band_idx].append((start, x, original_band_idx, band_idx + 1))
                    start = x
                if start < num_kpoints:
                    band_segments[band_idx].append((start, num_kpoints, original_band_idx, band_idx + 1))
    print(f"Number of intersections after merging: {len(merged_intersections)}")
    return band_segments, merged_intersections, potential_intersections

def calculate_flatness_score(kpoints, bands, original_indices, omega, epsilon=0.015):
    num_bands, num_kpoints = bands.shape
    if num_kpoints != kpoints.shape[0]:
        raise ValueError(f"Mismatch in kpoints ({kpoints.shape[0]}) and bands ({num_kpoints}) dimensions")
    band_segments, intersections, potential_intersections = find_intersections_and_segments(bands, original_indices, epsilon)
    curves = []
    intersections_by_x = {}
    for x, band_i, band_j in intersections:
        if x not in intersections_by_x:
            intersections_by_x[x] = []
        intersections_by_x[x].append((x, band_i, band_j))
    for start_band in range(num_bands):
        segments = band_segments[start_band]
        if not segments or segments[0][0] != 0:
            continue
        current_seg = segments[0]
        current_x = current_seg[1]
        current_curve = [(start_band, current_seg)]
        if current_x == num_kpoints:
            curves.append(current_curve)
            continue

        def extend_curve(curve, x):
            if x >= num_kpoints:
                if curve[-1][1][1] == num_kpoints:
                    curves.append(curve)
                return
            current_band = curve[-1][0]
            possible_bands = {current_band}
            if x in intersections_by_x:
                for _, band_i, band_j in intersections_by_x[x]:
                    if band_i == current_band:
                        possible_bands.add(band_j)
                    if band_j == current_band:
                        possible_bands.add(band_i)
            possible_segments = []
            for band_idx in possible_bands:
                for seg in band_segments[band_idx]:
                    if seg[0] == x:
                        possible_segments.append((band_idx, seg))
            for band_idx, seg in possible_segments:
                new_curve = curve + [(band_idx, seg)]
                extend_curve(new_curve, seg[1])

        extend_curve(current_curve, current_x)
    print(f"Number of possible curves: {len(curves)}")
    min_bandwidth = float('inf')
    flattest_curve = None
    best_curve_segments = None
    for curve in curves:
        energies = np.zeros(num_kpoints)
        for band_idx, (start, end, orig_idx, sel_idx) in curve:
            energies[start:end] = bands[band_idx, start:end]
        bandwidth = np.max(energies) - np.min(energies)
        if bandwidth < min_bandwidth:
            min_bandwidth = bandwidth
            flattest_curve = energies
            best_curve_segments = curve
    e_mid = (np.max(flattest_curve) + np.min(flattest_curve)) / 2 if flattest_curve is not None else None

    # Format best_curve_segments with original and selected indices
    formatted_best_curve_segments = [
        (f"Original Band {orig_idx + 1}", f"Selected Band {sel_idx}", (start, end))
        for band_idx, (start, end, orig_idx, sel_idx) in best_curve_segments
    ]

    return {
        "number_of_potential_intersections": potential_intersections,
        "number_of_intersections_after_merging": intersections,
        "number_of_possible_curves": len(curves),
        "intersections": intersections,
        "band_segments": band_segments,
        "best_curve_segments": formatted_best_curve_segments,
        "flattest_curve_bandwidth": min_bandwidth,
        "flattest_curve_e_mid": e_mid,
        "flattest_curve_energies": flattest_curve.tolist() if flattest_curve is not None else None
    }

def save_bs_label(bs_label, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(bs_label, f)

def load_bandstructure_files(directory, output_file,  omega_max=0.1):
    # Load existing data if output file exists
    bs_label = {}
    if os.path.exists(output_file):
        with open(output_file, 'rb') as f:
            bs_label = pickle.load(f)
        print(f"Loaded existing bs_label with {len(bs_label)} entries from {output_file}")


    processed_keys = set(bs_label.keys())
    t = 0
    for filename in os.listdir(directory):
        t += 1
        print(f"Processing file {t}: {filename}")
        if filename.endswith('.json'):
            key = os.path.splitext(filename)[0]
            if key in processed_keys:
                print(f"Skipping {filename}, already processed.")
                continue

            print(f"Preprocessing {filename}.")
            filepath = os.path.join(directory, filename)

            try:
                num_bands = 6
                kp_original, eigenv, efermi, if_spin_polarized, original_indices = load_and_process_bandstructure(
                    filepath, num_bands=num_bands
                )
                spins = ['1', '-1']
                info = {}  
                for spin_t in spins:
                    if spin_t not in eigenv:
                        print(f"Spin {spin_t} not found in bands_dict for {filename}")
                        continue
                    bands = eigenv[spin_t]
                    orig_indices = original_indices[spin_t]
                    info[spin_t] = calculate_flatness_score(
                        kp_original, bands, orig_indices, omega=omega_max, epsilon=0.015
                    )

                if if_spin_polarized and '1' in info and '-1' in info:
                    if info['1']["flattest_curve_bandwidth"] < info['-1']["flattest_curve_bandwidth"]:
                        band_info = info['1']
                        spin = '1'
                    else:
                        band_info = info['-1']
                        spin = '-1'

                band_info["spin"] = spin
                band_info["efermi"] = efermi

                bs_label[key] = band_info

                
                save_bs_label(bs_label, output_file)
                print(f"Loaded and saved: {filename}")
            except FileNotFoundError:
                print(f"File {filepath} not found.")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")

    return bs_label


if __name__ == "__main__":
    directory = "./bands/" # get from 2dmatpedia database upon request
    output_file = './bs_file.pkl'
    bs_label = load_bandstructure_files(directory, output_file)
