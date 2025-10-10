""" @author: Marius Glogger
Research Group Heilemann
Institute for Physical and Theoretical Chemistry, Goethe University Frankfurt a.M.
Analyzes two picasso DBSCAN files (from a two-target SMLM experiment, _dbclusters.hdf5) for cluster colocalization based
on user defined distance threshold. Saves individual, colocalizing single-molecule cluster information
(centroid coordinates, size, number of molecules etc.) as a new picasso compatible _dbcluster.hdf5. """

import h5py
import numpy as np
import matplotlib.pyplot as plt
import math
import yaml as _yaml
from numba import njit
from configparser import ConfigParser
import yaml
import os
import pandas as pd

def load_config(file_path="config.ini"):
    config = ConfigParser()
    if not config.read(file_path):
        raise FileNotFoundError(f"Configuration file '{file_path}' not found.")

    path1 = config["INPUT_FILES"].get("path1", fallback="")
    path2 = config["INPUT_FILES"].get("path2", fallback="")
    filename1 = config["INPUT_FILES"].get("filename1", fallback="")
    filename2 = config["INPUT_FILES"].get("filename2", fallback="")
    max_dist = float(config["PARAMETERS"].get("max_dist", fallback=1))
    pixelsize = int(config["PARAMETERS"].get("pixelsize", fallback=130))

    return path1, path2, filename1, filename2, max_dist, pixelsize

path1, path2, filename1, filename2, max_dist, pixelsize = load_config("config.ini")
max_dist = max_dist / pixelsize  # max_dist [px]
sf=1

class LoadHDF5(object):
    """ loads .hdf5 files from path.
        :return: lists containing individual dbscan_cluster information."""
    def __init__(self, path):  # path to data
        self.path = path

    def load(self):
        try:
            with h5py.File(self.path, "r") as locs_file:
                # Assumes the first key is the dataset name
                key = next(iter(locs_file.keys()))
                locs = locs_file[str(key)][()]  # Use [()] for cleaner NumPy array extraction
                dtype_structure = locs.dtype  # get dtype structure
                return pd.DataFrame(locs), dtype_structure
        except StopIteration:
            raise ValueError(f"HDF5 file '{self.file_path}' is empty or contains no datasets.")
        except Exception as e:
            raise IOError(f"Error reading HDF5 file '{self.file_path}': {e}")

    def load_yaml(self):
        yaml_name = self.path.replace('.hdf5', '.yaml')
        try:
            with open(yaml_name, 'r') as file:
                # Use safe_load for security and basic data loading
                yaml_data = yaml.safe_load_all(file)
                return yaml_name
        except StopIteration:
            raise ValueError(f"yaml_file '{yaml_name}' is empty or contains no datasets.")
        except Exception as e:
            raise IOError(f"Error reading yaml_file '{yaml_name}': {e}")

class ColocSave(object):
    """ Saves new .hdf5 files and corresponding .yaml file that contain all dbscan cluster information from cluster
        that colocalize within the dataset."""
    def __init__(self, data, path, dtype_structure, max_dist, yaml_file):
        self.data = data
        self.path = path
        self.dtype_structure = dtype_structure
        self.max_dist = max_dist * 130
        self.yaml_file = yaml_file


    def save_dbluster_coloc(self):
        # name suffix for saving file
        name = "cluster_coloc" + "_" + str(round(self.max_dist)) + "_nm.hdf5"

        # 1. Create the structured dtype (ds_dt) as before
        names = self.data.columns.values  # ["group", "x", "y"]
        formats = [self.dtype_structure[col] for col in names]
        ds_dt = np.dtype({'names': names, 'formats': formats})

        # 2. Pre-allocate the target NumPy structured array
        num_rows = len(self.data)
        data_array = np.zeros(num_rows, dtype=ds_dt)

        # 3. Fill the structured array one column at a time, ensuring correct casting
        # This avoids potential issues when casting the whole list-of-lists at once.
        for col_name in names:
            # Use .to_numpy() for the underlying array of values
            # and explicitly cast to the target format (e.g., '<i4')
            target_format = self.dtype_structure[col_name]

            data_array[col_name] = self.data[col_name].to_numpy().astype(target_format)

        # 4. Then save the data_array using h5py:
        with h5py.File(os.path.join(self.path, str(name)), "w") as locs_file:
            locs_file.create_dataset("locs", data=data_array)

    def save_yaml_coloc(self):

        # 2. Define the output filename and path
        output_filename = "cluster_coloc" + "_" + str(round(self.max_dist)) + "_nm.yaml"
        head, tail = os.path.split(self.yaml_file)
        name, ext = os.path.splitext(tail)
        full_output_path = os.path.join(self.path, output_filename)
        try:
            # 3. Read the input YAML file
            with open(self.yaml_file, 'r') as infile:
                # Consume the generator returned by load_all immediately into a list
                metadata_list = list(_yaml.load_all(infile, _yaml.FullLoader))
        except FileNotFoundError:
            print(f"Error: Input YAML file not found at: {self.yaml_file}. Cannot save metadata.")
            return

        # --- Saving to the New File ---
        try:
            if metadata_list:
                metadata_list[0]['cluster_coloc_max_dist_px'] = self.max_dist

            # 5. Open the NEW file path in write mode ('w') and save the data
            with open(full_output_path, 'w') as outfile:
                _yaml.dump_all(metadata_list, outfile, default_flow_style=False)

        except Exception as e:
            print(f"Error during YAML saving: {e}")


    def main(self):
        self.save_dbluster_coloc()
        self.save_yaml_coloc()


# generate instance of Load_HDF5() -> load first .hdf5 file
p1 = LoadHDF5(os.path.join(path1, filename1))
data1, dtype_structure1 = p1.load()
yaml_file1 = p1.load_yaml()
p2 = LoadHDF5(os.path.join(path2, filename2))
data2, dtype_structure2 = p2.load()
yaml_file2 = p2.load_yaml()

@njit
def cluster_dist(p1x, p1y, p2x, p2y, max_dist):
    """ calculates the Euclidean distance between cluster centroids (p1 and p2) and stores pairs in list if
        distance < user defined max_dist. Uses numba (njit) for runtime optimization
        :return: list of lists that contain all cluster with disance < max_distance"""
    return [(i, j) for i, (val_x, val_y) in enumerate(zip(p1x, p1y)) for j, (val2_x, val2_y) in
            enumerate(zip(p2x, p2y)) if math.sqrt((val2_x - val_x) ** 2 + (val2_y - val_y) ** 2) < max_dist]


index_pairs = cluster_dist(
    data1["x"].values,
    data1["y"].values,
    data2["x"].values,
    data2["y"].values,
    max_dist)


def extract_matched_rows(data1, data2, index_pairs):
    """
    Extracts rows from data1 and data2 based on a list of index pairs.

    The resulting DataFrames will have the same length and be aligned based
    on the order of the pairs in index_pairs.

    Args:
        data1 (pd.DataFrame): The first DataFrame.
        data2 (pd.DataFrame): The second DataFrame.
        index_pairs (list): A list of tuples, e.g., [(i1, j1), (i2, j2), ...],
                            where i is the index for data1 and j is for data2.

    Returns:
        tuple: A tuple containing two new, aligned DataFrames (df1_matched, df2_matched).
    """
    # 1. Separate the indices into two lists
    # Use a generator expression for efficiency
    indices_df1 = [i for i, j in index_pairs]
    indices_df2 = [j for i, j in index_pairs]

    # 2. Use .iloc to efficiently select the rows from data1
    # .iloc uses integer positions (0-based), which is what your list contains.
    df1_matched = data1.iloc[indices_df1].reset_index(drop=True)

    # 3. Use .iloc to efficiently select the rows from data2
    df2_matched = data2.iloc[indices_df2].reset_index(drop=True)

    return df1_matched, df2_matched


df1_matched, df2_matched = extract_matched_rows(data1, data2, index_pairs)  # Call function

# --- Plotting with Proper Labels and Units ---
if not df1_matched.empty:
    plt.figure(figsize=(10, 8))

    # Plot Set 1
    plt.scatter(
        df1_matched['x'],
        df1_matched['y'],
        color='green',
        label=f'Set 1 Colocalized ({len(df1_matched)} clusters)',
        alpha=0.6,
        s=10
    )

    # Plot Set 2
    plt.scatter(
        df2_matched['x'],
        df2_matched['y'],
        color='red',
        label=f'Set 2 Colocalized ({len(df2_matched)} clusters)',
        alpha=0.6,
        s=10
    )

    # --- Proper Plot Naming ---
    plt.title(f"Colocalized Clusters (Threshold: {max_dist:.1f} nm)")
    plt.xlabel("X Coordinate (pixels)")
    plt.ylabel("Y Coordinate (pixels)")
    plt.legend(loc='upper right')
    # Ensure equal axis scaling for accurate spatial representation
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

# Save coloc data p1 and p2
p1_save = ColocSave(df1_matched, path1, dtype_structure1, max_dist, yaml_file1)
p1_save.main()

p2_save = ColocSave(df2_matched, path2, dtype_structure2, max_dist, yaml_file2)
p2_save.main()

if __name__ == "__main__":
    print(__name__)


