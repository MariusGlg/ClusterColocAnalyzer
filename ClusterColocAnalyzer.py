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
import collections
from configparser import ConfigParser
import sys
import os

#  config.ini file
config = ConfigParser()
file = "config.ini"
config.read(file)
#  config file parameter
#  maximal distance allowed for allocation of colocalization (in pixel)
max_dist = float(config["PARAMETERS"]["max_dist"])
# load _dblucster.hdf5 files
path1 = config["INPUT_FILES"]["path1"]
path2 = config["INPUT_FILES"]["path2"]
#  extract file names
filename1 = config["INPUT_FILES"]["filename1"]
filename2 = config["INPUT_FILES"]["filename2"]

# db_cluster column header
col_dbcluster = {0: "groups", 1: "convex_hull", 2: "area", 3: "mean_frame", 4: "com_x", 5: "com_y", 6: "std_frame",
                 7: "std_x", 8: "std_y", 9: "n"}


class LoadHDF5(object):
    """ loads .hdf5 files from path.
        :return: lists containing individual dbscan_cluster information."""
    def __init__(self, path):  # path to data
        self.path = path

    def load(self):
        """load .hdf5_file"""
        f = h5py.File(self.path, mode="r")  # read mode only
        a_group_key = list(f.keys())[0]  # Check what keys are inside that group
        data = list(f[a_group_key])
        return data


class ColumnData(object):
    """ Extracts information from the list of dbscan_clusters and generates new list containing these information.
        data = list containing dbscan_cluster information (coordinates, area etc., see column header (line 34)
        args = user defined list containing column indexes to be extracted (0=groups, 1=convex_hull,...)."""

    def __init__(self, data, *args):
        self.data = data
        self.args = args
        self.groups = col_dbcluster

    def col_vals(self, data, col):
        """ extracts values from columns and appends it to empty array
            :return: list containing values to be extracted."""
        vals = []
        for i in range(len(data)):
            vals.append(data[i][col_dbcluster[col]])
        return vals

    def main(self):
        """ main function -> calls func col_vals
            :return: array containing all cluster centroid coordinates."""
        cluster_center = []
        for i, k in enumerate(self.args):
            temp = self.col_vals(self.data, k)
            cluster_center.append(temp)
        return np.array(cluster_center)


class ColocSave(object):
    """ Saves new .hdf5 files and corresponding .yaml file that contain all dbscan cluster information from cluster
        that colocalize within the dataset."""
    def __init__(self, p_coloc, path):
        self.p_coloc = p_coloc
        self.path = path
    data = {"max_dist": max_dist}

    def save_dbluster_coloc(self):
        name = "dbcluster_coloc" + "_" + str(max_dist) + "_px.hdf5"
        with h5py.File(os.path.join(self.path, str(name)), "w") as locs_file:
            locs_file.create_dataset("locs", data=self.p_coloc)

    def save_yaml_coloc(self, data):
        name = "dbcluster_coloc" + "_" + str(max_dist) + "px.yaml"
        with open(os.path.join(self.path, str(name)), 'w') as outfile:
            _yaml.dump(data, outfile)

    def main(self):
        self.save_dbluster_coloc()
        self.save_yaml_coloc(self.data)


# generate instance of Load_HDF5() -> load first .hdf5 file
p1 = LoadHDF5(os.path.join(path1, filename1))
data1 = p1.load()
# param *args (e.g. 0 for cluster index, 4 and 5 for com_x and com_y (cluster centroid coordinates)
dat_p1 = ColumnData(data1, 0, 4, 5)
p1_ind_xval_yval = dat_p1.main()

# generate instance of Load_HDF5() -> load first .hdf5 file
p2 = LoadHDF5(os.path.join(path2, filename2))
data2 = p2.load()
# param *args (e.g. 0 for cluster index, 4 and 5 for com_x and com_y (cluster centroid coordinates)
dat_p2 = ColumnData(data2, 0, 4, 5)
p2_ind_xval_yval = dat_p2.main()

p1ind, p1x, p1y = p1_ind_xval_yval
p2ind, p2x, p2y = p2_ind_xval_yval

@njit
def cluster_dist(p1x, p1y, p2x, p2y):
    """ calculates the euclidean distance between cluster centroids (p1 and p2) and stores pairs in list if
        distance < user defined max_dist. Uses numba (njit) for runtime optimization
        :return: list of lists that contain all cluster with disance < max_distance"""
    return [(i, j) for i, (val_x, val_y) in enumerate(zip(p1x, p1y)) for j, (val2_x, val2_y) in
            enumerate(zip(p2x, p2y)) if math.sqrt((val2_x - val_x) ** 2 + (val2_y - val_y) ** 2) < max_dist]


ind_list = cluster_dist(p1x, p1y, p2x, p2y)


def duplicates(ind_list, p1, p2):
    """Checks dataset if multiple cluster were found within a distance < max_distance (check for duplicates in list).
        Applies a nearest neighbor based analysis to assign colocalizing clusters (shortes centroid distances), removes
        duplicates from list."""

    p1vals = [p1 for (p1, p2) in ind_list]  # get p1 index form list of tuples
    # get p1 index from list of tuples if p1 has multiple p2 partners and vice versa
    p1_dupl = [p1 for p1, count in collections.Counter(p1vals).items() if count > 1]
    del_list = []
    for i, k in enumerate(p1_dupl):
        duplicates = [item for pos, item in enumerate(ind_list) if k in item and ind_list[pos][0] == k]
        d_list = []
        for j in duplicates:
            d = math.sqrt(
                (p2[1][j[1]] - p1[1][j[0]]) ** 2 + (p2[2][j[1]] - p1[2][j[0]]) ** 2)  # calc dist between pairs
            d_list.append(d)
        min_d = np.min(d_list)  # find min dist
        temp = [j for j, k in enumerate(d_list) if k != min_d]  # get index of pairs != min
        for l in temp:
            del_list.append(duplicates[l])  # append index of elements to be deleted to del_list

    # use del_list to mask ind_list and remove duplicates
    for i, k in enumerate(del_list):
        for j, l in enumerate(ind_list):
            if k == l:
                ind_list.remove(k)  # remove elements from ind_list
    # get corresponding p2 index and arrange in list of tuples
    # correction for p2:
    p2vals = [p2 for (p1, p2) in ind_list]  # get p2 index from list of tuples
    p2_dupl = [p2 for p2, count in collections.Counter(p2vals).items() if count > 1]
    del_list = []
    for i, k in enumerate(p2_dupl):
        duplicates = [item for pos, item in enumerate(ind_list) if k in item and ind_list[pos][1] == k]
        d_list = []
        for i in duplicates:
            d = math.sqrt(
                (p2[1][i[1]] - p1[1][i[0]]) ** 2 + (p2[2][i[1]] - p1[2][i[0]]) ** 2)  # calc dist between pairs
            d_list.append(d)
        min = np.min(d_list)  # find min dist
        temp = [j for j, k in enumerate(d_list) if k != min]  # get index of pairs != min
        for i in temp:
            del_list.append(duplicates[i])  # append index of elements to be deleted to del_list
    # use del_list to mask ind_list and remove wrong duplicates
    for i, k in enumerate(del_list):
        for j, l in enumerate(ind_list):
            if k == l:
                ind_list.remove(k)  # remove elements from ind_list
    return ind_list


corrected_list = duplicates(ind_list, p1_ind_xval_yval, p2_ind_xval_yval)


p1_coloc = []
p2_coloc = []

for i, k in enumerate(corrected_list):
    for j, l in enumerate(k):
        try:
            if j == 0:
                p1_coloc.append(data1[l])
            if j == 1:
                p2_coloc.append(data2[l])
        except Exception as exception:
            print(exception)
            raise
p1_coloc.sort(key=lambda y: y[0])  # sort list of tuples based on first index value

assert len(p1_coloc) == len(p2_coloc)  # test if length is equal

# plot data and save
for i, k in enumerate(p1_coloc):
    plt.plot(k[4], k[5], "r+", markersize=2)
for i, k in enumerate(p2_coloc):
    plt.plot(k[4], k[5], "g+", markersize=2)
name = "coloc_" + str(max_dist) + "_px.pdf"
plt.xlabel("pixel_x")
plt.ylabel("pixel_y")
plt.title("p1_p2_colocalization within {} pixel".format(max_dist))
plt.savefig(os.path.join(path1, name))
plt.show()


# Save coloc data p1 and p2
p1_save = ColocSave(p1_coloc, path1)
p1_save.main()

p2_save = ColocSave(p2_coloc, path2)
p2_save.main()

if __name__ == "__main__":
    print(__name__)