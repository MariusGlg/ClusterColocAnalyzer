# ClusterColocAnalyzer
The script loads two picasso cluster files (e.g. _dbcluster, cluster centers) and calculates the euclidean
distance between all cluster centroids. If the distance is below a user defined threshold, it is defined
as colocalization and the cluster information (x, y position, cluster size etc.) are extracted 
from the dataset and saved seperately. The script intends to detect rare colocalization within two-target
SMLM experiments and allows the seperate analysis of these clusters. 
- Loads HDF5 files (Picasso cluster files (dbcluster.hdf5))
- calculate the euclidean distance between cluster centroids
- user defined threshold defines colocalization between clusters
- information from clusters that colocalize are saved seperately (as .hdf5) for further analysis 

Requirements: python 3.7, os, configparser, h5py, numpy, matplotlib, yaml, numba, math, collections, sys, pandas

Input file: Picasso[1] hdf5 (picasso dbscan file)

Execution: ClusterColocAnalyzer.py

Config file: 

[INPUT_FILES]
path1/2: path to picasso dbscan files 
filename: name of picasso dbscan file (name.hdf5)
max_dist: maximum distance allowed for colocalization between cluster centers

[PARAMETERS]
max_dist:  maximal distance allowed for allocation of colocalization (in pixel)

links: 
[1] https://github.com/jungmannlab/picasso

