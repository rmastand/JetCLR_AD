"""
This script reads in data from the LHC Olympics datasets (e.g. ```LHC0/events_anomalydetection.h5``` from https://zenodo.org/record/2629073#.YoPkO5PMJpQ). It clusters the jets, selects the 50 hardest constituents from each jet, and outputs them into a more compact file.

The output of this script can then be sent to ```data_prep_standard_test_set_vf.ipynb``` for data selection and processing into datasets with specific signal / background fractions

****************************************

The LHCO dataset contains particle information $p_T$, $\eta$, $\phi$. 

The code:

- cluster the particles into jets
- take the highest mass jet(s)
- return $p_T$, $\eta$, $\phi$ for the jet and the 50 hardest-$p_T$ constituents

The output dataset_shape is (n, [$p_T$, $\eta$, $\phi$] , n_jets*(1 jet + 50 constituents))
We also get a dataset of labels if an event is signal (1) or background (0)
"""

# standard imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# specific imports
from modules.file_readers import phi_wrap, pandas_to_unclustered_particles, get_highest_mass_constituents, pandas_to_features, select_jets_1, select_jets_2 

# path to LHCO data
path_to_unclustered = "/clusterfs/ml4hep/rrmastandrea/LHC0/events_anomalydetection.h5"

# path to where you want to store the clustered data
outfile_location = "final_16_02_2022"
path_to_data_storage = "/clusterfs/ml4hep/rrmastandrea/processed_data/"+outfile_location+"/"


   
jetR = 0.8 # cluster radius
j_per_e = 2 # n_jets to analyze
center = "J1_phi_only_pi_2"
ncon_store = 50



begin = 0
end = 500
delta = 100

# Read in lines from the LHCO dataset in chunks of size delta. I found it's more reliable to have multiple compactified
        # datasets of delta ~ 5000

for start in range(begin,end,delta):
    stop = start + delta

    fname_data = "dijet_data_jetR_"+str(jetR)+"_"+center+"_"+str(start)+"_"+str(stop)+".npy"
    fname_labels = "dijet_labels_jetR_"+str(jetR)+"_"+center+"_"+str(start)+"_"+str(stop)+".npy"
    
    print("Reading in lines", start, "to", stop, "of dataset ...")

    # Read in the file
    unclustered_particles_data = pd.read_hdf(path_to_unclustered,start = start, stop = stop)
    # Convert pd to numpy; get labels
    unclustered_collisions, unclustered_particles_labels = pandas_to_unclustered_particles(unclustered_particles_data)
    # Cluster into jets, get highest mass constituents
    high_mass_consits_wjet, bad_indices = get_highest_mass_constituents(unclustered_collisions, jetR, center = center, j_per_e = j_per_e, ncon_store=ncon_store)
    high_mass_labels = np.delete(unclustered_particles_labels,bad_indices)
    np.save(path_to_data_storage+fname_data, high_mass_consits_wjet)
    np.save(path_to_data_storage+fname_labels, high_mass_labels)
    print("Saved data file "+fname_data)
    print("Saved labels file "+fname_labels)


    print()
    
print("Done!")




