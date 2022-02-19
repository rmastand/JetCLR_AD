import sys
# load standard python modules
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.file_readers import phi_wrap, pandas_to_unclustered_particles, get_highest_mass_constituents, pandas_to_features, select_jets_1, select_jets_2 
from modules.jet_visualizers import plot_jets_phase_plane, plot_nsubs
from modules.jet_augs import apply_single_jet_augs, translate_jets, rotate_jets, rescale_pts, distort_jets, collinear_fill_jets, crop_jets
from modules.jet_vars import nsub, convert_constits_coords

# choose from: "02092021", "dijet"

study_type = "final_16_02_2022"

clustered_npy_exists = False

path_to_unclustered = "/clusterfs/ml4hep/rrmastandrea/LHC0/events_anomalydetection.h5"
path_to_data_storage = "/clusterfs/ml4hep/rrmastandrea/processed_data/"+study_type+"/"

if not clustered_npy_exists:
    
    #start = 465000
    #stop = 470000
    jetR = 0.8
    j_per_e = 2
    center = "J1_phi_only_pi_2"
    ncon_store = 50
    
    delta = 10000
    
    for start in range(130000,140000,delta):
        stop = start + delta
    
    
        fname_data = "dijet_data_jetR_"+str(jetR)+"_"+center+"_"+str(start)+"_"+str(stop)+".npy"
        fname_labels = "dijet_labels_jetR_"+str(jetR)+"_"+center+"_"+str(start)+"_"+str(stop)+".npy"


        print("Reading in unclustered events...")
        # Read in the file
        unclustered_particles_data = pd.read_hdf(path_to_unclustered,start = start, stop = stop)
        # Convert pd to numpy; get labels
        unclustered_collisions, unclustered_particles_labels = pandas_to_unclustered_particles(unclustered_particles_data)
        # Cluster into jets, get highest mass constituents
        high_mass_consits_wjet, bad_indices = get_highest_mass_constituents(unclustered_collisions, jetR, center = center, j_per_e = j_per_e, ncon_store=ncon_store)
        high_mass_labels = np.delete(unclustered_particles_labels,bad_indices)
        np.save(path_to_data_storage+fname_data, high_mass_consits_wjet)
        np.save(path_to_data_storage+fname_labels, high_mass_labels)
        print("Saved file "+fname_data)
        print("Saved file "+fname_labels)
    

        print(high_mass_consits_wjet.shape,unclustered_particles_labels.shape)
        
        print()
        
       

    
