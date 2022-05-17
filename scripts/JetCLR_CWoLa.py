#!/usr/bin/env python
# coding: utf-8

"""

This script runs a weakly supervised CWoLa training procedure on the latent space representations. 

"""

# imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load custom modules 
from modules.jet_augs import remove_jet_and_rescale_pT
from modules.transformer import Transformer
from modules.neural_net import create_and_run_nn
from modules.CWoLa_helpers import generate_mixed_sample, generate_train_test_val

seed = 1
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.empty_cache()

from numba import cuda 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
device = cuda.get_current_device()
device.reset()

torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "device: " + str( device ), flush=True)


"""

Load in the data and crop

"""

path_to_save_dir = "/global/home/users/rrmastandrea/training_data_vf/"

sig_samp_id = "nCWoLa_sig_85000_nCWoLa_bkg_0_n_nonzero_50_n_pad_0_n_jet_2/"
bkg_samp_id = "nCWoLa_sig_0_nCWoLa_bkg_85000_n_nonzero_50_n_pad_0_n_jet_2/"

TEST_dir = "STANDARD_TEST_SET_n_sig_10k_n_bkg_10k_n_nonzero_50_n_pad_0_n_jet_2/"

n_constits_max = 50
n_jets = 2

path_to_sig_data = path_to_save_dir+sig_samp_id
print(path_to_sig_data)
path_to_bkg_data = path_to_save_dir+bkg_samp_id
print(path_to_bkg_data)
path_to_STS = path_to_save_dir+TEST_dir
print(path_to_STS)


sig_data = np.load(path_to_sig_data+"data_train.npy")
sig_labels = np.load(path_to_sig_data+"labels_train.npy")
bkg_data = np.load(path_to_bkg_data+"data_train.npy")
bkg_labels = np.load(path_to_bkg_data+"labels_train.npy")

STS_data = np.load(path_to_STS+"data.npy")
STS_labels = np.load(path_to_STS+"labels.npy")

# Crop the data, rescale pt
cropped_sig_data = remove_jet_and_rescale_pT(sig_data, n_jets)
cropped_bkg_data = remove_jet_and_rescale_pT(bkg_data, n_jets)
cropped_STS_data = remove_jet_and_rescale_pT(STS_data, n_jets)

# print data dimensions
print( "Sig data shape: " + str( cropped_sig_data.shape ), flush=True)
print( "Sig labels shape: " + str( sig_labels.shape ), flush=True)
print( "Bkg data shape: " + str( cropped_bkg_data.shape ), flush=True)
print( "Sig data shape: " + str( bkg_labels.shape ), flush=True)

print( "STS data shape: " + str( cropped_STS_data.shape ), flush=True)
print( "STS labels shape: " + str( STS_labels.shape ), flush=True)


# For the actual anomaly detection, we need to split the datasets in half
# i.e. the first "mixed" sample is the first half of the background events
# the second mixed sample is a blend of the second half of the background events

first_half_sig_data = cropped_sig_data[:42500]
second_half_sig_data = cropped_sig_data[42500:]

first_half_bkg_data = cropped_bkg_data[:42500]
second_half_bkg_data = cropped_bkg_data[42500:]


"""

Load in the transformer net

"""

model_dim = 48
# location of the saved transformer net
exp_id = "SB_ratios_22_04_10/0kS_50kB_dim_"+str(model_dim)+"_seed_"+str(seed)+"/"


# set up results directory
base_dir = "/global/home/users/rrmastandrea/MJetCLR/"  # change this to your working directory
expt_dir = base_dir + "projects/rep_learning/experiments/" + exp_id + "/"

# transformer hyperparams
# input dim to the transformer -> (pt,eta,phi)
input_dim = 3
output_dim = model_dim
dim_feedforward = model_dim
n_heads = 4
n_layers = 2
n_head_layers = 2
opt = "adam"

mask = False
cmask = True

learning_rate_trans = 0.0001

# Loading in the final transformer

loaded_net = Transformer( input_dim, model_dim, output_dim, n_heads, dim_feedforward, 
                  n_layers, learning_rate_trans, n_head_layers, dropout=0.1, opt=opt )
loaded_net.load_state_dict(torch.load(expt_dir+"final_model_"+str(n_constits_max)+".pt"))
loaded_net.to( device )
loaded_net.eval()


print("Loading data into net...")

first_half_sig_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( first_half_sig_data ).transpose(1,2), first_half_sig_data.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
second_half_sig_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( second_half_sig_data ).transpose(1,2), second_half_sig_data.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()

first_half_bkg_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( first_half_bkg_data ).transpose(1,2), first_half_bkg_data.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
second_half_bkg_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( second_half_bkg_data ).transpose(1,2), second_half_bkg_data.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()


STS_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( cropped_STS_data ).transpose(1,2), cropped_bkg_data.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()

print("Data loaded!")


"""

Test the classifier in the fully supervised case

"""


# Make the datasets

M1 = first_half_sig_reps
M2 = first_half_bkg_reps

layer = 0 # here, the layers correspond to the transformer output
    
S = np.reshape(M1[:,layer,:], (M1.shape[0]*M1.shape[2],1))
B = np.reshape(M2[:,layer,:], (M2.shape[0]*M2.shape[2],1))

data_train, labels_train, data_val, labels_val = generate_train_test_val(M1, M2)

# print data dimensions
print( "Train data shape: " + str( data_train.shape ), flush=True)
print( "Train labels shape: " + str( labels_train.shape ), flush=True)
print( "Val data shape: " + str( data_val.shape ), flush=True)
print( "Val labels shape: " + str( labels_val.shape ), flush=True)

# Define the NN parameters

full_sup_AUC = {i:0 for i in range(3)}
full_sup_maxsic = {i:0 for i in range(3)}
fixed_TPR = 0.5
full_sup_FPRatTPR = {i:0 for i in range(3)}

input_shape = model_dim   
num_epochs = 1000
batch_size = 500
update_epochs = 10
lr = 0.0001

# Run the NN

for trait in range(first_half_sig_reps.shape[1]): # going through the layers of the transformer
    
    print("On layer", trait)
    

    performance_stats = create_and_run_nn(device, input_shape, num_epochs, batch_size, update_epochs, lr, 
                          data_train[:,trait,:], labels_train, 
                          data_val[:,trait,:], labels_val,
                          STS_reps[:,trait,:], STS_labels, 
                          verbose = True, early_stop = True, LRschedule = False)
        

    # Plot the output losses   
    plt.figure()
    plt.plot(performance_stats["epochs"],performance_stats["losses"], label = "loss")
    plt.plot(performance_stats["val_epochs"],performance_stats["val_losses"], label = "val loss")
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.yscale("log")
    plt.legend()
    plt.title(trait)
    plt.show()

    plt.figure()
    plt.plot(performance_stats["tpr"], 1.0/performance_stats["fpr"])
    plt.yscale("log")
    plt.xlabel("True Positive Rate")
    plt.ylabel("1/(False Positive Rate)")
    plt.title(trait)
    plt.show()

    print("Accuracy of the network: %d %%" % (100.00 *performance_stats["acc"]))
    print("ROC AUC:", performance_stats["auc"])
    
    full_sup_AUC[trait] = performance_stats["auc"]
    
    SIC = performance_stats["tpr"]/np.sqrt(performance_stats["fpr"])
    finite_SIC = SIC[np.isfinite(SIC)]
    full_sup_maxsic[trait] = np.max(finite_SIC)
    
    dist_from_fixed_tpr = np.abs(performance_stats["tpr"] - fixed_TPR)
    min_dist_ind = np.where(dist_from_fixed_tpr == np.min(dist_from_fixed_tpr))[0][0]
    full_sup_FPRatTPR[trait] = np.sqrt(performance_stats["fpr"][min_dist_ind])
    
    print(performance_stats["tpr"][min_dist_ind],performance_stats["fpr"][min_dist_ind])
    
    
"""

Test the classifier in the weakly supervised case

"""


# Define the NN parameters

input_shape = model_dim
num_epochs = 1000
batch_size = 500
update_epochs = 10
lr = 0.0001

visualize = False

f1_vals = []
ROC_AUC_vals = {i:[] for i in range(3)}
maxsic_vals = {i:[] for i in range(3)}
FPRatTPR_vals = {i:[] for i in range(3)}

f1_to_probe = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1]

for f1 in f1_to_probe:
    
    print("Starting CWoLa training run with f1 =", f1)
    f1_vals.append(f1)
    
    if f1 == 0: # weird bug unless this is manually dne
        data_train, labels_train, data_val, labels_val = generate_train_test_val(second_half_bkg_reps, first_half_bkg_reps)
    elif f1 == 1:
        data_train, labels_train, data_val, labels_val = generate_train_test_val(second_half_sig_reps, first_half_bkg_reps)
    else:
        # make the datasets / mixed samples
        
        # We only want to keep M1 here, as that's the set with a fraction f of the anomaly 
        M1, M2 = generate_mixed_sample(second_half_sig_reps, first_half_bkg_reps, f1)
    
        # The other "mixed" sample is background only
        data_train, labels_train, data_val, labels_val = generate_train_test_val(M1, first_half_bkg_reps)
        

    # print data dimensions
    print( "Train data shape: " + str( data_train.shape ), flush=True)
    print( "Train labels shape: " + str( labels_train.shape ), flush=True)
    print( "Val data shape: " + str( data_val.shape ), flush=True)
    print( "Val labels shape: " + str( labels_val.shape ), flush=True)

    # Run the NN

    for trait in range(first_half_sig_reps.shape[1]): # going through the layers of the transformer

        print("On layer", trait)
        
        # visualise the mixed samples
        
        if visualize:

            S = np.reshape(M1[:,trait,:], (M1.shape[0]*M1.shape[2],1))
            B = np.reshape(M2[:,trait,:], (M2.shape[0]*M2.shape[2],1))

            bins = np.linspace(-.5,.4,60)
            plt.figure()
            plt.hist(S, bins, label = "S", alpha = 0.5)
            plt.hist(B, bins, label = "B", alpha = 0.5)
            plt.legend()
            plt.title("Transformer layer "+str(trait))
            plt.show()

        performance_stats = create_and_run_nn(device, input_shape, num_epochs, batch_size, update_epochs, lr, 
                              data_train[:,trait,:], labels_train, 
                              data_val[:,trait,:], labels_val,
                              STS_reps[:,trait,:], STS_labels, 
                              verbose = True, early_stop = True, LRschedule = False)

        if visualize:
            # Plot the output losses   
            plt.figure()
            plt.plot(performance_stats["epochs"],performance_stats["losses"], label = "loss")
            plt.plot(performance_stats["val_epochs"],performance_stats["val_losses"], label = "val loss")
            plt.xlabel("Epochs")
            plt.ylabel("Losses")
            plt.yscale("log")
            plt.legend()
            plt.title(trait)
            plt.show()

            plt.figure()
            plt.plot(performance_stats["tpr"], 1.0/performance_stats["fpr"])
            plt.yscale("log")
            plt.xlabel("True Positive Rate")
            plt.ylabel("1/(False Positive Rate)")
            plt.title(trait)
            plt.show()

        print("Accuracy of the network: %d %%" % (100.00 *performance_stats["acc"]))
        print("ROC AUC:", performance_stats["auc"])
        
        ROC_AUC_vals[trait].append(performance_stats["auc"])
        
        SIC = performance_stats["tpr"]/np.sqrt(performance_stats["fpr"])
        finite_SIC = SIC[np.isfinite(SIC)]
        maxsic_vals[trait].append(np.max(finite_SIC))

        dist_from_fixed_tpr = np.abs(performance_stats["tpr"] - fixed_TPR)
        min_dist_ind = np.where(dist_from_fixed_tpr == np.min(dist_from_fixed_tpr))[0][0]
        FPRatTPR_vals[trait].append(np.sqrt(performance_stats["fpr"][min_dist_ind]))
    
        print()

        

"""

Save the data 

"""


cwola_npy_save_dict = "CWoLa_results_npy/dim_"+str(model_dim)+"/"

# save the f1 vals scanned over
np.save(cwola_npy_save_dict+"f1_vals"+"_seed"+str(seed), f1_vals)


for layer in range(3):
    
    # save the full sup 
    np.save(cwola_npy_save_dict+"full_sup_AUC_layer"+str(layer)+"_seed"+str(seed), np.full(len(f1_vals), full_sup_AUC[layer]))
    np.save(cwola_npy_save_dict+"full_sup_maxsic"+str(layer)+"_seed"+str(seed), np.full(len(f1_vals), full_sup_maxsic[layer]))
    np.save(cwola_npy_save_dict+"full_sup_FPRatTPR"+str(layer)+"_seed"+str(seed), np.full(len(f1_vals), full_sup_FPRatTPR[layer]))
    
    # save the Cwola
    np.save(cwola_npy_save_dict+"CWoLa_AUC_layer"+str(layer)+"_seed"+str(seed), [max(x, 1.0-x) for x in ROC_AUC_vals[layer]])
    np.save(cwola_npy_save_dict+"CWoLa_sup_maxsic"+str(layer)+"_seed"+str(seed), maxsic_vals[layer])
    np.save(cwola_npy_save_dict+"CWoLa_sup_FPRatTPR"+str(layer)+"_seed"+str(seed), FPRatTPR_vals[layer])
    
