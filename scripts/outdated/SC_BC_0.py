#!/usr/bin/env python
# coding: utf-8

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
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from matplotlib.backends.backend_pdf import PdfPages

# load torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# load custom modules required for jetCLR training
from modules.jet_augs import remove_jet_and_rescale_pT
from modules.transformer import Transformer
from modules.losses import contrastive_loss, align_loss, uniform_loss, contrastive_loss_num_den
from modules.perf_eval import get_perf_stats, linear_classifier_test, plot_losses
from modules.neural_net import create_and_run_nn
from modules.jet_vars import nsub, convert_constits_coords, mj
from modules.utils import LRScheduler, EarlyStopping

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


data_frac = 1


path_to_save_dir = "/global/home/users/rrmastandrea/training_data_vf/"
BC_dir = "nBC_sig_85000_nBC_bkg_85000_n_nonzero_50_n_pad_0_n_jet_2/"
TEST_dir = "STANDARD_TEST_SET_n_sig_10k_n_bkg_10k_n_nonzero_50_n_pad_0_n_jet_2/"

grading = 50
n_constits_max = 50
n_jets = 2

path_to_BC = path_to_save_dir+BC_dir
print(path_to_BC)

path_to_test = path_to_save_dir+TEST_dir
print(path_to_test)




data_train = np.load(path_to_BC+"data_train.npy")
labels_train = np.load(path_to_BC+"labels_train.npy")
data_val = np.load(path_to_BC+"data_val.npy")
labels_val = np.load(path_to_BC+"labels_val.npy")
data_test_f = np.load(path_to_test+"data.npy")
labels_test_f = np.load(path_to_test+"labels.npy")

data_train = remove_jet_and_rescale_pT(data_train, n_jets)[:int(data_frac*data_train.shape[0])]
data_val = remove_jet_and_rescale_pT(data_val, n_jets)[:int(data_frac*data_val.shape[0])]
data_test_f = remove_jet_and_rescale_pT(data_test_f, n_jets)[:int(data_frac*data_test_f.shape[0])]

labels_train = labels_train[:int(data_train.shape[0])]
labels_val = labels_val[:int(data_val.shape[0])]
labels_test_f = labels_test_f[:int(data_test_f.shape[0])]

# print data dimensions
print( "BC training data shape: " + str( data_train.shape ), flush=True)
print( "BC training labels shape: " + str( labels_train.shape ), flush=True)
print( "BC val data shape: " + str( data_val.shape ), flush=True)
print( "BC val labels shape: " + str( labels_val.shape ), flush=True)
print( "BC test data shape: " + str( data_test_f.shape ), flush=True)
print( "BC test labels shape: " + str( labels_test_f.shape ), flush=True)



# Define the Binary Classifier transformer net
model_dim  = 128

# transformer hyperparams
input_dim = 3
output_dim = model_dim
dim_feedforward = model_dim
n_heads = 4
n_layers = 2
n_head_layers = 2
opt = "adam"

mask= False
cmask = True

learning_rate_trans = 0.00001
batch_size = 400

    
expt_dir = "sanity_checks/"



netBC = Transformer( input_dim, model_dim, output_dim, n_heads, dim_feedforward, 
                  n_layers, learning_rate_trans, n_head_layers, dropout=0.1, opt=opt, BC = True )

## send network to device
netBC.to( device )

# define lr scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( netBC.optimizer, factor=0.2 )

run_BC_transformer = True

criterion = nn.BCELoss()

early_stop = True
if early_stop:
    early_stopping = EarlyStopping()
    
if run_BC_transformer:

    # THE TRAINING LOOP

    # initialise lists for storing training stats, validation loss
    losses_BC_num_jets = {i:[] for i in range(grading,n_constits_max+grading,grading)}
    loss_validation_num_jets = {i:[[],[]] for i in range(grading,n_constits_max+grading,grading)} #epoch, loss

    n_epochs = 5000
    loss_check_epoch = 10
    verbal_epoch = 10

    for constit_num in range(grading,n_constits_max+grading,grading):

        t0 = time.time()

        print( "starting training loop, running for " + str( n_epochs ) + " epochs" + " with " + str(constit_num) + " constituents" 
              , flush=True)
        print("BC training data shape:",data_train.shape)
        print("BC val data shape:",data_val.shape)
        print( "---", flush=True )

        # re-batch the data on each epoch
        for epoch in range( n_epochs + 1 ):

            # get batch_size number of indices
            indices_list = torch.split( torch.randperm( data_train.shape[0] ), batch_size )

            # initialise lists to store batch stats
            losses_BC_e = []

            # the inner loop goes through the dataset batch by batch
            # augmentations of the jets are done on the fly
            for i, indices in enumerate( indices_list ): # random jets from the dataset
                netBC.optimizer.zero_grad()
                """
                DATA PREPARATION
                """
                x_i = data_train[indices,:,:]
                labels_i = labels_train[indices]

                x_i = torch.Tensor( x_i ).transpose(1,2).to( device ) # shape (batchsize, 2, 3)
                labels_i = torch.Tensor( labels_i ).to( device )
                z_i = netBC( x_i, use_mask=mask, use_continuous_mask=cmask ) # shape (batchsize, output_dim)

                """
                LOSS CALCULATIONS
                """           

                # compute the loss based on predictions of the netBC and the correct answers
                loss = criterion( z_i, labels_i.reshape(-1,1)).to( device )
                loss.backward()
                netBC.optimizer.step()
                netBC.optimizer.zero_grad()
                
                losses_BC_e.append( loss.detach().cpu().numpy() )
                
            """
            AVERAGING OF LOSSES
            """ 
            loss_BC_e = np.mean( np.array( losses_BC_e ) )
            ## scheduler
            scheduler.step( loss_BC_e )

            # storage
            losses_BC_num_jets[constit_num].append( loss_BC_e )
           
            """
            EVERY SO OFTEN, GIVEN AN UPDATE
            """

            if epoch % verbal_epoch == 0:

                print( "epoch: " + str( epoch ) + ", loss: " + str( round(losses_BC_num_jets[constit_num][-1], 5) ), flush=True )
                print("time taken up to now: " + str(time.time()-t0))
                print()

            if epoch % loss_check_epoch == 0:

                """
                Get the validation loss
                """
                print("Getting the validation loss...")
                # store the epoch
                loss_validation_num_jets[constit_num][0].append(epoch)

                with torch.no_grad():
                    netBC.eval()

                    # get batch_size number of indices
                    indices_list_val = torch.split( torch.randperm( data_val.shape[0] ), batch_size )
                    local_val_losses = []

                    for j, indices_val in enumerate( indices_list_val ):
                        
                        """
                        DATA PREPARATION
                        """
                        a_i = data_val[indices_val,:,:]
                        labelsa_i = labels_val[indices_val]

                        a_i = torch.Tensor( a_i ).transpose(1,2).to( device ) # shape (batchsize, 2, 3)
                        labelsa_i = torch.Tensor( labelsa_i ).to( device )
                        w_i = netBC( a_i, use_mask=mask, use_continuous_mask=cmask ) # shape (batchsize, output_dim)

                        """
                        LOSS CALCULATIONS
                        """           

                        # compute the loss based on predictions of the netBC and the correct answers
                        loss_val = criterion( w_i, labelsa_i.reshape(-1,1)).to( device )
                        local_val_losses.append(loss_val.detach().cpu().numpy())
                
                    loss_val_e = np.mean( np.array( local_val_losses ) )
                    loss_validation_num_jets[constit_num][1].append(loss_val_e)
                  
                    if early_stop:
                        early_stopping(loss_val_e)
            
            if early_stopping.early_stop:
                break

        t1 = time.time()

        print( "BC TRAINING DONE, time taken: " + str( np.round( t1-t0, 2 ) ), flush=True)


        # save out results
        print( "saving out data/results", flush=True)
        np.save( expt_dir+"BC_losses_train_"+str(data_frac)+".npy", losses_BC_num_jets[constit_num] )
        np.save( expt_dir+"BC_losses_val_"+str(data_frac)+".npy", loss_validation_num_jets[constit_num] )

        # save out final trained model
        print( "saving out final BC model", flush=True )
        torch.save(netBC.state_dict(), expt_dir+"final_model_BC_"+str(data_frac)+".pt")
        print()
        

