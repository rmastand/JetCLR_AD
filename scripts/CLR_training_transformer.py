#!/usr/bin/env python
# coding: utf-8

"""

This script trains a transformer network to map events into a latent space using the contrastive loss. It then trains a BC to discriminate signal from background on the latent space reps.

Things you can modify:

- the latent space dimension model_dim
- the transformer architecture (look for the heading "Define the transformer net")

"""

# imports
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
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
from modules.jet_augs import apply_single_jet_augs, translate_jets, rotate_jets, rescale_pts, distort_jets, collinear_fill_jets, crop_jets, remove_jet_and_rescale_pT
from modules.jet_augs import shift_eta, shift_phi
from modules.transformer import Transformer
from modules.losses import contrastive_loss, align_loss, uniform_loss, contrastive_loss_num_den
from modules.perf_eval import get_perf_stats, linear_classifier_test, plot_losses
from modules.neural_net import create_and_run_nn
from modules.utils import LRScheduler, EarlyStopping


# RUN PARMETERS
seed = 5
model_dim = 128

# torch params + computing settings
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.cuda.empty_cache()

from numba import cuda 
os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = cuda.get_current_device()
device.reset()

# set the number of threads that pytorch will use
torch.set_num_threads(2)

# set gpu device
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print( "device: " + str( device ), flush=True)


"""

Directory paths

"""

# path to the output directory
exp_id = "subtract_colsplit/50kS_50kB_dim_"+str(model_dim)+"_seed_"+str(seed)+"/"

# set up results directory
base_dir = "/global/home/users/rrmastandrea/MJetCLR/"  # change this to your working directory
expt_dir = base_dir + "projects/rep_learning/experiments/" + exp_id + "/"

#check if experiment alreadyexists
if os.path.isdir(expt_dir):
    print("ERROR: experiment already exists, don't want to overwrite it by mistake")
    pass
else:
    os.makedirs(expt_dir)

print("experiment: "+str(exp_id) , flush=True)


"""

Load in the processed data

"""

path_to_save_dir = "/global/home/users/rrmastandrea/MJetCLR/training_data_vf/"
CLR_dir = "nCLR_sig_50000_nCLR_bkg_50000_n_nonzero_50_n_pad_0_n_jet_2/"
BC_dir = "nBC_sig_85000_nBC_bkg_85000_n_nonzero_50_n_pad_0_n_jet_2/"
TEST_dir = "STANDARD_TEST_SET_n_sig_10k_n_bkg_10k_n_nonzero_50_n_pad_0_n_jet_2/"

n_constits_max = 50
n_jets = 2

path_to_CLR = path_to_save_dir+CLR_dir
print(path_to_CLR)

path_to_BC = path_to_save_dir+BC_dir
print(path_to_BC)

path_to_test = path_to_save_dir+TEST_dir
print(path_to_test)

clr_train = np.load(path_to_CLR+"clr_train.npy")
clr_val = np.load(path_to_CLR+"clr_val.npy")
data_train = np.load(path_to_BC+"data_train.npy")
labels_train = np.load(path_to_BC+"labels_train.npy")
data_val = np.load(path_to_BC+"data_val.npy")
labels_val = np.load(path_to_BC+"labels_val.npy")
data_test_f = np.load(path_to_test+"data.npy")
labels_test_f = np.load(path_to_test+"labels.npy")

# print data dimensions
print( "CLR training data shape: " + str( clr_train.shape ), flush=True)
print( "CLR val data shape: " + str( clr_val.shape ), flush=True)
print()

# Removing the jet from the jet + constituents array
    # This will be done during training for the CLR datasets
cropped_train = remove_jet_and_rescale_pT(data_train, n_jets)
cropped_val = remove_jet_and_rescale_pT(data_val, n_jets)
cropped_test = remove_jet_and_rescale_pT(data_test_f, n_jets)

# Make some even smaller datasets to train the validation NN, LCT
    # Make this as large as your computing resources will allow!
percentage = 0.1

num_val_epoch_train = int(percentage*cropped_train.shape[0])
val_epoch_cropped_train = cropped_train[:num_val_epoch_train,:,:]
val_epoch_cropped_train_labels = labels_train[:num_val_epoch_train]

num_val_epoch_val = int(percentage*cropped_val.shape[0])
val_epoch_cropped_val = cropped_val[:num_val_epoch_val,:,:]
val_epoch_cropped_val_labels = labels_val[:num_val_epoch_val]


print("BC training set shape:", cropped_train.shape)
print("BC validation set shape:", cropped_val.shape)
print("STS shape:", cropped_test.shape)
print()

print("BC epoch check training set shape:", val_epoch_cropped_train.shape)
print("BC epoch check validation set shape:", val_epoch_cropped_val.shape)

print()
print()

# Plot num constituents

def get_num_constits(dataset):
    consits_list = []
    for collision in dataset:
        pts = collision[0,:]

        pads = np.where(pts==0)
        consits_list.append(dataset.shape[2]-len(pads[0]))
        
    return consits_list
        
    
"""

Define the transformer net

"""

# transformer hyperparams
# input dim to the transformer -> (pt,eta,phi)
input_dim = 3
output_dim = model_dim
dim_feedforward = model_dim
n_heads = 4
n_layers = 2
n_head_layers = 2
opt = "adam"

mask= False
cmask = True

learning_rate_trans = 0.0001
batch_size = 400
temperature = .1

early_stop = True

if early_stop:
    early_stopping = EarlyStopping()

# Turn on all the augmentations
rot = True # rotations
trs = False # translations
dis = True # distortion
col = False # collinear

center = "J1_phi_only_pi_2"


net = Transformer( input_dim, model_dim, output_dim, n_heads, dim_feedforward, 
                  n_layers, learning_rate_trans, n_head_layers, dropout=0.1, opt=opt )

# send network to device
net.to( device )

# define lr scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( net.optimizer, factor=0.2 )

"""
Define training parameters
"""

run_transformer = True
train_num_only = False # For fun if you just want to train on the alignment / uniformity loss alone
train_den_only = False

check_with_LCT = True # run a LCT at each validation epoch
check_with_NN = True

n_epochs = 800
loss_check_epoch = 10  # do validation loss, run a LCT and NN on the current reps
verbal_epoch = 10

"""

Train the CLR transformer

"""

if run_transformer:
    
    # THE TRAINING LOOP

    # initialise lists for storing training stats, validation loss
    losses_clr_num_jets = []
    loss_validation_num_jets = [[],[]] #epoch, loss
    
    losses_clr_numer_num_jets = []
    losses_clr_denom_num_jets = []
    
    lct_auc_num_jets = [[],[],[],[]] #epoch, auc (pt, eta, phi)
    nn_auc_num_jets = [[],[],[],[]] #epoch, auc (pt, eta, phi)

    mean_consts_post_split = [] # number of constituents in the jet after collinear splitting

    t0 = time.time()
    print( "---", flush=True )
    print( "Starting training loop, running for " + str( n_epochs ) + " epochs", flush=True)
    print( "---", flush=True )

    # re-batch the data on each epoch
    for epoch in range( n_epochs + 1 ):
        net.optimizer.zero_grad()
        net.train()

        # get batch_size number of indices
        indices_list = torch.split( torch.randperm( clr_train.shape[0] ), batch_size )

        # initialise lists to store batch stats
        losses_clr_e = []
        losses_clr_numer_e = []
        losses_clr_denom_e = []

        # the inner loop goes through the dataset batch by batch
        # augmentations of the jets are done on the fly
        for i, indices in enumerate( indices_list ): # random jets from the dataset
            """
            TRANSFORMATIONS AND DATA PREPARATION
            """
            x_i = clr_train[indices,:,:]
            x_i, x_j = apply_single_jet_augs(x_i, 2, center, rot, trs, dis, col)
            x_j = shift_phi(x_j)
            x_j = shift_eta(x_j)

            # rescaling pT
            max_pt = np.max(x_i[:,0,:])
            pt_rescale_denom  = max_pt/ 10.
            x_i = rescale_pts( x_i, pt_rescale_denom )
            x_j = rescale_pts( x_j, pt_rescale_denom )

            mean_consts_post_split.append(np.mean(get_num_constits(x_j)))

            x_i = torch.Tensor( x_i ).transpose(1,2).to( device ) # shape (batchsize, 2, 3)
            x_j = torch.Tensor( x_j ).transpose(1,2).to( device )
            z_i = net( x_i, use_mask=mask, use_continuous_mask=cmask ) # shape (batchsize, output_dim)
            z_j = net( x_j, use_mask=mask, use_continuous_mask=cmask )

            """
            LOSS CALCULATIONS
            """            
            # compute the loss based on predictions of the net and the correct answers
            loss = contrastive_loss( z_i, z_j, device, temperature, 1 ).to( device )
            loss_numer, loss_denom = contrastive_loss_num_den( z_i, z_j, device , 1)

            if train_den_only:
                loss_denom.backward()
            elif train_num_only:
                loss_numer.backward()
            else:
                loss.backward()

            loss_numer = loss_numer.detach().cpu().numpy()
            loss_denom = loss_denom.detach().cpu().numpy()

            net.optimizer.step()

            losses_clr_e.append( loss.detach().cpu().numpy() )
            losses_clr_numer_e.append(loss_numer)
            losses_clr_denom_e.append(loss_denom)

        """
        AVERAGING OF LOSSES
        """ 
        loss_clr_e = np.mean( np.array( losses_clr_e ) )
        ## scheduler
        scheduler.step( loss_clr_e )

        # storage
        losses_clr_num_jets.append( loss_clr_e )
        losses_clr_numer_num_jets.append( np.mean( np.array( losses_clr_numer_e ) ) )
        losses_clr_denom_num_jets.append( np.mean( np.array( losses_clr_denom_e ) ) )

        """
        EVERY SO OFTEN, GIVEN AN UPDATE
        """

        if epoch % verbal_epoch == 0:


            print( "epoch: " + str( epoch ) + ", loss: " + str( round(losses_clr_num_jets[-1], 5) ), flush=True )
            #print( "lr: " + str( scheduler._last_lr ), flush=True  )
            # summarize alignment and uniformity stats
            print( "numerator: " + str( losses_clr_numer_num_jets[-1] ) + ", denominator: " + str( losses_clr_denom_num_jets[-1] ), flush=True)
            print("time taken up to now: " + str(time.time()-t0))
            print()

        if epoch % loss_check_epoch == 0:
            net.eval()

            """
            Get the validation loss
            """
            print("Getting the validation CLR loss...")
            # store the epoch
            loss_validation_num_jets[0].append(epoch)

            with torch.no_grad():

                # get batch_size number of indices
                indices_list_val = torch.split( torch.randperm( clr_val.shape[0] ), batch_size )
                local_val_losses = []

                for j, indices_val in enumerate( indices_list_val ):

                    a_i = clr_val[indices_val,:,:]
                    a_i, a_j = apply_single_jet_augs(a_i, 2, center, rot, trs, dis, col)
                    a_j = shift_phi(a_j)
                    a_j = shift_eta(a_j)

                    # rescaling pT
                    max_pt = np.max(a_i[:,0,:])
                    pt_rescale_denom  = max_pt/ 10.
                    a_i = rescale_pts( a_i, pt_rescale_denom )
                    a_j = rescale_pts( a_j, pt_rescale_denom )

                    a_i = torch.Tensor( a_i ).transpose(1,2).to( device ) # shape (batchsize, 2, 3)
                    a_j = torch.Tensor( a_j ).transpose(1,2).to( device )
                    w_i = net( a_i, use_mask=mask, use_continuous_mask=cmask ) # shape (batchsize, output_dim)
                    w_j = net( a_j, use_mask=mask, use_continuous_mask=cmask )

                    loss_val = contrastive_loss( w_i, w_j, device, temperature, 1 ).to( device )
                    local_val_losses.append(loss_val.detach().cpu().numpy())

                loss_val_e = np.mean( np.array( local_val_losses ) )
                loss_validation_num_jets[1].append(loss_val_e)

                if early_stop:
                    early_stopping(loss_val_e)

            if check_with_LCT:
                """
                Run a LCT for signal vs background (supervised)
                """

                lct_train_reps = F.normalize( net.forward_batchwise( torch.Tensor( val_epoch_cropped_train ).transpose(1,2), data_test_f.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
                lct_test_reps = F.normalize( net.forward_batchwise( torch.Tensor( val_epoch_cropped_val ).transpose(1,2), data_val.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()

                print("Doing a short LCT...")

                lct_auc_num_jets[0].append(epoch)
                with torch.no_grad():
                    for trait in range(lct_train_reps.shape[1]): # going through the layers of the transformer
                        # run the LCT
                        #reg = LinearRegression().fit(lct_train_reps[:,trait,:], labels_test_f)
                        ridge = Ridge(alpha=1.0)
                        ridge.fit(lct_train_reps[:,trait,:], val_epoch_cropped_train_labels)

                        # make the prediction
                        predictions = ridge.predict(lct_test_reps[:,trait,:])
                        auc = roc_auc_score(val_epoch_cropped_val_labels, predictions)
                        lct_auc_num_jets[1+trait].append(auc)


            if check_with_NN:
                """
                Run a NN for signal vs background (supervised)
                """


                lct_train_reps = F.normalize( net.forward_batchwise( torch.Tensor( val_epoch_cropped_train ).transpose(1,2), data_test_f.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
                lct_test_reps = F.normalize( net.forward_batchwise( torch.Tensor( val_epoch_cropped_val ).transpose(1,2), data_val.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
               
                print("Doing a short NN...")
                num_epochs_nn = 800
                batch_size_nn = 400
                update_epochs_nn = 2*num_epochs_nn # no validation
                input_shape = model_dim
                lr_nn = 0.0001

                nn_auc_num_jets[0].append(epoch)
                # we need the grad turned on to train the nn
                for trait in range(lct_train_reps.shape[1]): # going through the layers of the transformer
                    # run the NN
                    performance_stats_nn = create_and_run_nn(device, input_shape, num_epochs_nn, batch_size_nn, 
                                                             update_epochs_nn,lr_nn, 
                                                              lct_train_reps[:,trait,:], val_epoch_cropped_train_labels, 
                                                              lct_train_reps[:,trait,:], val_epoch_cropped_train_labels, # actually no validation set
                                                              lct_test_reps[:,trait,:], val_epoch_cropped_val_labels, True)



                    nn_auc_num_jets[1+trait].append(performance_stats_nn["auc"])

                    plt.figure()
                    plt.plot(performance_stats_nn["tpr"], 1.0/performance_stats_nn["fpr"])
                    plt.yscale("log")
                    plt.xlabel("True Positive Rate")
                    plt.ylabel("1/(False Positive Rate)")
                    plt.title("NN"+str(trait))
                    plt.show()


                    plot_nn_losses = []
                    plot_nn_losses.append((performance_stats_nn["epochs"],
                                       performance_stats_nn["losses"], "NN loss"))
                    fig = plot_losses(plot_nn_losses, "NN"+str(trait)+" losses, training", True)  



        if early_stopping.early_stop:
            break
                            

    t1 = time.time()

    print( "JETCLR TRAINING DONE, time taken: " + str( np.round( t1-t0, 2 ) ), flush=True)

    # save out results
    print( "saving out data/results", flush=True)
    np.save( expt_dir+"clr_losses_train"+".npy", losses_clr_num_jets )
    np.save( expt_dir+"clr_losses_val_"+".npy", loss_validation_num_jets)
    np.save( expt_dir+"clr_numer_loss_train"+".npy", losses_clr_numer_num_jets )
    np.save( expt_dir+"clr_denom_loss_train"+".npy", losses_clr_denom_num_jets )

    np.save( expt_dir+"lct_auc"+".npy", lct_auc_num_jets )
    np.save( expt_dir+"nn_auc"+".npy", nn_auc_num_jets )

    # save out final trained model
    print( "saving out final jetCLR model", flush=True )
    torch.save(net.state_dict(), expt_dir+"final_model"+".pt")
    print()

    print("Avg # constits:", np.mean(mean_consts_post_split))

"""

Plot the training losses

"""

if run_transformer:
    
    losses_pdf_name = expt_dir + "CLR_training_losses.pdf"
    pp = PdfPages(losses_pdf_name)


    """
    Plot the training contrastive losses
    """
    plot_clr_losses = []

    plot_clr_losses.append((range(len(losses_clr_num_jets)),
                           losses_clr_num_jets, "CLR loss"))
    plot_clr_losses.append((loss_validation_num_jets[0],
                           loss_validation_num_jets[1],"Val loss"))
    fig = plot_losses(plot_clr_losses, "Contrastive losses, training", True)  

    pp.savefig(fig)

    """
    Plot the LC + NN AUC
    """

    plot_LCT_stats = []
    plot_LCT_stats.append((lct_auc_num_jets[0], lct_auc_num_jets[1],
                             "LC transformer"))
    plot_LCT_stats.append((lct_auc_num_jets[0], lct_auc_num_jets[2],
                            "LC hidden layer"))
    plot_LCT_stats.append((lct_auc_num_jets[0], lct_auc_num_jets[3],
                            "LC output layer"))

    fig = plot_losses(plot_LCT_stats, "ROC Area", False)  
    pp.savefig(fig)

    plot_NN_stats = []
    plot_NN_stats.append((nn_auc_num_jets[0], nn_auc_num_jets[1],
                             "NN transformer"))
    plot_NN_stats.append((nn_auc_num_jets[0], nn_auc_num_jets[2],
                            "NN hidden layer"))
    plot_NN_stats.append((nn_auc_num_jets[0], nn_auc_num_jets[3],
                            "NN output layer"))

    fig = plot_losses(plot_NN_stats, "ROC Area", False)  
    pp.savefig(fig)

    """
    Plot the training contrastive losses num + denom
    """

    plot_num_val_losses = []
    plot_num_val_losses.append((range(len(losses_clr_numer_num_jets)),
                           -np.array(losses_clr_numer_num_jets), ""))
    fig = plot_losses(plot_num_val_losses, "-Alignment losses (should increase)", True)  
    pp.savefig(fig)

    plot_den_val_losses = []
    plot_den_val_losses.append((range(len(losses_clr_denom_num_jets)),
                           np.array(losses_clr_denom_num_jets),  ""))
    fig = plot_losses(plot_den_val_losses, "Uniformity losses (should decrease)", True)  
    pp.savefig(fig)


    pp.close()


"""

Run a BC on the final representations

"""

# Loading in the final transformer

loaded_net = Transformer( input_dim, model_dim, output_dim, n_heads, dim_feedforward, 
                  n_layers, learning_rate_trans, n_head_layers, dropout=0.1, opt=opt )

loaded_net.load_state_dict(torch.load(expt_dir+"final_model"+".pt"))
loaded_net.eval()

print("Loading data into net...")
lct_train_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( cropped_train ).transpose(1,2), data_train.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
lct_val_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( cropped_val ).transpose(1,2), data_test_f.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()
lct_test_reps = F.normalize( loaded_net.forward_batchwise( torch.Tensor( cropped_test ).transpose(1,2), data_test_f.shape[0], use_mask=mask, use_continuous_mask=cmask ).detach().cpu(), dim=-1  ).numpy()

print("Data loaded!")


# First, run a LCT

LCT_pdf_name = expt_dir + "LCT_plots.pdf"
pp = PdfPages(LCT_pdf_name)

fig = plt.figure()

print("Doing a LCT...")
# Need to transform the data into the representation space first
with torch.no_grad():
    for trait in range(lct_train_reps.shape[1]): # going through the layers of the transformer
        # run the LCT
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(lct_train_reps[:,trait,:], labels_train)
            
        # make the prediction
        predictions = ridge.predict(lct_test_reps[:,trait,:])
        fpr, tpr, _ = roc_curve(labels_test_f, predictions)
        
        plt.plot(tpr, 1.0/fpr, label = "LCT"+str(trait))
        
        np.save( expt_dir+"CLR_LCT"+str(trait)+"_fpr"+".npy", fpr )
        np.save( expt_dir+"CLR_LCT"+str(trait)+"_tpr"+".npy", tpr )
        
        predicted = np.round(predictions).reshape(labels_test_f.size)
        total = labels_test_f.size
        correct = (predicted == labels_test_f).sum()    
        
        print("CLR_LCT"+str(trait),"acc:",correct/total)
        
        

plt.yscale("log")
plt.xlabel("True Positive Rate")
plt.ylabel("1/(False Positive Rate)")
plt.legend()
plt.show()

pp.savefig(fig)
pp.close()

        
print("LCT data saved")



# Then, run a FCN

# FCN hyperparameters
num_epochs_nn = 1200
batch_size_nn = 400
update_epochs_nn = 10
input_shape = model_dim
lr_nn = 0.001

NN_pdf_name = expt_dir + "NN_plots.pdf"
pp = PdfPages(NN_pdf_name)

print("Doing a NN...")
# Need to transform the data into the representation space first
#with torch.no_grad():
for trait in range(lct_train_reps.shape[1]): # going through the layers of the transformer
    
    # make dictionaries to stre the losses
    losses_nn_latent_train = []
    losses_nn_latent_val = [[],[]] #epoch, loss

    # run the NN
    performance_stats_nn = create_and_run_nn(device, input_shape, num_epochs_nn, batch_size_nn, update_epochs_nn,lr_nn, 
                                     lct_train_reps[:,trait,:], labels_train, 
                  lct_val_reps[:,trait,:], labels_val,
                  lct_test_reps[:,trait,:], labels_test_f, True)

    #plt.plot(performance_stats_nn["tpr"], 1.0/performance_stats_nn["fpr"], label = "NN"+str(trait))
    
    np.save( expt_dir+"CLR_NN"+str(trait)+"_fpr"+".npy", performance_stats_nn["fpr"] )
    np.save( expt_dir+"CLR_NN"+str(trait)+"_tpr"+".npy", performance_stats_nn["tpr"] )

    print("Accuracy of the network: %d %%" % (100.00 *performance_stats_nn["acc"]))
    print("ROC AUC:", performance_stats_nn["auc"])
    
    
    plot_nn_losses = []
    plot_nn_losses.append((performance_stats_nn["epochs"],
                       performance_stats_nn["losses"], "NN loss"))
    plot_nn_losses.append((performance_stats_nn["val_epochs"],
                       performance_stats_nn["val_losses"],"Val loss"))
    fig = plot_losses(plot_nn_losses, "NN losses, training", True)  
    pp.savefig(fig)
    
    
    # Save out the losses dictionaries
    losses_nn_latent_train = performance_stats_nn["losses"]
    losses_nn_latent_val[0] = performance_stats_nn["val_epochs"]
    losses_nn_latent_val[1] = performance_stats_nn["val_losses"]
    
    np.save( expt_dir+"NN"+str(trait)+"_latent_losses_train"+".npy", losses_nn_latent_train )
    np.save( expt_dir+"NN"+str(trait)+"_latent_losses_val"+".npy", losses_nn_latent_val )
    

pp.close()

print("NN data saved")



