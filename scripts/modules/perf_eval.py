# import standard python modules
import os
import sys
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# import torch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# import simple FCN network
from modules.fcn_linear import fully_connected_linear_network

# useful function to find nearest number in a list
def find_nearest( array, value ):
    array = np.asarray( array )
    idx = ( np.abs( array-value ) ).argmin()
    return array[idx]

# function to compute auc and inverse-mistag at 0.5 eff from roc curve
def get_perf_stats( labels, measures ):
    measures = np.nan_to_num( measures )
    auc = metrics.roc_auc_score( labels, measures )
    fpr,tpr,thresholds = metrics.roc_curve( labels, measures )
    fpr2 = [ fpr[i] for i in range( len( fpr ) ) if tpr[i]>=0.5]
    tpr2 = [ tpr[i] for i in range( len( tpr ) ) if tpr[i]>=0.5]
    try:
        imtafe = np.nan_to_num( 1 / fpr2[ list( tpr2 ).index( find_nearest( list( tpr2 ), 0.5 ) ) ] )
    except:
        imtafe = 1
    return auc, imtafe

# a simple LCT
def linear_classifier_test( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, reps_tr_in, trlab_in, reps_te_in, telab_in ):
    # define device
    xdevice = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    # initialise the network
    fcn_linear = fully_connected_linear_network( linear_input_size, 1, linear_learning_rate )
    fcn_linear.to( xdevice )
    # batch indices
    # define losses and sigmoid
    bce_loss = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    losses = []
    # start training
    for epoch in range( linear_n_epochs ):
        indices_list = torch.split( torch.randperm( reps_tr_in.shape[0] ), linear_batch_size )
        for i, indices in enumerate( indices_list ):
            losses_e = []
            fcn_linear.optimizer.zero_grad()
            x = reps_tr_in[indices,:] # (linear_batch_size, output_dim)
            l = trlab_in[indices] # (linear_batch_size)
 
            x = torch.Tensor( x ).view(linear_batch_size, -1,  ).to( xdevice )
            l = torch.Tensor( l ).view(linear_batch_size, -1 ).to( xdevice )
            z = sigmoid( fcn_linear( x ) ).to( xdevice )
            loss = bce_loss( z, l ).to( xdevice )
            loss.backward()
            fcn_linear.optimizer.step()
            losses_e.append( loss.detach().cpu().numpy() )    
        losses.append( np.mean( np.array( losses_e )  ) )
    # evaluate the network on the testing data
    #out_dat = sigmoid( fcn_linear( torch.Tensor( reps_te_in ).view(-1, linear_input_size).to( xdevice ) ) ).detach().cpu().numpy()
    out_dat = fcn_linear( torch.Tensor( reps_te_in ).view(-1, linear_input_size).to( xdevice ) ).detach().cpu().numpy()
    out_lbs = telab_in
    return out_dat, out_lbs, losses

# a more detailed LCT
def linear_classifier_test_detailed( linear_input_size, linear_batch_size, linear_n_epochs, linear_learning_rate, reps_tr_in, trlab_in, reps_te_in, telab_in ):
    # define device
    xdevice = torch.device( "cuda" if torch.cuda.is_available() else "cpu" )
    # initialise the network
    fcn_linear = fully_connected_linear_network( linear_input_size, 1, linear_learning_rate )
    fcn_linear.to( xdevice )
    # define losses and sigmoid
    bce_loss = nn.BCELoss()
    sigmoid = nn.Sigmoid()
    # initialise lists to track training stats
    losses = []
    test_losses = []
    aucs = []
    imtafes = []
    # start training
    for epoch in range( linear_n_epochs ):
        # batch indices
        indices_list = torch.split( torch.randperm( reps_tr_in.shape[0] ), linear_batch_size )
        for i, indices in enumerate( indices_list ):
            losses_e = []
            fcn_linear.optimizer.zero_grad()
            x = reps_tr_in[indices,:]
            l = trlab_in[indices]
            x = torch.Tensor( x ).view( -1, linear_input_size ).to( xdevice )
            l = torch.Tensor( l ).view( -1, 1 ).to( xdevice )
            z = sigmoid( fcn_linear( x ) ).to( xdevice )
            loss = bce_loss( z, l ).to( xdevice )
            loss.backward()
            fcn_linear.optimizer.step()
            losses_e.append( loss.detach().cpu().numpy() )    
        losses.append( np.mean( np.array( losses_e )  ) )
        if epochs%10==0:
            out_dat = fcn_linear( torch.Tensor( reps_te_in ).view(-1, linear_input_size).to( xdevice ) ).detach().cpu().numpy()
            out_lbs = telab_in
            test_loss = bce_loss( out_dat, out_lbs )
            auc_e, imtafe_e = get_perf_stats( out_lbs, out_dat )
            tes_tlosses.append( test_loss )
            aucs.append( auc_e )
            imtafes.append( imtafe_e )
    # evaluate the network on the testing data
    #out_dat = sigmoid( fcn_linear( torch.Tensor( reps_te_in ).view(-1, linear_input_size).to( xdevice ) ) ).detach().cpu().numpy()
    out_dat = fcn_linear( torch.Tensor( reps_te_in ).view(-1, linear_input_size).to( xdevice ) ).detach().cpu().numpy()
    out_lbs = telab_in
    return out_dat, out_lbs, losses, test_losses, aucs, imtafes


def plot_losses(list_of_plots, title, ylog = False):
    """
    list_of_plots = [(plot_x, plot_y, label)]
    """
    plt.figure()
    for to_plot in list_of_plots:
       
        plt.plot(to_plot[0], to_plot[1], label=to_plot[2])
    plt.legend()
    if ylog:
        plt.yscale("log")
    plt.xlabel("Epochs")
    plt.title(title)
    plt.show()
