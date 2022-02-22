import os
import sys
import numpy as np
import random
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.file_readers import phi_wrap
from modules.jet_visualizers import plot_jets_phase_plane


def translate_jets( batch, width=1.0 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta-phi translated jets, same shape as input
    '''
    mask = (batch[:,0] > 0) # 1 for constituents with non-zero pT, 0 otherwise
    ptp_eta  = np.ptp(batch[:,1,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    ptp_phi  = np.ptp(batch[:,2,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    low_eta  = -width*ptp_eta
    high_eta = +width*ptp_eta
    low_phi  = np.maximum(-width*ptp_phi, -np.pi-np.amin(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    high_phi = np.minimum(+width*ptp_phi, +np.pi-np.amax(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    shift_eta = mask*np.random.uniform(low=low_eta, high=high_eta, size=(batch.shape[0], 1))
    shift_phi = mask*np.random.uniform(low=low_phi, high=high_phi, size=(batch.shape[0], 1))
    shift = np.stack([np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1)
    shifted_batch = batch+shift
    return shifted_batch


def rotate_jets( batch, num_jets = 1 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets rotated independently in eta-phi, same shape as input
    '''
    """
    print(batch.shape)
    split_batch = np.split(batch, num_jets, axis = 2)
    print([x.shape for x in split_batch])
    """
    rot_angle = np.random.rand(batch.shape[0])*2*np.pi
    c = np.cos(rot_angle)
    s = np.sin(rot_angle)
    o = np.ones_like(rot_angle)
    z = np.zeros_like(rot_angle)
    rot_matrix = np.array([[o, z, z], [z, c, -s], [z, s, c]]) # (3, 3, batchsize)
    return np.einsum('ijk,lji->ilk', batch, rot_matrix)

def normalise_pts( batch ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-normalised jets, pT in each jet sums to 1, same shape as input
    '''
    batch_norm = batch.copy()
    batch_norm[:,0,:] = np.nan_to_num(batch_norm[:,0,:]/np.sum(batch_norm[:,0,:], axis=1)[:, np.newaxis], posinf = 0.0, neginf = 0.0 )
    return batch_norm

def rescale_pts( batch, pt_rescale_denom ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of pT-rescaled jets, each constituent pT is rescaled by pt_rescale_denom, same shape as input
    '''
    batch_rscl = batch.copy()
    batch_rscl[:,0,:] = np.nan_to_num(batch_rscl[:,0,:]/pt_rescale_denom, posinf = 0.0, neginf = 0.0 )
    return batch_rscl

def crop_jets( batch, nc ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of cropped jets, each jet is cropped to nc constituents, shape (batchsize, 3, nc)
    '''
    batch_crop = batch.copy()
    return batch_crop[:,:,0:nc]

def distort_jets( batch, strength=0.1, pT_clip_min=0.1 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with each constituents position shifted independently, shifts drawn from normal with mean 0, std strength/pT, same shape as input
    '''
    pT = batch[:,0]   # (batchsize, n_constit)
    shift_eta = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / pT.clip(min=pT_clip_min), posinf = 0.0, neginf = 0.0 )# * mask
    shift_phi = np.nan_to_num( strength * np.random.randn(batch.shape[0], batch.shape[2]) / pT.clip(min=pT_clip_min), posinf = 0.0, neginf = 0.0 )# * mask
    shift = np.stack( [ np.zeros( (batch.shape[0], batch.shape[2]) ), shift_eta, shift_phi ], 1)
    return batch + shift

def collinear_fill_jets( batch ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of jets with collinear splittings, the function attempts to fill as many of the zero-padded args.nconstit
    entries with collinear splittings of the constituents by splitting each constituent at most once, same shape as input
    '''
    batchb = batch.copy()
    nc = batch.shape[2]
    nzs = np.array( [ np.where( batch[:,0,:][i]>0.0)[0].shape[0] for i in range(len(batch)) ] )
    for k in range(len(batch)):
        zs1 = int(nc-nzs[k])
        nfill = np.min( [ zs1, nzs[k] ] )
        els = np.random.choice( np.linspace(0,nzs[k]-1,nzs[k]), size=nfill, replace=False )
        rs = np.random.uniform( size=nfill )
        for j in range(nfill):
            batchb[k,0,int(els[j])] = rs[j]*batch[k,0,int(els[j])]
            batchb[k,0,int(nzs[k]+j)] = (1-rs[j])*batch[k,0,int(els[j])]
            batchb[k,1,int(nzs[k]+j)] = batch[k,1,int(els[j])]
            batchb[k,2,int(nzs[k]+j)] = batch[k,2,int(els[j])]
        
    return batchb


def shift_eta( batch, width=0.5 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of eta translated jets, same shape as input
    
    ** intended for use as an event augmentation **
    '''
    mask = (batch[:,0] > 0) # 1 for constituents with non-zero pT, 0 otherwise
    ptp_eta  = np.ptp(batch[:,1,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    low_eta  = -width*ptp_eta
    high_eta = +width*ptp_eta
    shift_eta = mask*np.random.uniform(low=low_eta, high=high_eta, size=(batch.shape[0], 1))
    shift_phi = mask*np.random.uniform(low=0, high=0, size=(batch.shape[0], 1))
    shift = np.stack([np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1)
    shifted_batch = batch+shift
    return shifted_batch


def shift_phi( batch, width=1.0 ):
    '''
    Input: batch of jets, shape (batchsize, 3, n_constit)
    dim 1 ordering: (pT, eta, phi)
    Output: batch of phi translated jets, same shape as input
    
    ** intended for use as an event augmentation **
    '''
    mask = (batch[:,0] > 0) # 1 for constituents with non-zero pT, 0 otherwise
    ptp_phi  = np.ptp(batch[:,2,:], axis=-1, keepdims=True) # ptp = 'peak to peak' = max - min
    low_phi  = np.maximum(-width*ptp_phi, -np.pi-np.amin(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    high_phi = np.minimum(+width*ptp_phi, +np.pi-np.amax(batch[:,2,:], axis=1).reshape(ptp_phi.shape))
    shift_eta = mask*np.random.uniform(low=0, high=0, size=(batch.shape[0], 1))
    shift_phi = mask*np.random.uniform(low=low_phi, high=high_phi, size=(batch.shape[0], 1))
    shift = np.stack([np.zeros((batch.shape[0], batch.shape[2])), shift_eta, shift_phi], 1)
    shifted_batch = batch+shift
    return shifted_batch


def apply_single_jet_augs(events, njets, center, rot, trs, dis, col, trsw = 1.0, ptst = 0.1, ptcm = 1.0):
    """
    INPUTS:
    
    events: np.array of size (# events, 3, # jets * (1 + # consts / jet))
        The last index goes (in decreasing jet mass): jet1, consts in jet1, jet2, consts in jet2
        There may be zero pads
        NOTE that for the jet, pT eta phi are for the jet itself. The constituents may have pT eta phi RELATIVE to 
            some center point
        
    njets: int with the number of jets contained in each event
    
    center: Contans a code to calculate the tuple containing (eta_c, phi_c) for the point that the CONSTITUENTS 
                (not jet) are centered around
        i.e. the constituents have (eta, phi) = (eta_orig - eta_c, phi_orig - phi_c) where eta_orig, phi_orig 
            were output by the jet clustering algorithm
            
        center options: (describe the input constituents)
        "none": does not affect the eta, phi of the stores constituents
        "J1_delta": returns eta - etaJ, phi - phiJ where J represents the hardest jet 
        "J1_phi_only_pi_2": shifts phi s.t. the hardest jet is centred at pi/2, NO change to eta
     
    OUTPUTS:

    orig_events: np.array of size (# events, 3, # jets * (# consts / jet))
    mod_events: np.array of size (# events, 3, # jets * (# consts / jet))
    
    PROCEDURE:
    (1) split the event into njets jets. For each jet:
        (2) Recenter all the constituents at eta, phi = 0, 0
        (3) apply the augmentation
        (4) un-center the jet
    (5) return the combined event
    
    This modifies the jets, so it should be done on a COPY of the original array
    """
    
    #l = 2
    aug_events = events.copy()
    
    # get the etas, phis for the hardest mass jet (used for the recentering)
    hardest_etas = aug_events[:,1,0]
    hardest_phis = aug_events[:,2,0]
    
    # split the event into the jets
    split_jets = np.split(events, njets, axis = 2)
    aug_split_jets = np.split(aug_events, njets, axis = 2)
   
    # for storing the augmented SINGLE jets
    orig_jets = []
    modified_jets = []
    
    # remove the jets from the event representation, store
    for i, subjet in enumerate(split_jets):
        orig_jets.append(subjet[:,:,1:])

    # now go through each jet
    for i, subjet in enumerate(aug_split_jets):
        
        # get the jet eta, phi
        jet_etas = subjet[:,1,0]
        jet_phis = subjet[:,2,0]
        
        # take only the constituents (i.e. drop the 0th, which is the jet)
        subjet = subjet[:,:,1:]
        
        # calculate the jet recentering coordinates
        if center == "J1_phi_only_pi_2":
            eta_shift = -jet_etas
            phi_shift = hardest_phis - jet_phis - np.pi/2
        elif center == "none":
            eta_shift = -jet_etas
            phi_shift = -jet_phis  
        else:
            print("ERROR: CENTER POSITION NOT CHOSEN")
            
        #plot_jets_phase_plane(subjet[l], subjet[l], 2, xlims=(-3,3), ylims=(-3,3))  
            
        # copy the array for each of the constituents
        eta_shift = np.repeat(eta_shift, subjet.shape[2], axis = 0).reshape(subjet.shape[0],subjet.shape[2])
        phi_shift = np.repeat(phi_shift, subjet.shape[2], axis = 0).reshape(subjet.shape[0],subjet.shape[2])   
            
        # recenter the jet
        subjet[:,1,:] += eta_shift
        subjet[:,2,:] += phi_shift
        # phi wrapping
        for e in range(subjet[:,2,:].shape[0]):
            for c in range(subjet[:,2,:].shape[1]):
                subjet[e,2,c] = phi_wrap(subjet[e,2,c])
                
        #plot_jets_phase_plane(subjet[l], subjet[l], 2, xlims=(-3,3), ylims=(-3,3))  

        # apply the augmentation
        if rot:
            subjet = rotate_jets( subjet )
        if trs:
            subjet = translate_jets( subjet, width=trsw )
        if col:
            subjet = collinear_fill_jets( subjet )
            subjet = collinear_fill_jets( subjet )
        if dis:
            subjet = distort_jets( subjet, strength=ptst, pT_clip_min=ptcm )
            
        #plot_jets_phase_plane(subjet[l], subjet[l], 2, xlims=(-3,3), ylims=(-3,3))  
        
        # un-recenter the jet
        subjet[:,1,:] -= eta_shift
        subjet[:,2,:] -= phi_shift
        # phi wrapping
        for e in range(subjet[:,2,:].shape[0]):
            for c in range(subjet[:,2,:].shape[1]):
                subjet[e,2,c] = phi_wrap(subjet[e,2,c])
                
        #plot_jets_phase_plane(subjet[l], subjet[l], 2, xlims=(-3,3), ylims=(-3,3))  
        
        modified_jets.append(subjet)
    
    # recombine the jets
    orig_events = np.concatenate(orig_jets, axis = 2)
    mod_events = np.concatenate(modified_jets, axis = 2)
    
    return(orig_events, mod_events)

def remove_jet_and_rescale_pT(events, njets):
    """
    INPUTS: 
    
    events: np.array of size (# events, 3, # jets * (1 + # consts / jet))
        The last index goes (in decreasing jet mass): jet1, consts in jet1, jet2, consts in jet2
        There may be zero pads
        NOTE that for the jet, pT eta phi are for the jet itself. The constituents may have pT eta phi RELATIVE to 
            some center point
        
    njets: int with the number of jets contained in each event
    
    OUTPUTS:

    consts_only: np.array of size (# events, 3, # jets * (# consts / jet))
    """
        
    consts_only = []
    split_jets = np.split(events, njets, axis = 2)

    # now go through each jet
    for i, subjet in enumerate(split_jets):
        # Take only the constituents (i.e. drop the 0th, which is the jet)
        subjet = subjet[:,:,1:]
        consts_only.append(subjet)
    
    # combine the jets
    consts_only = np.concatenate(consts_only, axis = 2)
    
    # rescale pTs
    max_pt = np.max(consts_only[:,0,:])
    pt_rescale_denom  = max_pt/ 10.
    consts_only = rescale_pts( consts_only, pt_rescale_denom )
    
    return(consts_only)