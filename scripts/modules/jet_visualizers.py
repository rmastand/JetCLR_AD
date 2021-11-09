import numpy as np
import matplotlib.pyplot as plt

from modules.jet_vars import nsub, convert_constits_coords, mj, mjj

def plot_jets_phase_plane(jet1, jet2, s, xlims=(-.5,.5), ylims=(-.5,.5)):
    
    """ 
    # plotting pt
    fig, ax = plt.subplots(1,2, figsize = (10,4))
    ax[0].hist(jet1[0,:])
    ax[0].set_xlabel("Rescaled pT")
    ax[1].hist(jet2[0,:])
    ax[1].set_xlabel("Rescaled pT")
    fig.show()"""
    
    
    # plotting the eta-phi plane
    fig, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].scatter(jet1[1,:][::-1], jet1[2,:][::-1], s=s*jet1[0,:][::-1])
    ax[0].set_xlabel("$\Delta\eta$")
    ax[0].set_ylabel("$\Delta\phi$")
    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    
    ax[1].scatter(jet2[1,:][::-1], jet2[2,:][::-1],s=s*jet2[0,:][::-1])
    ax[1].set_xlabel("$\Delta\eta$")
    ax[1].set_ylabel("$\Delta\phi$")
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    
    fig.show()
    
def plot_nsubs(list_of_jets_1, list_of_jets_2, nbins = 20, title = ""):
    """
    INPUTS
    2 numpy arrays of size (# jets, 3, # constituents)where the 1-index goes through (pT, eta, phi)
    """
    
    # convert the pT, eta, phi coords to E, px py, px
    list_of_jets_1_cart = convert_constits_coords(list_of_jets_1)
    list_of_jets_2_cart = convert_constits_coords(list_of_jets_2)
    
    # calculate the n_jettiness
    taus_1 = nsub(list_of_jets_1_cart)
    taus_2 = nsub(list_of_jets_2_cart)
    
    taus_1_21 = taus_1[:,0]
    taus_1_32 = taus_1[:,1]
    taus_2_21 = taus_2[:,0]
    taus_2_32 = taus_2[:,1]
               
    bins = np.linspace(0,1,nbins)
    alpha = 0.3

    # plot tau 21
    fig, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].hist(taus_1_21, bins = bins, label = "Orig.", alpha = alpha)
    ax[0].hist(taus_2_21, bins = bins, label = "Mod.", alpha = alpha)
    ax[0].set_xlabel("$\\tau_{21}$"+title)
    ax[0].set_ylabel("Counts")
    ax[0].legend()
    
    # plot tau 32
    ax[1].hist(taus_1_32, bins = bins, label = "Orig.", alpha = alpha)
    ax[1].hist(taus_2_32, bins = bins, label = "Mod.", alpha = alpha)
    ax[1].set_xlabel("$\\tau_{32}$"+title)
    ax[1].set_ylabel("Counts")
    ax[1].legend()
    
    fig.show()
    
def plot_mj(list_of_jets_1, list_of_jets_2, bins = np.linspace(0,700,20), title = ""):
    """
    INPUTS
    2 numpy arrays of size (# jets, 3, # constituents)where the 1-index goes through (pT, eta, phi)
    """
    
    # calculate the jet mass
    list_of_m1 = mj(list_of_jets_1)
    list_of_m2 = mj(list_of_jets_2)
    
    alpha = 0.3

    # plot 
    fig, ax = plt.subplots(1,1, figsize = (6,6))
    ax.hist(list_of_m1, bins = bins, label = "Orig.", alpha = alpha)
    ax.hist(list_of_m2, bins = bins, label = "Mod.", alpha = alpha)
    ax.set_xlabel("$m_{j}$"+title)
    ax.set_ylabel("Counts")
    ax.legend()
    
    fig.show()
    
def plot_mjj(list_jets1_orig, list_jets2_orig, list_jets1_mod, list_jets2_mod, bins = np.linspace(0,7000,20), title = ""):
    """
    INPUTS
    2 numpy arrays of size (# jets, 3, # constituents)where the 1-index goes through (pT, eta, phi)
    """
    
    # calculate the mjj
    list_of_mjj_orig = mjj(list_jets1_orig, list_jets2_orig)
    list_of_mjj_mod = mjj(list_jets1_mod, list_jets2_mod)
    
    alpha = 0.3

    # plot 
    fig, ax = plt.subplots(1,1, figsize = (6,6))
    ax.hist(list_of_mjj_orig, bins = bins, label = "Orig.", alpha = alpha)
    ax.hist(list_of_mjj_mod, bins = bins, label = "Mod.", alpha = alpha)
    ax.set_xlabel("$m_{jj}$"+title)
    ax.set_ylabel("Counts")
    ax.legend()
    
    fig.show()
    
    