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
    ax[0].scatter(jet1[1,:][::-1], jet1[2,:][::-1], s=s*jet1[0,:][::-1], color = "mediumseagreen")
    ax[0].set_xlabel("$\eta$")
    ax[0].set_ylabel("$\phi$")

    ax[0].set_xlim(xlims)
    ax[0].set_ylim(ylims)
    ax[0].text(0.6, 0.9, "Original Event", fontsize = 16, transform=ax[0].transAxes)
    
    ax[1].scatter(jet2[1,:][::-1], jet2[2,:][::-1],s=s*jet2[0,:][::-1], color = "darkcyan")
    ax[1].set_xlabel("$\eta$")
    ax[1].set_ylabel("$\phi$")
    ax[1].set_xlim(xlims)
    ax[1].set_ylim(ylims)
    ax[1].text(0.6, 0.9, "Modified Event", fontsize = 16, transform=ax[1].transAxes)
    
    fig.show()
    
    return fig
    
def plot_nsubs(orig_hardest, mod_hardest, orig_second, mod_second, nbins = 20, title = ""):
    """
    INPUTS
    2 numpy arrays of size (# jets, 3, # constituents)where the 1-index goes through (pT, eta, phi)
    """
    
    # convert the pT, eta, phi coords to E, px py, px
    list_of_orig_hardest_cart = convert_constits_coords(orig_hardest)
    list_of_mod_hardest_cart = convert_constits_coords(mod_hardest)
    list_of_orig_second_cart = convert_constits_coords(orig_second)
    list_of_mod_second_cart = convert_constits_coords(mod_second)
    
    # calculate the n_jettiness
    taus_orig_hardest = nsub(list_of_orig_hardest_cart)
    taus_mod_hardest = nsub(list_of_mod_hardest_cart)
    taus_orig_second = nsub(list_of_orig_second_cart)
    taus_mod_second = nsub(list_of_mod_second_cart)
    
    
    taus_orig_hardest_21 = taus_orig_hardest[:,0]
    taus_orig_hardest_32 = taus_orig_hardest[:,1]
    taus_mod_hardest_21 = taus_mod_hardest[:,0]
    taus_mod_hardest_32 = taus_mod_hardest[:,1]
    
    taus_orig_second_21 = taus_orig_second[:,0]
    taus_orig_second_32 = taus_orig_second[:,1]
    taus_mod_second_21 = taus_mod_second[:,0]
    taus_mod_second_32 = taus_mod_second[:,1]
               
    bins = np.linspace(0,1.1,nbins)
    alpha = 0.7

    # plot tau 21
    fig21, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].hist(taus_orig_hardest_21, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax[0].hist(taus_mod_hardest_21, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax[0].set_xlabel("$\\tau_{21}$, hardest jet")
    ax[0].set_ylabel("Counts")
    ax[0].legend()
    
    ax[1].hist(taus_orig_second_21, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax[1].hist(taus_mod_second_21, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax[1].set_xlabel("$\\tau_{21}$, second jet")
    ax[1].set_ylabel("Counts")
    plt.ylim(0,550)
    ax[1].legend()
    


    fig21.show()
    
    bins = np.linspace(0,1.1,nbins)
    
    # plot tau 32
    fig32, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].hist(taus_orig_hardest_32, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax[0].hist(taus_mod_hardest_32, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax[0].set_xlabel("$\\tau_{32}$, hardest jet")
    ax[0].set_ylabel("Counts")
    ax[0].legend()
    
    ax[1].hist(taus_orig_second_32, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax[1].hist(taus_mod_second_32, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax[1].set_xlabel("$\\tau_{32}$, second jet")
    ax[1].set_ylabel("Counts")
    ax[1].legend()
    

    fig32.show()
    
    
    
    return fig21, fig32


    
def plot_mj(list_of_hardest_orig, list_of_hardest_mod,list_of_second_orig, list_of_second_mod, bins = np.linspace(0,700,20), title = ""):
    """
    INPUTS
    2 numpy arrays of size (# jets, 3, # constituents)where the 1-index goes through (pT, eta, phi)
    """
    
    # calculate the jet mass
    list_of_hardest_orig = mj(list_of_hardest_orig)
    list_of_hardest_mod = mj(list_of_hardest_mod)
    list_of_second_orig = mj(list_of_second_orig)
    list_of_second_mod = mj(list_of_second_mod)
    
    alpha = .7

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].hist(list_of_hardest_orig, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax[0].hist(list_of_hardest_mod, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax[0].set_xlabel("Mass [GeV], hardest jet")
    ax[0].set_ylabel("Counts")
    ax[0].legend()
    
    ax[1].hist(list_of_second_orig, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax[1].hist(list_of_second_mod, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax[1].set_xlabel("Mass [GeV], second jet")
    ax[1].set_ylabel("Counts")
    ax[1].legend()
    
    fig.show()
    
    return fig
    
def plot_mjj(list_jets1_orig, list_jets2_orig, list_jets1_mod, list_jets2_mod, bins = np.linspace(0,7000,20), title = ""):
    """
    INPUTS
    2 numpy arrays of size (# jets, 3, # constituents)where the 1-index goes through (pT, eta, phi)
    """
    
    # calculate the mjj
    list_of_mjj_orig = mjj(list_jets1_orig, list_jets2_orig)
    list_of_mjj_mod = mjj(list_jets1_mod, list_jets2_mod)
    
    alpha = 0.7

    # plot 
    fig, ax = plt.subplots(1,1, figsize = (6,6))
    ax.hist(list_of_mjj_orig, bins = bins, label = "Original Events", alpha = alpha,color = "mediumseagreen")
    plt.rcParams['hatch.linewidth'] = 2
    ax.hist(list_of_mjj_mod, bins = bins, label = "Modified Events", color = "darkcyan" , histtype = "step", hatch="/", linewidth = 2)
    ax.set_xlabel("$m_{jj}$"+title)
    ax.set_ylabel("Counts")
    ax.legend()
    
    fig.show()
    
    return fig
    
    