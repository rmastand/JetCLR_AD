import numpy as np
import pandas as pd
import sys

# pyjet
from pyjet import cluster

"""
FROM UNCLUSTERED PARTICLES
"""

def phi_wrap(x):
    if x > np.pi:
        return x - 2*np.pi
    elif x < -np.pi:
        return x + 2*np.pi
    else:
        return x


def pandas_to_unclustered_particles(input_frame):
    
    """
    Reads in the dataset of 700 UNCLUSTERED particles (pt, eta, phi) and returns a list (len N) of lists (len M) of np.dtypes
    Also returns a list of labels
    List elements of the outer list are a list of the dtypes, each one corresponding to a particle
    M particles, N collision events
    """
    
    labels = []
    
    dataset = input_frame.copy()
    list_of_events = []
    for row in range(dataset.shape[0]):
        if row % 1000 == 0:
            print("Reading row", row, "of", dataset.shape[0], "...")
        loc_row = dataset.iloc[row]
        collision =  np.array([(loc_row[3*i],loc_row[3*i+1],loc_row[3*i+2],0) for i in range(int(len(loc_row)/3))],
                     np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))  
        list_of_events.append(collision) 
        labels.append(loc_row[2100])  #the event label

    return np.array(list_of_events), np.array(labels)

def cluster_and_prepare(dataset, n, R):
    
    """
    Takes a list (len N) of lists (len M), M unclustered particles, N collision events
    Clusters into n jets
    Returns output for the jet CLR code : 3d numpy array (N collision events, 3 jet params, n jets)
    """
    
    all_collisions = []
    for i,collision in enumerate(dataset):
        # cluster
        if i % 500 == 0:
            print("Clustering collision", i, "of", len(dataset), "...")
        sequence = cluster(collision, R=R, p=-1)
        jets = sequence.inclusive_jets()[:n]
        collision_array = np.array([[jet.pt, jet.eta, jet.phi] for jet in jets]).transpose()
        all_collisions.append(collision_array)
        
    return np.array(all_collisions)

    
def get_highest_mass_constituents(dataset, R, deltaJ = False, ncon_store = 200):
    
    """
    Takes a list (len N) of lists (len M), M unclustered particles, N collision events
    Clusters into n jets, returns the constituents for the highest MASS jet
    Returns output for the jet CLR code : 3d numpy array (N collision events, 3 jet params, n jets)
    
    deltaJ: return eta - etaJ, phi - phiJ
    """
    
    all_collisions = []
    bad_indices = []
    
    #n_constits_high_mass = []
    #n_consists_next_mass = []
    
    for i, collision in enumerate(dataset):
        # cluster
        if i % 500 == 0:
            print("Clustering collision", i, "of", len(dataset), "...")
        sequence = cluster(collision, R=R, p=-1)
        jets = sequence.inclusive_jets()
        
        jet_mass_dict = {}
        jet_nconstit_dict = {}
        # get the jet masses and num constituents
        for index, jet in enumerate(jets):
            jet_mass_dict[index] = jet.mass
            jet_nconstit_dict[index] = len(jet)
            
        # get the highest mass jets
        
        Z = [x for _,x in sorted(zip(jet_mass_dict.values(),jet_mass_dict.keys()))][::-1]
        
        high_mass_index = Z[0]
        jet_info = np.array([[jets[high_mass_index].pt], [jets[high_mass_index].eta], [jets[high_mass_index].phi]])
        next_mass_index = Z[1]
        
        # save n_consists
        #n_constits_high_mass.append(jet_nconstit_dict[high_mass_index])
        #n_consists_next_mass.append(jet_nconstit_dict[next_mass_index])
        
        jet_eta = jets[high_mass_index].eta
        jet_phi = jets[high_mass_index].phi
        
        
        # order constituents by pt
        constit_pts = [constit.pt for constit in jets[high_mass_index]]
        bad_event = False
        
        try:
            Z = [x for _,x in sorted(zip(constit_pts,jets[high_mass_index]))][::-1]
        except NotImplementedError:
            bad_event = True
            bad_indices.append(i)
            
        if not bad_event:
            
            # save the constituents for the high mass jet
            if deltaJ:
                collision_array = np.array([[constit.pt, constit.eta-jet_eta, phi_wrap(constit.phi-jet_phi)] for constit in Z]).transpose()
                

            else: 
                collision_array = np.array([[constit.pt, constit.eta, constit.phi] for constit in Z]).transpose()
            
            # add the jet info as the first element
            collision_array = np.concatenate((jet_info,collision_array), axis = 1) 
            
            # ZERO PAD
            try:
                zero_pad = np.zeros((3,ncon_store-collision_array.shape[1]))
                collision_array = np.concatenate((collision_array,zero_pad), axis = 1)
            except ValueError:
                #print("OVERFLOW: num constituents =",collision_array.shape[1])
                collision_array = collision_array[:,:ncon_store]
           
            all_collisions.append(collision_array)
            
      
    """ 
    # plot number of constituents for highest and second mass jets
    bins = np.linspace(0,150,76)
    plt.figure()
    plt.hist(n_constits_high_mass, label = "high mass jet", bins = bins, alpha = .5)
    plt.hist(n_consists_next_mass, label = "next mass jet",  bins = bins, alpha = .5)
    plt.xlabel("n_constituents")
    plt.ylabel("counts")
    plt.legend()
    plt.show()
    """
    
        
    return np.array(all_collisions), bad_indices


"""
FROM CLUSTERED PARTICLES
"""

def pandas_to_features(input_frame):
    
    dataset = input_frame.copy()
    
    labels = dataset[["label"]]
    
    """
    dataset["mjj"] = (((input_frame["pxj1"]**2+input_frame["pyj1"]**2+input_frame["pzj1"]**2+input_frame["mj1"]**2)**0.5+(input_frame["pxj2"]**2+input_frame["pyj2"]**2+input_frame["pzj2"]**2+input_frame["mj2"]**2)**0.5)**2-(input_frame["pxj1"]+input_frame["pxj2"])**2-(input_frame["pyj1"]+input_frame["pyj2"])**2-(input_frame["pzj1"]+input_frame["pzj2"])**2)**0.5/1000.
    """
    
    # we want the features for the highest mass jet ONLY
    
    
    dataset["mjHigh"] = dataset[["mj1", "mj2"]].max(axis=1)
    dataset["mjLow"] = dataset[["mj1", "mj2"]].min(axis=1)
    

    dataset["pxjHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["pxj1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["pxj2"])/1000
    
    dataset["pyjHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["pyj1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["pyj2"])/1000
    
    dataset["pzjHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["pzj1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["pzj2"])/1000
    
    dataset["tau1jHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["tau1j1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["tau1j2"])/1000
    
    dataset["tau2jHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["tau2j1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["tau2j2"])/1000
    
    dataset["tau3jHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["tau3j1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["tau3j2"])/1000
    
    
    # at the moment, we want the high-lebel features for the first jet only
    # 'pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1'

    dataset = dataset.fillna(0)
    dataset = dataset[["mjHigh","pxjHigh","pyjHigh", "pzjHigh", "tau1jHigh", "tau2jHigh", "tau3jHigh"]] 
    
    

  
    return dataset.to_numpy(), labels.to_numpy().reshape(len(labels))

def pandas_to_features_2(input_frame):
    
    dataset = input_frame.copy()
    
    labels = dataset[["label"]]
    
    """
    dataset["mjj"] = (((input_frame["pxj1"]**2+input_frame["pyj1"]**2+input_frame["pzj1"]**2+input_frame["mj1"]**2)**0.5+(input_frame["pxj2"]**2+input_frame["pyj2"]**2+input_frame["pzj2"]**2+input_frame["mj2"]**2)**0.5)**2-(input_frame["pxj1"]+input_frame["pxj2"])**2-(input_frame["pyj1"]+input_frame["pyj2"])**2-(input_frame["pzj1"]+input_frame["pzj2"])**2)**0.5/1000.
    """
    
    # we want the features for the highest mass jet ONLY
    
    
    dataset["mjHigh"] = dataset[["mj1", "mj2"]].max(axis=1)
    dataset["mjLow"] = dataset[["mj1", "mj2"]].min(axis=1)
    

    dataset["pxjHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["pxj1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["pxj2"])/1000
    
    dataset["pyjHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["pyj1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["pyj2"])/1000
    
    dataset["pzjHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["pzj1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["pzj2"])/1000
    
    dataset["pTjHigh"] = (dataset["pxjHigh"]**2+dataset["pyjHigh"]**2)**0.5
    
    dataset["tau1jHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["tau1j1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["tau1j2"])/1000
    
    dataset["tau2jHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["tau2j1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["tau2j2"])/1000
    
    dataset["tau3jHigh"] = ((dataset["mjHigh"] == dataset["mj1"])*dataset["tau3j1"]+(dataset["mjHigh"] == dataset["mj2"])*dataset["tau3j2"])/1000
    
    dataset["tau21jHigh"] = dataset["tau2jHigh"]/dataset["tau1jHigh"]
    
    
    # at the moment, we want the high-lebel features for the first jet only
    # 'pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1'

    dataset = dataset.fillna(0)
    dataset = dataset[["pTjHigh","mjHigh","tau21jHigh"]] 
    
    

  
    return dataset.to_numpy(), labels.to_numpy().reshape(len(labels))


def select_jets(high_mass_consits,high_mass_labels, pt_cut, eta_cut):

    # makes selection cuts on the jets for pt, eta
    # pt_cut = [min, max]
    # eta_cut = [min, max]
    
    high_mass_jets = high_mass_consits[:,:,0] #(n_jets, 3) 3 = pt, eta, phi
    jet_selector = np.where((high_mass_jets[:,0] >= pt_cut[0]) & (high_mass_jets[:,0]<= pt_cut[1]) & (high_mass_jets[:,1] >= eta_cut[0]) &(high_mass_jets[:,1] <= eta_cut[1]))

    return high_mass_consits[jet_selector], high_mass_labels[jet_selector]
    