import ctypes
import numpy as np
from pyjet import cluster
    
lib = ctypes.cdll.LoadLibrary("/global/home/users/rrmastandrea/fjcontrib-1.046/libnsub.so")


def nsub(constit):
    """
    INPUT: numpy array of size ( # jets, 4, # constituents) with zero padding containing the four-momenta (pt, px, py, pz) of the constituents

    RETURNS: numpy array of size (# jets, 2) where the first column contains Tau_2/Tau_1 and the second column contains Tau_3/Tau_2
    """
    constit = constit.astype(np.float64)
    res = np.zeros((constit.shape[0], 2))
    lib.calc_nsub(
        constit.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        res.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        constit.ctypes.shape_as(ctypes.c_uint64),
        constit.ctypes.strides_as(ctypes.c_uint64),
        res.ctypes.strides_as(ctypes.c_uint64)
    )
    return res


def convert_constits_coords(constits_phase):
    """
    INPUT: np.array (# jets, 3, # constituents) where the 1-index goes through (pT, eta, phi)
    
    OUTPUT: np.array (# jets, 4, # constituents) where the 1-index goes through (pt, px, py, pz)
    """
    const_pT = constits_phase[:,0,:]
    const_eta = constits_phase[:,1,:]
    const_phi = constits_phase[:,2,:]
    
    constits_cartesian = np.zeros((constits_phase.shape[0], 4, constits_phase.shape[2]))
    # convert to cartesian coordinates
    const_E = const_pT*np.cosh(const_eta)
    const_px = const_pT*np.cos(const_phi)
    const_py = const_pT*np.sin(const_phi)
    const_pz = const_pT*np.sinh(const_eta)

    constits_cartesian[:,0,:] = const_E
    constits_cartesian[:,1,:] = const_px
    constits_cartesian[:,2,:] = const_py
    constits_cartesian[:,3,:] = const_pz
    
    return(constits_cartesian)


def mj(constits):
    
    """
    INPUT: np.array (# jets, 3, # constituents) where the 1-index goes through (pT, eta, phi)
    
    OUTPUT: np.array containing the jet masss
    """
    jet_masses = []
    
    for event in constits:
        
        # create an array to store the collision that will be recognized by pyjet for clustering
        collision =  np.array([(event[0,i],event[1,i],event[2,i],0) for i in range(event.shape[1])],
                     np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))  
        
        sequence = cluster(collision, R=100, p=-1) # ridiculously large R as we want to include everything
        jets = sequence.inclusive_jets()
        jet_masses.append(jets[0].mass)

    return np.array(jet_masses)

def mjj(constits_1, constits_2):
    
    """
    INPUT: np.arrays (# jets, 3, # constituents) where the 1-index goes through (pT, eta, phi)
    
    OUTPUT: np.array containing mjj
    """
    mjjs = []
    
    for event_id in range(constits_1.shape[0]):
        
        
        
        # cluster jet 1
        collision_1 =  np.array([(constits_1[event_id][0,i],constits_1[event_id][1,i],constits_1[event_id][2,i],0) 
                                 for i in range(constits_1[event_id].shape[1])],
                     np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))  
        sequence_1 = cluster(collision_1, R=100, p=-1) # ridiculously large R as we want to include everything
        jet_1 = sequence_1.inclusive_jets()[0]
        
        
        # cluster jet 2
        collision_2 =  np.array([(constits_2[event_id][0,i],constits_2[event_id][1,i],constits_2[event_id][2,i],0) 
                                 for i in range(constits_2[event_id].shape[1])],
                     np.dtype([('pT', 'f8'), ('eta', 'f8'), ('phi', 'f8'), ('mass', 'f8')]))  
        sequence_2 = cluster(collision_2, R=100, p=-1) # ridiculously large R as we want to include everything
        jet_2 = sequence_2.inclusive_jets()[0]
        
        # calculate mjj   
        # mjj = sqrt((E1 + E2)^2 - (px1 + px2)^2 - (py1 + py2)^2 - (pz1 + pz2)^2)
        loc_mjj = np.sqrt((jet_1.e + jet_2.e)**2 - (jet_1.px+jet_2.px)**2 - (jet_1.py+jet_2.py)**2 - (jet_1.pz+jet_2.pz)**2)

        mjjs.append(loc_mjj)
        
        # cluster jet 2

    return np.array(mjjs)