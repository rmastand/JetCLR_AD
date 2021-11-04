import ctypes
import numpy as np

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
    
