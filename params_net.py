import numpy as np


''' File that will contain the parameters of the Excitatory and Inhibitory neuron types and the amount of neurons in a cluser + number of clusters. '''

# Excitatory Neuron Parameters
pyr_param = {
    "a": -0.8,
    "b": 65,
    "V_th": -52,
    "Delta_T": 0.8,
    "I_e": 0.0,
    "C_m": 104.0,
    "g_L": 4.3,
    "V_reset": -53,
    "tau_w": 88,
    "t_ref": 5.0,
    "V_peak": -40.0,
    "E_L": -65,
    "E_ex": 0.0,
    "E_in": -70.0,
}

# Inhibitory Neuron Parameters
pv_param = {
    "a": 1.8,
    "b": 60.0,
    "V_th": -42.0,
    "Delta_T": 3.0,
    "I_e": 0.0,
    "C_m": 59,
    "g_L": 3,
    "V_reset": -54.0,
    "tau_w": 20.0,
    "t_ref": 5.0,
    "V_peak": -40.0,
    "E_L": -62,
    "E_ex": 0.0,
    "E_in": -70.0,
}

## Number of Neurons in a Cluster
N_i = 12
N_e = N_i * 4  # 4:1 ratio of excitatory to inhibitory neurons

## weights to be initialized in the network; from C. van Vreeswijk and H. Sompolinsky (1996) paper
w_dict = {"wee": 1, "wie": 1, "wei": -2, "wii": -1.8}

## Number of Clusters
N_clusters = 2



# Setting the CV for some parameters: -- from "Brain-inspired methods for achieving robust computation in heterogeneous mixed-signal neuromorphic processing systems"
# time constants : Tau_m = C_m / g_L and overall should be 18% so i will vary both a bit (more the conductance), trying to get the overall 18% for tau_m
g_L_CV = 0.09
C_m_CV = 0.09

# refractory period
t_ref_CV = 0.08

# weight parameters.... for initial setting.... -- but all of them! 
w_CV = 0.2

# Synapse time constants from 7% to 10% depending on the type -- NMDA or AMPA
# TODO: include / specify the types of synapses then?!
syn_CV = 0.08








################### Helper functions ####################
def generate(target, CV, N):
    """
    Generate a Gaussian-distributed set of parameters with a specified mean and coefficient of variation.

    Parameters:
    - target: float, the mean (μ) of the distribution.
    - CV: float, the coefficient of variation (σ / μ).
    - N: int, the number of parameters to generate.

    Returns:
    - parameters: numpy array of generated parameters.
    """
    # Calculate the standard deviation (σ)
    std_dev = CV * target
    
    # Generate N parameters from a Gaussian distribution
    parameters = np.random.normal(loc=target, scale=std_dev, size=N)
    
    return parameters