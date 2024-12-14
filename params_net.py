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
    'tau_syn_ex': 0.2,
    'tau_syn_in': 2.0,
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
    'tau_syn_ex': 0.2,
    'tau_syn_in': 2.0,
}

str_dict = {
    "N_i": 12,
    "N_e": 'N_i' * 4,
    "N_clusters": 2,
}

## weights to be initialized in the network; from C. van Vreeswijk and H. Sompolinsky (1996) paper
w_dict = {"wee": 1, "wie": 1, "wei": -2, "wii": -1.8}


# Setting the CV for some parameters: -- from "Brain-inspired methods for achieving robust computation in heterogeneous mixed-signal neuromorphic processing systems"
# Coefficient of Variation for Heterogeneous Parameters
CV_dict = {
    "g_L": 0.09,
    "C_m": 0.09, # for total tau_m = C_m / g_L with 18% CV 
    "t_ref": 0.08, # refractory period
    "w": 0.2, # for all the weights
    "syn": 0.08, # Synapse time constants from 7% to 10% depending on the type -- NMDA or AMPA
                    # TODO: include / specify the types of synapses then?!

}


# Helper Function
def generate(target, CV, N):
    """Generate heterogeneous parameter values."""
    std_dev = CV * target
    return np.random.normal(loc=target, scale=std_dev, size=N)