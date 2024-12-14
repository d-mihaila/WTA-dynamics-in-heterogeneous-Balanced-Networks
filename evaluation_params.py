import numpy as np
import matplotlib.pyplot as plt
import nest
from params_net import *

'''In this file we have the functions to evaluate the variance of the parameters such as to replicate Fig. 6 of "Brain-inspired methods for achieving robust computation in heterogeneous mixed-signal neuromorphic processing systems"'''

# We shall have the benchmarks of the Coefficient of Variation of:
# 1. time-to-first-spike (9%) Fig 2
# 2. count time (s) and weight (V) (10 - 20%) Fig 2
# 3. neuron time constant (18%) Fig 4
# 4. refractory period (8%) Fig 4
# 5. synapse time constant (7 - 10 % (dep on NMDA or AMPA)) Fig 4
# 6. weight parameter (14 - 30 %) Fig 4

# in-between check of uncorrelated heterogeneiety is Fig 5

# Final check is Fig 6 with firing rates of 16 neurons from the same core with same parameters, but ofc heterogeneous


## make the g_L, C_m, t_ref, w, syn heterogeneous accorfing to the coefficient of variation
gL_e =  generate( pyr_param['g_L'],  CV_dict['g_L'],  str_dict['N_e'])
gL_i =  generate( pv_param['g_L'],  CV_dict['g_L'],  str_dict['N_i'])
Cm_e =  generate( pyr_param['C_m'],  CV_dict['C_m'],  str_dict['N_e'])
Cm_i =  generate( pv_param['C_m'],  CV_dict['C_m'],  str_dict['N_i'])
t_ref_e =  generate( pyr_param['t_ref'],  CV_dict['t_ref'],  str_dict['N_e'])
t_ref_i =  generate( pv_param['t_ref'],  CV_dict['t_ref'],  str_dict['N_i'])
tau_syn_ex_e =  generate( pyr_param['tau_syn_ex'],  CV_dict['tau_syn_ex'],  str_dict['N_e'])
tau_syn_in_e =  generate( pyr_param['tau_syn_in'],  CV_dict['tau_syn_in'],  str_dict['N_e'])
tau_syn_ex_i =  generate( pv_param['tau_syn_ex'],  CV_dict['tau_syn_ex'],  str_dict['N_i'])
tau_syn_in_i =  generate( pv_param['tau_syn_in'],  CV_dict['tau_syn_in'],  str_dict['N_i'])

het_dict = {
    "gL_e": gL_e,
    "gL_i": gL_i,
    "Cm_e": Cm_e,
    "Cm_i": Cm_i,
    "t_ref_e": t_ref_e,
    "t_ref_i": t_ref_i,
    "tau_syn_ex_e": tau_syn_ex_e,
    "tau_syn_in_e": tau_syn_in_e,
    "tau_syn_ex_i": tau_syn_ex_i,
    "tau_syn_in_i": tau_syn_in_i
}


# NOTE: we are only making a little test-drive network to see that the heterogeneiety follows the paper. 
# inputs are only some current in. 
def mini_network(pyr_param, pv_param, w_dict, str_dict, CV_dict, heterogeneity = True):
    '''make a dummy network to test out these things'''
    
    pyr = nest.Create("aeif_cond_exp", params= pyr_param, n = str_dict['N_e'])
    pv = nest.Create("aeif_cond_exp", params= pv_param, n = str_dict['N_i'])

    if heterogeneity:
        ## make the g_L, C_m, t_ref, w, syn heterogeneous accorfing to the coefficient of variation
        gL_e =  generate(pyr_param['g_L'], CV_dict['g_L'], str_dict['N_e'])
        gL_i =  generate(pv_param['g_L'], CV_dict['g_L'], str_dict['N_i'])
        Cm_e =  generate(pyr_param['C_m'], CV_dict['C_m'], str_dict['N_e'])
        Cm_i =  generate(pv_param['C_m'], CV_dict['C_m'], str_dict['N_i'])
        t_ref_e =  generate(pyr_param['t_ref'], CV_dict['t_ref'], str_dict['N_e'])
        t_ref_i =  generate(pv_param['t_ref'], CV_dict['t_ref'], str_dict['N_i'])
        tau_syn_ex_e =  generate(pyr_param['tau_syn_ex'], CV_dict['tau_syn_ex'], str_dict['N_e'])
        tau_syn_in_e =  generate(pyr_param['tau_syn_in'], CV_dict['tau_syn_in'], str_dict['N_e'])
        tau_syn_ex_i =  generate(pv_param['tau_syn_ex'], CV_dict['tau_syn_ex'], str_dict['N_i'])
        tau_syn_in_i =  generate(pv_param['tau_syn_in'], CV_dict['tau_syn_in'], str_dict['N_i'])

        # TODO neurons are not connected for these tests so the CV of the weights has to kinda be to the input!?       
        w_e =  generate(w_dict['wee'], CV_dict['w'], str_dict['N_e'])
        w_i =  generate(w_dict['wii'], CV_dict['w'], str_dict['N_i'])

        nest.SetStatus(pyr, "g_L", gL_e)
        nest.SetStatus(pv, "g_L", gL_i)
        nest.SetStatus(pyr, "C_m", Cm_e)
        nest.SetStatus(pv, "C_m", Cm_i)
        nest.SetStatus(pyr, "t_ref", t_ref_e)
        nest.SetStatus(pv, "t_ref", t_ref_i)
        nest.SetStatus(pyr, "tau_syn_ex", tau_syn_ex_e)
        nest.SetStatus(pyr, "tau_syn_in", tau_syn_in_e)
        nest.SetStatus(pv, "tau_syn_ex", tau_syn_ex_i)
        nest.SetStatus(pv, "tau_syn_in", tau_syn_in_i)


    dc = nest.Create("dc_generator", params={"amplitude": 200.0})
    
    nest.Connect(dc, pyr, syn_spec={"weight": w_e})
    nest.Connect(dc, pv, syn_spec={"weight": w_i})

    conn = nest.GetConnections()

    multi_pyr = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})
    multi_pv = nest.Create("multimeter", params={"record_from": ["V_m"], "interval": 0.1})

    nest.Connect(multi_pyr, pyr)
    nest.Connect(multi_pv, pv)

    return conn, multi_pyr, multi_pv



# Figure 4 replica
def plot_histograms_and_variance(arrays, labels, colors, parameter_names):
    """
    Generate histograms for heterogeneous parameter distributions and plot the variance-mean relationship.

    Parameters:
    - arrays: list of list of 4 arrays, each containing values for one parameter.
    - labels: list of strings for each parameter (e.g., ["tau_ref", "w", "tau_mem"]).
    - colors: list of colors corresponding to each parameter for the plots.
    - parameter_names: list of strings for parameter descriptions for histograms.
    """
    num_parameters = len(arrays)

    # Prepare figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    means = []
    stds = []

    # Generate histograms for each parameter
    for i, (ax, data, label, color, param_name) in enumerate(zip(axs.flat[:-1], arrays, labels, colors, parameter_names)):
        for values, linestyle in zip(data, ['-', '--', '-.', ':']):
            ax.hist(values, bins=30, alpha=0.5, color=color, histtype='stepfilled', linewidth=1.5, linestyle=linestyle, label=f"$\mu$ = {np.mean(values):.2f} ms")
        
        ax.set_xlabel(param_name)
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"Histogram of {label}")

        # Calculate means and stds for variance plot
        means.extend([np.mean(values) for values in data])
        stds.extend([np.std(values) for values in data])

    # Variance vs Mean plot
    axs[1, 1].plot(means, stds, 'o-', color='black')
   


# Figure 2 replica -- maybe this is not so so very important -- to implement only if i have time
        # needs a different network to be done and apply a different current just for this 



# Figure 6 replica




# first, most basic test of the CV for heterogeneous paramters
def calculate_cv(values):
    """
    Calculate the coefficient of variation (CV) for a given array of values.

    Parameters:
    - values: array-like, the input values to compute the CV.

    Returns:
    - cv: float, the coefficient of variation (CV = std / mean).
    """
    if len(values) == 0:
        raise ValueError("The input array is empty. Cannot calculate CV.")
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    if mean_val == 0:
        raise ValueError("Mean of the input array is zero. CV is undefined.")
    
    cv = std_val / mean_val
    return cv

def calculate_cv_for_parameters(dict = het_dict):
    # here we plot a table of all the CVs for the parameters
    print(f"{'Parameter':<20} {'CV':<10}")
    print("-" * 30)
    for key, values in dict.items():
        cv = calculate_cv(values)
        print(f"{key:<20} {cv:<10.4f}")

