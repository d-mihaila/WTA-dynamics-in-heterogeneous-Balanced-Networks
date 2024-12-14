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

# TODO neurons are not connected for these tests so the CV of the weights has to kinda be to the input!?       
w_e =  generate(w_dict['wee'], CV_dict['w'], str_dict['N_e'])
w_i =  generate(w_dict['wii'], CV_dict['w'], str_dict['N_i'])


# NOTE: we are only making a little test-drive network to see that the heterogeneiety follows the paper. 
# inputs are only some current in. 
def mini_network(pyr_param, pv_param, w_dict, str_dict, CV_dict, heterogeneity = True):
    '''make a dummy network to test out these things'''
    
    pyr = nest.Create("aeif_cond_exp", params= pyr_param, n = str_dict['N_e'])
    pv = nest.Create("aeif_cond_exp", params= pv_param, n = str_dict['N_i'])

    if heterogeneity:
        nest.SetStatus(pyr, "g_L", het_dict['gL_e'])
        nest.SetStatus(pv, "g_L", het_dict['gL_i'])
        nest.SetStatus(pyr, "C_m", het_dict['Cm_e'])
        nest.SetStatus(pv, "C_m", het_dict['Cm_i'])
        nest.SetStatus(pyr, "t_ref", het_dict['t_ref_e'])
        nest.SetStatus(pv, "t_ref", het_dict['t_ref_i'])
        nest.SetStatus(pyr, "tau_syn_ex", het_dict['tau_syn_ex_e'])
        nest.SetStatus(pyr, "tau_syn_in", het_dict['tau_syn_in_e'])
        nest.SetStatus(pv, "tau_syn_ex", het_dict['tau_syn_ex_i'])
        nest.SetStatus(pv, "tau_syn_in", het_dict['tau_syn_in_i'])


    dc = nest.Create("dc_generator", params={"amplitude": 200.0})
    
    nest.Connect(dc, pyr, syn_spec={"weight": w_e})
    nest.Connect(dc, pv, syn_spec={"weight": w_i})

    spike_pyr = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
    spike_pv = nest.Create("spike_detector", params={"withgid": True, "withtime": True})

    nest.Connect(pyr, spike_pyr)
    nest.Connect(pv, spike_pv)

    return pyr, pv, spike_pyr, spike_pv



# Figure 4 replica
def plot_histograms_and_variance(arr1, arr2, arr3):
    """
    Generate histograms for heterogeneous parameter distributions and plot the variance-mean relationship.

    Parameters:
    - arr1: array-like, values for the first parameter.
    - arr2: array-like, values for the second parameter.
    - arr3: array-like, values for the third parameter.
    """
    arrays = [arr1, arr2, arr3]
    labels = ["param1", "param2", "param3"]
    colors = ["blue", "green", "red"]

    # Prepare figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    means = []
    stds = []

    # Generate histograms for each parameter
    for i, (ax, data, label, color) in enumerate(zip(axs.flat[:-1], arrays, labels, colors)):
        ax.hist(data, bins=30, alpha=0.5, color=color, histtype='stepfilled', linewidth=1.5, label=f"$\mu$ = {np.mean(data):.2f}")
        
        # Calculate and plot the curve of best fit
        bin_heights, bin_edges = np.histogram(data, bins=30)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax.plot(bin_centers, bin_heights, color=color, linestyle='--')

        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title(f"Histogram of {label}")

        # Calculate means and stds for variance plot
        means.append(np.mean(data))
        stds.append(np.std(data))

    # Variance vs Mean plot
    axs[1, 1].plot(means, stds, 'o-', color='black')
    for i, (mean, std, label) in enumerate(zip(means, stds, labels)):
        axs[1, 1].text(mean, std, label, fontsize=12, ha='right')
    
    axs[1, 1].set_xlabel("Mean")
    axs[1, 1].set_ylabel("Standard Deviation")
    axs[1, 1].set_title("Standard Deviation vs Mean")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


# Figure 2 replica -- maybe this is not so so very important -- to implement only if i have time
        # needs a different network to be done and apply a different current just for this 
# for this, used the returns of the multimeter of course. 
def plot_time_to_first_spike(spike_pyr, spike_pv):
    # Extract spike times
    spike_times_pyr = spike_pyr.get("events")["times"]
    spike_times_pv = spike_pv.get("events")["times"]

    # Calculate time-to-first-spike for all neurons
    t_pyr = [spike_times_pyr[spike_times_pyr == gid][0] for gid in np.unique(spike_pyr.get("events")["senders"])]
    t_pv = [spike_times_pv[spike_times_pv == gid][0] for gid in np.unique(spike_pv.get("events")["senders"])]

    # Plot histograms of time-to-first-spike
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(t_pyr, bins=30, alpha=0.5, color='red', label="Pyramidal")
    ax.hist(t_pv, bins=30, alpha=0.5, color='blue', label="PV")
    ax.set_xlabel("Time-to-First-Spike (ms)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.set_title("Histogram of Time-to-First-Spike")
    plt.show()


# Figure 6 replica
# Each neuron is stimulated via its own excitatory synapse, driven by the same set of input spike sequences of increasing frequency 
#### so i need to take the network that i created but apply different inputs to it and meaure the firing rate of each neuron -- loop over frequencies

def rate_per_neuron(pyr, pv):
    # Create a dictionary to store the firing rates of each neuron
    rates = {"pyr": [], "pv": []}

    # Loop over different input frequencies
    for freq in range(1, 11):
        # Create a Poisson generator with the given frequency
        pg = nest.Create("poisson_generator", params={"rate": freq * 1000.0})
        
        # Connect the Poisson generator to the excitatory neurons
        nest.Connect(pg, pyr, syn_spec={"weight": w_e})
        nest.Connect(pg, pv, syn_spec={"weight": w_i})

        # Create spike detectors for the pyramidal and PV neurons
        spike_pyr = nest.Create("spike_detector", params={"withgid": True, "withtime": True})
        spike_pv = nest.Create("spike_detector", params={"withgid": True, "withtime": True})

        nest.Connect(pyr, spike_pyr)
        nest.Connect(pv, spike_pv)

        # Simulate the network
        nest.Simulate(1000.0)

        # Extract the spike times
        spike_times_pyr = spike_pyr.get("events")["times"]
        spike_times_pv = spike_pv.get("events")["times"]

        # Calculate the firing rates
        rate_pyr = len(spike_times_pyr) / len(pyr) * 1000.0
        rate_pv = len(spike_times_pv) / len(pv) * 1000.0

        # Store the firing rates
        rates["pyr"].append(rate_pyr)
        rates["pv"].append(rate_pv)

    # plot firing fates against input frequency
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 11), rates["pyr"], 'o-', color='red', label="Pyramidal")
    ax.plot(range(1, 11), rates["pv"], 'o-', color='blue', label="PV")
    ax.set_xlabel("Input Frequency (Hz)")
    ax.set_ylabel("Firing Rate (Hz)")
    ax.legend()
    ax.set_title("Firing Rate vs Input Frequency")
    plt.show()


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

# Figure 5 replica: Uncorrelated heterogeneity
def params_map(dictionary=het_dict):
    # Extract the values from the dictionary
    # For excitatory neurons    
    t_ref_e = dictionary['t_ref_e']
    C_m_e = dictionary['Cm_e']
    g_L_e = dictionary['gL_e']
    tau_syn_ex_e = dictionary['tau_syn_ex_e']
    tau_syn_in_e = dictionary['tau_syn_in_e']
    tau_m_e = C_m_e / g_L_e

    # For inhibitory neurons
    t_ref_i = dictionary['t_ref_i']
    C_m_i = dictionary['Cm_i']
    g_L_i = dictionary['gL_i']
    tau_syn_ex_i = dictionary['tau_syn_ex_i']
    tau_syn_in_i = dictionary['tau_syn_in_i']
    tau_m_i = C_m_i / g_L_i

    # Calculate the CV for each parameter
    cv_t_ref_e = calculate_cv(t_ref_e)
    cv_t_ref_i = calculate_cv(t_ref_i)
    cv_tau_m_e = calculate_cv(tau_m_e)
    cv_tau_m_i = calculate_cv(tau_m_i)
    cv_tau_syn_ex_e = calculate_cv(tau_syn_ex_e)
    cv_tau_syn_ex_i = calculate_cv(tau_syn_ex_i)
    cv_tau_syn_in_e = calculate_cv(tau_syn_in_e)
    cv_tau_syn_in_i = calculate_cv(tau_syn_in_i)

    # Prepare figure with subplots
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the spatial distribution of the parameters as heatmaps
    params = [
        (t_ref_e, "Refractory Period (ms)", cv_t_ref_e),
        (tau_m_e, "Membrane Time Constant (ms)", cv_tau_m_e),
        (tau_syn_ex_e, "Excitatory Synapse Time Constant (ms)", cv_tau_syn_ex_e),
        (t_ref_i, "Refractory Period (ms)", cv_t_ref_i),
        (tau_m_i, "Membrane Time Constant (ms)", cv_tau_m_i),
        (tau_syn_in_i, "Inhibitory Synapse Time Constant (ms)", cv_tau_syn_in_i)
    ]

    for i, (data, label, cv) in enumerate(params):
        ax = axs[i // 3, i % 3]
        heatmap, xedges, yedges = np.histogram2d(np.arange(len(data)), data, bins=30)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        cax = ax.imshow(heatmap.T, extent=extent, origin='lower', aspect='auto', cmap='viridis')
        fig.colorbar(cax, ax=ax)
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel(label)
        ax.set_title(f"Histogram of {label} (CV = {cv:.2f})")

    plt.tight_layout()
    plt.show()