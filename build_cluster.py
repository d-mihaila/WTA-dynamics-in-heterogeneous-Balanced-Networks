'''big boy class creating the network'''

import nest
import numpy as np
import matplotlib.pyplot as plt
# from params_net import pyr_param, pv_param, str_dict, w_dict # and not only
# from params_input import *



class Cluster:
    """
    A class that builds one cluster to check the balanced state 
    of it -- with or without input, and can apply heterogeneous parameters.
    """
    def __init__(self, config):
        self.str_dict = config['structure']
        self.pyr_param = config['exc_params'] # removed the .copy() from here and next 2 lines. check if works
        self.pv_param = config['inh_params']
        self.w_dict = config['weights']
        self.CV = config['CV']
        self.simulate = config['simulate']
        self.sin_input = config['sinusoidal_input']
        self.connections = config['conn_spec']
        self.syn_ee = config['syn_spec']['ee']
        self.syn_ie = config['syn_spec']['ie']
        self.syn_ei = config['syn_spec']['ei']
        self.syn_ii = config['syn_spec']['ii']


    def build_one_cluster(self):
        # Create neurons with default parameters
        self.pyr_neurons = nest.Create("aeif_cond_exp", n=self.str_dict['N_e'], params=self.pyr_param)
        self.pv_neurons = nest.Create("aeif_cond_exp", n=self.str_dict['N_i'], params=self.pv_param)

        # Apply heterogeneous parameters if CV != 0
        for param, cv_value in self.CV.items():
            if cv_value != 0.0:
                # Determine if param belongs to excitatory or inhibitory set
                # Apply to excitatory population if the parameter is in pyr_param
                if param in self.pyr_param:
                    original_val = self.pyr_param[param]
                    heterogeneous_values = generate(original_val, cv_value, self.str_dict['N_e'])
                    nest.SetStatus(self.pyr_neurons, param, heterogeneous_values)
                    # print(f"Setting excitatory param '{param}' heterogeneous with CV={cv_value} ({cv_value*100}%): Mean={original_val}, std={cv_value*original_val}")

                # Apply to inhibitory population if the parameter is in pv_param
                if param in self.pv_param:
                    original_val = self.pv_param[param]
                    heterogeneous_values = generate(original_val, cv_value, self.str_dict['N_i'])
                    nest.SetStatus(self.pv_neurons, param, heterogeneous_values)
                    # print(f"Setting inhibitory param '{param}' heterogeneous with CV={cv_value} ({cv_value*100}%): Mean={original_val}, std={cv_value*original_val}")

        # Connect the neurons
            # NOTE: i should make a dictionary that chooses betweeen static synapse or also STDP
        nest.Connect(self.pyr_neurons, self.pyr_neurons, conn_spec= self.connections, 
                     syn_spec=self.syn_ee)
        nest.Connect(self.pv_neurons, self.pv_neurons, conn_spec=self.connections, 
                     syn_spec=self.syn_ii)
        nest.Connect(self.pyr_neurons, self.pv_neurons, conn_spec=self.connections, 
                     syn_spec=self.syn_ie)
        nest.Connect(self.pv_neurons, self.pyr_neurons, conn_spec=self.connections, 
                     syn_spec=self.syn_ei)

        # Monitors
        self.pyr_spikes = nest.Create('spike_recorder')
        self.pv_spikes = nest.Create('spike_recorder')
        nest.Connect(self.pyr_neurons, self.pyr_spikes)
        nest.Connect(self.pv_neurons, self.pv_spikes)

        self.m_e = nest.Create('multimeter', params={'record_from': ['V_m', 'w', 'g_ex', 'g_in'], 'interval': 0.1})
        self.m_i = nest.Create('multimeter', params={'record_from': ['V_m', 'w', 'g_ex', 'g_in'], 'interval': 0.1})
        nest.Connect(self.m_e, self.pyr_neurons)
        nest.Connect(self.m_i, self.pv_neurons)

        het_params = [param for param, cv in self.CV.items() if cv != 0.0]
        print(f'Network built successfully with heterogeneous parameters {het_params}.')

    # def add_input(self):
        
    #     # Extract sinusoidal input parameters from the config
    #     sin_input = self.sin_input
    #     rate = np.random.randint(sin_input['rate'][0], sin_input['rate'][1])
    #     amplitude = np.random.randint(sin_input['amplitude'][0], sin_input['amplitude'][1])
    #     frequency = np.random.randint(sin_input['frequency'][0], sin_input['frequency'][1])

    #     # Create sinusoidal poisson generator
    #     input_node = nest.Create("sinusoidal_poisson_generator", params={
    #         "rate": rate,
    #         "amplitude": amplitude,
    #         "frequency": frequency
    #     })

    #     # Connect input to the pyramidal population
    #     nest.Connect(input_node, self.pyr_neurons, {'rule': 'all_to_all'} ,syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0})

    #     # Create a parrot neuron to record the input spikes
    #     parrot = nest.Create('parrot_neuron')
    #     nest.Connect(input_node, parrot)
    #     self.input_spikes = nest.Create('spike_recorder')
    #     nest.Connect(parrot, self.input_spikes)

    #     print('Sinusoidal input added to the network.')


    def run_simulation(self):
        nest.Simulate(self.simulate['duration'])
        print('Simulation completed!')

    def plot_results(self):
        # Get excitatory and inhibitory spike data
        pyr_data = nest.GetStatus(self.pyr_spikes, keys='events')[0]
        pv_data = nest.GetStatus(self.pv_spikes, keys='events')[0]

        if self.with_input and hasattr(self, 'input_spikes'):
            input_data = nest.GetStatus(self.input_spikes, keys='events')[0]
            # Three subplots if input is present
            plt.figure(figsize=(8, 12))

            # 1. Input Spikes
            plt.subplot(3, 1, 1)
            plt.hist(input_data['times'], bins=50, color='gray', alpha=0.7)
            plt.xlabel('Time (ms)')
            plt.ylabel('Spike Count')
            plt.title('Histogram of Input Spikes')

            # 2. Pyr Spikes
            plt.subplot(3, 1, 2)
            plt.hist(pyr_data['times'], bins=50, color='blue', alpha=0.7)
            plt.xlabel('Time (ms)')
            plt.ylabel('Spike Count')
            plt.title('Histogram of Pyr Spikes')

            # 3. PV Spikes
            plt.subplot(3, 1, 3)
            plt.hist(pv_data['times'], bins=50, color='red', alpha=0.7)
            plt.xlabel('Time (ms)')
            plt.ylabel('Spike Count')
            plt.title('Histogram of PV Spikes')

        else:
            # If no input, just two subplots for Pyr and PV
            plt.figure(figsize=(8, 8))

            # Pyr Spikes
            plt.subplot(2, 1, 1)
            plt.hist(pyr_data['times'], bins=50, color='blue', alpha=0.7)
            plt.xlabel('Time (ms)')
            plt.ylabel('Spike Count')
            plt.title('Histogram of Pyr Spikes')

            # PV Spikes
            plt.subplot(2, 1, 2)
            plt.hist(pv_data['times'], bins=50, color='red', alpha=0.7)
            plt.xlabel('Time (ms)')
            plt.ylabel('Spike Count')
            plt.title('Histogram of PV Spikes')

        plt.tight_layout()
        plt.show()
        print('Histograms generated!')


def generate(target, CV, N):
    """Generate heterogeneous parameter values using a Gaussian distribution."""
    std_dev = abs(CV * target)  # CV is relative standard deviation
    return np.random.normal(loc=target, scale=std_dev, size=N)