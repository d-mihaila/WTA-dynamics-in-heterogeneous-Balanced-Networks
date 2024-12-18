'''big boy class creating the network'''

import nest
from params_net import pyr_param, pv_param, str_dict, w_dict # and not only
from params_input import *


class Network:

    def __init__(self):
        """Create a single cluster with excitatory and inhibitory neurons."""
        # Create excitatory neurons
        pyr_neurons = nest.Create("aeif_cond_exp", n=self.N_e, params=pyr_param)
        
        # Create inhibitory neurons
        pv_neurons = nest.Create("aeif_cond_exp", n=self.N_i, params=pv_param)
        
        # Set heterogeneous thresholds
        th_pyr = uniform_distribution(pyr_param["V_th"], std=1, num_samples=self.N_e)
        th_pv = uniform_distribution(pv_param["V_th"], std=1, num_samples=self.N_i)
        nest.SetStatus(pyr_neurons, "V_th", th_pyr)
        nest.SetStatus(pv_neurons, "V_th", th_pv)
        
        # Store cluster neurons
        self.clusters[cluster_id] = {
            "pyr": pyr_neurons,
            "pv": pv_neurons
        }


    def create_network(self):
        '''create the network'''

        # create the network
        nest.ResetKernel()
        nest.SetKernelStatus({"resolution": 0.01})

        # create clusters (loop for number of clusters with N_e and N_i etc)
        for i in range(N_clusters):
            # create the cluster
            self.create_cluster(i)


        # create the connections within the cluster 
            # as we are using triplet stdp i am not sure if the 'initialisation weight' can be employed!?
        self.create_connections()

        
    def input(self):
        # create the input

        # connect the input to each cluster 

    def WTA(self): 
        # create the connections between the clusters -- for Winner take all dynamics 
        # basically the lateral connections from excitatory to inhibitory neurons of another cluster


    def learning_and_testing(self): 
        # start the simulation time employing the learning for t time and then testing after a d delay for a x time window.

