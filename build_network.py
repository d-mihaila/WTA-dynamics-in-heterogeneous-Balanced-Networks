import nest
import numpy as np
import random


# this is the file where i build all the clusters 


class Network:
    """
    write later
    """

    def __init__(self, config, input_param):
        self.config = config
        self.str_dict = config['structure']
        self.pyr_param = config['exc_params']
        self.pv_param = config['inh_params']
        self.w_dict = config['weights']
        self.CV = config['CV']
        self.simulate = config['simulate']
        self.simulation = config['simulate']
        self.connectivity = config['conn_params']
        self.syn_ee = config['syn_spec']['ee']
        self.syn_ie = config['syn_spec']['ie']
        self.syn_ei = config['syn_spec']['ei']
        self.syn_ii = config['syn_spec']['ii']
        self.lat_inhib = config['lat_inhib']

        self.clusters = [] 
        self.inputs = [] 
        self.parrot_spikes = [] 
        self.trackers = [] # for multimeters in each cluster.
        
        self.indexes_conn = [] # to keep track of the indexes of the connections to the input
        self.input_param = input_param

    def build_network(self):
        '''make the input, using external parameters
        and make the clusters + connect '''
        num_clusters = self.str_dict['N_clusters']

        i_rate = self.input_param[0]
        i_amplitude = self.input_param[1]
        i_freq = self.input_param[2]

        input = nest.Create(
            'sinusoidal_poisson_generator',
            n=num_clusters,
            params={
            "rate": i_rate,
            "amplitude": i_amplitude,
            "frequency": i_freq
            }
        )
        self.inputs = input # keep all inputs
        parrot_neurons = nest.Create("parrot_neuron", num_clusters)
        
        self.conn_dict = {}

        for i in range(num_clusters):

            cluster = OneCluster(self.config)
            cluster.build_one_cluster()  # sets cluster.pyr_neurons, cluster.pv_neurons, etc.

            # Connect the input to this cluster’s pyramidal neurons
            nest.Connect(
                input[i],
                cluster.pyr_neurons,
                conn_spec=self.connectivity['input'],
                syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0}
            )

            # Retrieve the actual connections
            conn_list = nest.GetConnections(source=input[i], target=cluster.pyr_neurons)

            # Extract the GIDs of the target
            connected_gids = list(set(conn_list.target))

            # Convert cluster.pyr_neurons into a Python list of GIDs
            cluster_pyr_gids = cluster.pyr_neurons.tolist()

            # For each connected GID, find its local index in cluster.pyr_neurons
            connected_idx = []
            for gid in connected_gids:
                idx = cluster_pyr_gids.index(gid)  # local position within c.pyr_neurons
                connected_idx.append(idx)

            connected_idx = sorted(set(connected_idx))

            # Store these local indices in a dictionary for later
            self.conn_dict[i] = {
                'connected_idx': connected_idx
            }

            print(f"Cluster {i} connected to input. local indices: {connected_idx}")

            # tracking the parrot neurons per cluster -- connect it only to that one input
            nest.Connect(input[i], parrot_neurons[i], 'one_to_one')
            parrot_spikes = nest.Create('spike_recorder')
            # stack them
            nest.Connect(parrot_neurons[i], parrot_spikes)
            self.parrot_spikes.append(parrot_spikes)

            tracker = nest.Create('multimeter', params={'record_from': ['g_ex', 'g_in', 'V_m'], 'interval': 0.1})
            # connecting it to all the neurons in the cluster 
            nest.Connect(tracker, cluster.pyr_neurons)
            nest.Connect(tracker, cluster.pv_neurons)

            self.clusters.append(cluster)
            self.trackers.append(tracker)

        # not sure how the spiking activity is being recorded and tracked here. it s all going into the 

        print(f"Built {num_clusters} clusters, each connected to its own sinusoidal input.")


    def stop_learning(self): # this will probably be moved in the build network function // or not cuz the learning happens without it hm.... 
        '''freeze the weights after the learning period''' 
        # during the WTA should there be plasticity from the excitatory to the inhibitory!?
        for cluster in self.clusters:
            pairs = [
            (cluster.pyr_neurons, cluster.pyr_neurons),
            (cluster.pyr_neurons, cluster.pv_neurons),
            (cluster.pv_neurons, cluster.pyr_neurons),
            (cluster.pv_neurons, cluster.pv_neurons),
            ]

            for src_neurons, tgt_neurons in pairs:
                conns = nest.GetConnections(source=src_neurons, target=tgt_neurons)
                if len(conns) == 0:
                    continue

            weights = nest.GetStatus(conns, "weight")
            sources = nest.GetStatus(conns, "source")
            targets = nest.GetStatus(conns, "target")

            nest.Disconnect(conns)
            # i dearly hope this keeps the weights intact
            nest.Connect(sources, targets, conn_spec="one_to_one", syn_spec = {"weight": weights})

        print(f"Learning has stopped and the weights have been kept.")

    def delay(self):
        '''function cuts connections of inputs and adds the delay period'''

        for i in range(len(self.inputs)):
            conn_to_input = nest.GetConnections(source=self.inputs[i])
            nest.SetStatus(conn_to_input, {'weight': 0.0})

        delay = self.simulation['delay']
        nest.Simulate(delay)

    def lateral_inhibition(self):
        '''making up the structure of the lateral inhibition'''

        # add lateral inhibition from excitatory to inhibitory neurons
        p_e_out = self.lat_inhib['p_exc_out']
        p_i_in = self.lat_inhib['p_inhib_in']
        N_out = int(np.round(self.str_dict['N_e'] * p_e_out))
        # making up N_out random indexes 
        indexes = [random.choice(range(self.str_dict['N_e'])) for _ in range(N_out)]

        N_in = int(np.round(self.str_dict['N_i'] * p_i_in))
        print(f"FOR LATERAL INHIBITION: Connecting {N_out} excitatory neurons to {N_in} inhibitory neurons each per cluster.")

        lat_inhib_conn = {"rule": self.lat_inhib['conn_rule'], 'indegree': N_in}
        lat_inhib_syn = {"synapse_model": "static_synapse", "weight": self.w_dict['wli']} # here change maybe if lateral inhibition is not enough.

        for src_cluster_idx, src_cluster in enumerate(self.clusters):
            for i in indexes:
                # For each pyramidal neuron in src_cluster...
                for tgt_cluster_idx, tgt_cluster in enumerate(self.clusters):
                    # ...connect to PV neurons in *all other* clusters only
                    if tgt_cluster_idx != src_cluster_idx:
                        nest.Connect(
                            src_cluster.pyr_neurons[i],
                            tgt_cluster.pv_neurons,
                            conn_spec=lat_inhib_conn,
                            syn_spec=lat_inhib_syn
                        )

        print("Lateral inhibition was added.")


    def trigger(self):

        random_index = self.input_param[-1]
        trigger_input = self.inputs[random_index]

        for i, cluster in enumerate(self.clusters):
            connected_idx = self.conn_dict[i]['connected_idx']
            # cluster.pyr_neurons[connected_idx] -> a NodeCollection with the subset we want

            # the node collection indexing is very particular here -- so needed the indexes from the GIDs for earlier 
            nest.Connect(
                trigger_input, 
                cluster.pyr_neurons[connected_idx],
                'all_to_all',
                syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0}
            )

        print("Trigger input has been added to the network.")

        trigger_time = self.simulation['trigger']
        nest.Simulate(trigger_time)

        
        self.winner = random_index
        print(f"The correct winner should be cluster number {self.winner}.")


    def run_simulation(self):
        """
        Runs the Nest simulation for the desired duration.
        """
        self.build_network()

        sim_time = self.simulation['duration']
        nest.Simulate(sim_time)
        print(f"Simulation completed for {sim_time} ms.")

        self.stop_learning() # comment out if learning keeps going

        # run delay period
        self.delay()
        print(f"Delay period of {self.delay} ms has passed, with no more connection to input and having kept the learnt weights.")

        # build lateral inhibition
        self.lateral_inhibition()

        # add the trigger input -- leave that function cuz it cuts connection to trigger after the trigger time
        self.trigger()
        
        # run the test period
        test = self.simulation['test']
        nest.Simulate(test)
        print(f"Test period of {test} ms has passed.")
        

######################
## Helper functions ##

def make_input(config): 
    '''function that generates the random parameters of the input 
    important to make this outside the class for reproducibility.'''

    # we make sure that the values chosen are not too close -- thus normal distribution and std deviation. 

    num = config['structure']['N_clusters']

    trigger_input = np.random.randint(0, num) # index for the trigger input

    rate_bounds = config['sinusoidal_input']['rate']
    if len(rate_bounds) == 2:
        bound_interval = rate_bounds[1] - rate_bounds[0]
        std_dev = bound_interval / (5 * num)
        dummy_values = np.linspace(rate_bounds[0], rate_bounds[1], num)
        i_rate = np.random.normal(dummy_values, std_dev)
    else:
        i_rate = np.array([rate_bounds[0]] * num)

    ampl_bounds = config['sinusoidal_input']['amplitude']
    if len(ampl_bounds) == 2:
        bound_interval = ampl_bounds[1] - ampl_bounds[0]
        std_dev = bound_interval / (5 * num)
        dummy_values = np.linspace(ampl_bounds[0], ampl_bounds[1], num)
        i_amplitude = np.random.normal(dummy_values, std_dev)
    else:
        i_amplitude = np.array([ampl_bounds[0]] * num)

    freq_bounds = config['sinusoidal_input']['frequency']
    if len(freq_bounds) == 2:
        bound_interval = freq_bounds[1] - freq_bounds[0]
        std_dev = bound_interval / (5 * num)
        dummy_values = np.linspace(freq_bounds[0], freq_bounds[1], num)
        i_freq = np.random.normal(dummy_values, std_dev)

    else:
        i_freq = np.array([freq_bounds[0]] * num)


    return [i_rate, i_amplitude, i_freq, trigger_input]


def generate(target, CV, N):
    """Generate heterogeneous parameter values using a Gaussian distribution."""
    std_dev = abs(CV * target)  # CV is relative standard deviation
    return np.random.normal(loc=target, scale=std_dev, size=N)


class OneCluster:
    def __init__(self, config):
        self.str_dict = config['structure']
        self.pyr_param = config['exc_params'] # removed the .copy() from here and next 2 lines. check if works
        self.pv_param = config['inh_params']
        self.w_dict = config['weights']
        self.CV = config['CV']
        self.simulate = config['simulate']
        self.connections = config['conn_params']['cluster']
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
        nest.Connect(self.pyr_neurons, self.pyr_neurons, conn_spec= self.connections, syn_spec=self.syn_ee)
        nest.Connect(self.pv_neurons, self.pv_neurons, conn_spec= self.connections, syn_spec=self.syn_ii)
        nest.Connect(self.pyr_neurons, self.pv_neurons, conn_spec= self.connections, syn_spec=self.syn_ie)
        nest.Connect(self.pv_neurons, self.pyr_neurons, conn_spec= self.connections, syn_spec=self.syn_ei)

        # Monitors --- maybe we dont need yet?
        self.pyr_spikes = nest.Create('spike_recorder')
        self.pv_spikes = nest.Create('spike_recorder')
        nest.Connect(self.pyr_neurons, self.pyr_spikes)
        nest.Connect(self.pv_neurons, self.pv_spikes)

        # self.m_e = nest.Create('multimeter', params={'record_from': ['V_m', 'w', 'g_ex', 'g_in'], 'interval': 0.1})
        # self.m_i = nest.Create('multimeter', params={'record_from': ['V_m', 'w', 'g_ex', 'g_in'], 'interval': 0.1})
        # nest.Connect(self.m_e, self.pyr_neurons)
        # nest.Connect(self.m_i, self.pv_neurons)

        het_params = [param for param, cv in self.CV.items() if cv != 0.0]
        print(f'Network built successfully with heterogeneous parameters {het_params}.')
