import nest
import numpy as np
import matplotlib.pyplot as plt
from build_cluster import Cluster
import random


# this is the file where i build all the clusters 


class Network:
    """
    write later
    """

    def __init__(self, config, input_param):
        self.str_dict = config['structure']
        self.pyr_param = config['exc_params']
        self.pv_param = config['inh_params']
        self.w_dict = config['weights']
        self.CV = config['CV']
        self.simulate = config['simulate']
        self.simulation = config['simulate']
        self.connections = config['conn_spec']
        self.syn_ee = config['syn_spec']['ee']
        self.syn_ie = config['syn_spec']['ie']
        self.syn_ei = config['syn_spec']['ei']
        self.syn_ii = config['syn_spec']['ii']

        self.clusters = [] 
        self.inputs = [] 
        self.parrot_spikes = [] 
        self.trackers = [] # for multimeters in each cluster.

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

        # Create corresponding parrot neurons, one per input
            # only memorise the parrot_spikes in self later :)
        parrot_neurons = nest.Create("parrot_neuron", num_clusters)


        for i in range(num_clusters):
            # use the Cluster class function build_one_cluster to build each cluster
            cluster_config = {
                'structure': self.str_dict,
                'exc_params': self.pyr_param,
                'inh_params': self.pv_param,
                'weights': self.w_dict,
                'CV': self.CV,
                'simulate': self.simulation, # i removed the input from here. 
                'conn_spec': self.connections,
                'sinusoidal_input': self.inputs[i],
                'syn_spec': {
                    'ee': self.syn_ee,
                    'ie': self.syn_ie,
                    'ei': self.syn_ei,
                    'ii': self.syn_ii
                }
            }
            cluster = Cluster(cluster_config)
            cluster.build_one_cluster()

            nest.Connect(
                input[i],
                cluster.pyr_neurons,
                syn_spec={'synapse_model': 'static_synapse', 'weight': 1.0}
            )

            # tracking the parrot neurons per cluster -- connect it only to that one input
            nest.Connect(input[i], parrot_neurons[i], 'one_to_one')
            parrot_spikes = nest.Create('spike_recorder')
            nest.Connect(parrot_neurons[i], parrot_spikes)
            self.parrot_spikes.append(parrot_spikes)

            tracker = nest.Create('multimeter', params={'record_from': ['g_ex', 'g_in', 'V_m'], 'interval': 0.1})
            # connecting it to all the neurons in the cluster 
            nest.Connect(tracker, cluster.pyr_neurons)
            nest.Connect(tracker, cluster.pv_neurons)
            self.trackers.append(tracker)

            self.clusters.append(cluster)


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
        p_e_out = 10.0 / 100
        p_i_in = 33.0 / 100
        N_out = int(np.round(self.str_dict['N_e'] * p_e_out))
        # making up N_out random indexes 
        indexes = [random.choice(range(self.str_dict['N_e'])) for _ in range(N_out)]

        N_in = int(np.round(self.str_dict['N_i'] * p_i_in))
        print(f"FOR LATERAL INHIBITION: Connecting {N_out} excitatory neurons to {N_in} inhibitory neurons each per cluster.")

        lat_inhib_conn = {"rule": 'fixed_indegree', 'indegree': N_in}
        lat_inhib_syn = {"synapse_model": "static_synapse", "weight": self.w_dict['wli']} # here change maybe 

        j = 0
        n = self.str_dict['N_clusters']
        n = N_out * n * (n - 1)

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

                        j = j + 1

        print(f"{j}/{n} Lateral inhibition groups connected.")


    def trigger(self):

        random_index = self.input_param[-1]
        trigger_input = self.inputs[random_index]

        for cluster in self.clusters:
            nest.Connect(trigger_input, cluster.pyr_neurons, 'all_to_all')

        print(f"Trigger input has been added to the network.")
        
        trigger_time = self.simulation['trigger']
        nest.Simulate(trigger_time)

        # cut the connection after the trigger time
        for cluster in self.clusters:
            conn_trigger = nest.GetConnections(source=trigger_input, target=cluster.pyr_neurons)
            nest.SetStatus(conn_trigger, {'weight': 0.0})


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
        



def make_input(config): 
    '''function that generates the random parameters of the input 
    important to make this outside the class for reproducibility.'''

    num = config['structure']['N_clusters']

    trigger_input = np.random.randint(0, num) # index for the trigger input

    rate_bounds = config['sinusoidal_input']['rate']
    if len(rate_bounds) == 2:
        i_rate = np.random.randint(rate_bounds[0], rate_bounds[1], num)
    else:
        i_rate = np.array([rate_bounds[0]] * num)

    ampl_bounds = config['sinusoidal_input']['amplitude']
    if len(ampl_bounds) == 2:
        i_amplitude = np.random.randint(ampl_bounds[0], ampl_bounds[1], num)
    else:
        i_amplitude = np.array([ampl_bounds[0]] * num)

    freq_bounds = config['sinusoidal_input']['frequency']
    if len(freq_bounds) == 2:
        # i_freq = np.linspace(freq_bounds[0], freq_bounds[1], num)
        i_freq = np.random.randint(freq_bounds[0], freq_bounds[1], num)

    else:
        i_freq = np.array([freq_bounds[0]] * num)


    return [i_rate, i_amplitude, i_freq, trigger_input]