import numpy as np
import matplotlib.pyplot as plt
import nest
import numpy as np
from build_network import Network, make_input
from collections import Counter
# from scipy.optimize import curve_fit


'''This file is evaluating the persormance of the WTA dybamics of the more clusters'''

# 0. basic preliminary histograms per cluster just to make sure the cluster building is working.
# 1. make the overall current and the inputted current of the clusters as subplots using the previous functions from evaluate_balance and evaluate_input
# 1. visualise somehow the learning 
# 2. 
# 3. decision // inhibition of the other clusters
# 4. correlation with the input?

# just checking the code worked.


####################
## Visualisations ##
def plot_results_preliminary(net):
   # just plotting a histogram of all the populations (plus the parrot neuron for input) to check that the code is working

    for i, cluster in enumerate(net.clusters):
        # 1) Retrieve Parrot spikes
        parrot_data = nest.GetStatus(net.parrot_spikes[i], keys='events')[0]

        # 2) Retrieve Excitatory spikes
        pyr_data = nest.GetStatus(cluster.pyr_spikes, keys='events')[0]

        # 3) Retrieve Inhibitory spikes
        pv_data = nest.GetStatus(cluster.pv_spikes, keys='events')[0]

        # -- Plotting --
        plt.figure(figsize=(6, 10))
        plt.suptitle(f"Cluster {i}", fontsize=14)

        # Subplot 1: Parrot (Input)
        plt.subplot(3, 1, 1)
        plt.hist(parrot_data['times'], bins=50, color='gray', alpha=0.7)
        plt.title('Parrot (Input) Spikes')
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike Count')

        # Subplot 2: Pyr
        plt.subplot(3, 1, 2)
        plt.hist(pyr_data['times'], bins=50, color='blue', alpha=0.7)
        plt.title('Excitatory (Pyr) Spikes')
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike Count')

        # Subplot 3: PV
        plt.subplot(3, 1, 3)
        plt.hist(pv_data['times'], bins=50, color='red', alpha=0.7)
        plt.title('Inhibitory (PV) Spikes')
        plt.xlabel('Time (ms)')
        plt.ylabel('Spike Count')

        plt.tight_layout()
        plt.show()

        print(f"Histograms for Cluster {i + 1} generated!") # cuz dont forget, the first cluster is 0

def plot_input(net):
    # please implement it asap
    pass 


def plot_net_conductance(net):
    n_clusters = len(net.trackers)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(10, 2*n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    duration = net.simulation['duration']
    delay = net.simulation['delay']
    trigger = net.simulation['trigger']
    test = net.simulation['test']
    trigger_start = duration + delay
    trigger_end = trigger_start + trigger
    test_start = trigger_end


    def smooth_data(x, window_size=100):
        return np.convolve(x, np.ones(window_size)/window_size, mode='same')

    # Use all times instead of filtering
    for i, tracker in enumerate(net.trackers):
        data = nest.GetStatus(tracker, 'events')[0]
        t_array = np.array(data['times'])
        net_g = data['g_ex'] + data['g_in']

        smoothed_g = smooth_data(net_g, window_size=100)

        axes[i].plot(t_array, smoothed_g, color=colors[i], linewidth=1)
        axes[i].set_ylim([0, 25])
        axes[i].set_xlim([0, duration + delay + trigger + test])
        # axes[i].set_xlim([trigger_start, test_end])  # Decomment to zoom in

       # Add vertical lines at boundaries
        for b in [duration, duration+delay, duration+delay+trigger, duration+delay+trigger+test]:
            axes[i].axvline(b, linestyle=':', color='k')

    plt.tight_layout()
    plt.show()

def activity_plot(net):
    '''function that looks into the spiking activity of each cluster and plots that'''

    n_clusters = len(net.clusters)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(8, 3*n_clusters), sharex=True)
    if n_clusters == 1:
        axes = [axes]

    duration = net.simulation['duration']
    delay = net.simulation['delay']
    trigger = net.simulation['trigger']
    test = net.simulation['test']

    for i, cluster in enumerate(net.clusters):
        pyr_data = nest.GetStatus(cluster.pyr_spikes, keys='events')[0]
        times = pyr_data['times']

        # Define bins in steps of 5 ms
        if len(times) == 0:
            # If no spikes, just skip
            axes[i].set_title(f"Cluster {i}: No spikes")
            continue

        bins = np.arange(0, times.max() + 5, 5)
        counts, edges = np.histogram(times, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        half_heights = counts / 2.0

        axes[i].plot(centers, half_heights, linewidth=1)
        axes[i].set_xlim([0, duration+delay+trigger+test])
        # axes[i].set_xlim([trigger_start, test_end])
        axes[i].set_ylim([0, half_heights.max() + 5])
        axes[i].set_title(f"Cluster {i}")
        axes[i].set_xlabel("Time (ms)")
        axes[i].set_ylabel("Half Spike Count")

        # Add vertical lines at boundaries
        for b in [duration, duration+delay, duration+delay+trigger, duration+delay+trigger+test]:
            axes[i].axvline(b, linestyle=':', color='k')

    plt.tight_layout()
    plt.show()



############################
## Performance Evaluation ##
def get_winner(net):
    # in this function we will get the total amount of excitatory spikes elicited by each clusters 
    # in the trigger and test period and compare the test trigger answer to the input actually elicited. 

    winner_correct = net.winner

    spikes_trigger = []
    spikes_test = []

    duration = net.simulation['duration']
    delay = net.simulation['delay']
    trigger = net.simulation['trigger']
    test = net.simulation['test']

    for cluster in net.clusters: 
        pyr_data = nest.GetStatus(cluster.pyr_spikes, keys='events')[0]
        times = pyr_data['times']

        # Define the boundaries for trigger and test periods
        trigger_start = duration + delay
        trigger_end = trigger_start + trigger
        test_start = trigger_end
        test_end = test_start + test

        # Count spikes in trigger period
        trigger_spikes = np.sum((times >= trigger_start) & (times < trigger_end))
        spikes_trigger.append(trigger_spikes)

        # Count spikes in test period
        test_spikes = np.sum((times >= test_start) & (times < test_end))
        spikes_test.append(test_spikes)

    # just for me to sanity check the differences in spikes: 
    print(f"Spikes in trigger period: {spikes_trigger}")
    print(f"Spikes in test period: {spikes_test}")

    winner_triger = np.argmax(spikes_trigger)
    winner_test = np.argmax(spikes_test)

    print(f"Winner in trigger period: Cluster {winner_triger}")
    print(f"Winner in test period: Cluster {winner_test}")
    print(f"Correct winner: Cluster {winner_correct}")

    return winner_triger, winner_test, winner_correct


def success_rate(correct, trigger, test):
    # Ensure all inputs have the same shape (N rounds x n repeats)
    if correct.shape != trigger.shape or correct.shape != test.shape:
        print("All input matrices must have the same shape.")
        return

    N, M = correct.shape

    # Prepare accumulators
    total_trigger_matches = 0
    total_test_matches = 0
    total_both_matches = 0

    print("+-------------------------------------------------------------+")
    print("| Round | Trigger Match (%) | Test Match (%) | Both Match (%) |")
    print("+-------------------------------------------------------------+")

    # Process results per round
    for i in range(N):
        match_trigger = 0
        match_test = 0
        match_both = 0
        for j in range(M):
            if correct[i, j] == trigger[i, j]:
                match_trigger += 1
            if correct[i, j] == test[i, j]:
                match_test += 1
            if correct[i, j] == trigger[i, j] == test[i, j]:
                match_both += 1

        # Calculate round-wise success rates
        round_trigger_rate = match_trigger / M * 100 if M else 0
        round_test_rate = match_test / M * 100 if M else 0
        round_both_rate = match_both / M * 100 if M else 0

        # Update totals
        total_trigger_matches += match_trigger
        total_test_matches += match_test
        total_both_matches += match_both

        print(f"| {i+1:5d} | {round_trigger_rate:17.2f} | {round_test_rate:15.2f} | {round_both_rate:14.2f} |")

    # Calculate overall success rates
    total = N * M
    rate_trigger = total_trigger_matches / total * 100 if total else 0
    rate_test = total_test_matches / total * 100 if total else 0
    rate_both = total_both_matches / total * 100 if total else 0

    print("+-------------------------------------------------------------+")
    print(f"|Overall| {rate_trigger:17.2f} | {rate_test:15.2f} | {rate_both:14.2f} |")
    print("+-------------------------------------------------------------+")

    return rate_trigger, rate_test, rate_both


#########################
## Running Experiments ##

def run_non_hetero(config):
    '''just normal experiments, to see how the clusters perform'''
    per_round = config['repeats']['per_round']
    rounds = config['repeats']['rounds']

    correct = np.zeros((rounds, per_round))
    triggered = np.zeros((rounds, per_round))
    test = np.zeros((rounds, per_round))

    total_iterations = rounds * per_round

    for i in range(rounds):
        input_param = make_input(config)
        print(f'frequency of round {i+1}/{rounds}: {input_param[2]}')
        
        for j in range(per_round):
            net = Network(config, input_param)
            net.run_simulation()
            
            # evaluate the performance:
            winner_correct, winner_triger, winner_test = get_winner(net)
            correct[i, j] = winner_correct
            triggered[i, j] = winner_triger
            test[i, j] = winner_test

            nest.ResetKernel()

            print(f'Round {i+j+1}/{total_iterations}')

    success_rate(correct, triggered, test)


def run_CV_experiments(config):
    """
    Runs a series of experiments over multiple CV values,
    automatically computing and plotting success rates.
    """

    # Extract needed values from the config
    c_m_max       = config["CV"]["g_L"]
    n_increments  = config["repeats"]["CV_steps"]
    n_repeats     = config["repeats"]["per_round"]
    rounds        = config["repeats"]["rounds"]


    g_L_values = np.linspace(0, c_m_max, n_increments)  

    correct_winners  = [ [ [] for _ in range(rounds) ] for _ in range(n_increments) ]
    trigger_winners  = [ [ [] for _ in range(rounds) ] for _ in range(n_increments) ]
    test_winners     = [ [ [] for _ in range(rounds) ] for _ in range(n_increments) ]

    trigger_success  = [ [] for _ in range(n_increments) ]
    test_success     = [ [] for _ in range(n_increments) ]
    both_success     = [ [] for _ in range(n_increments) ]

    for c_i, g_L in enumerate(g_L_values):
        config["CV"]["g_L"] = g_L
        print(f"\n======= Running for C_m = {g_L} =======")

        # For each round, we generate one input_params and reuse it for n_repeats
        for r_i in range(rounds):
            input_params = make_input(config)

            # Run 'n_repeats' times with the same input_params
            for _ in range(n_repeats):
                net = Network(config, input_params)  
                net.run_simulation()

                # Return (trigger, test, correct)
                trigger, test, correct = get_winner(net)

                trigger_winners[c_i][r_i].append(trigger)
                test_winners[c_i][r_i].append(test)
                correct_winners[c_i][r_i].append(correct)

                # Reset NEST for the next iteration
                nest.ResetKernel()

            # Now, compute success rate for this specific round
            # We have a list of length `n_repeats` in each array, so shape is (1, n_repeats)
            rate_trigger, rate_test, rate_both = success_rate(
                np.array(trigger_winners[c_i][r_i]).reshape(1, -1),
                np.array(test_winners[c_i][r_i]).reshape(1, -1),
                np.array(correct_winners[c_i][r_i]).reshape(1, -1)
            )

            # Store them in our 2D success lists
            trigger_success[c_i].append(rate_trigger)
            test_success[c_i].append(rate_test)
            both_success[c_i].append(rate_both)

    # (Optional) restore the config["CV"]["C_m"]
    config["CV"]["C_m"] = c_m_max

    # ==============================
    # Plot the three success curves
    # ==============================
    all_cms_trigger = []
    all_rates_trigger = []
    all_cms_test = []
    all_rates_test = []
    all_cms_both = []
    all_rates_both = []

    # We'll also prepare arrays for the mean lines:
    mean_trigger_by_c = []
    mean_test_by_c = []
    mean_both_by_c = []

    for c_i, c_m_val in enumerate(g_L_values):
        round_trigger_rates = trigger_success[c_i]  # length 'rounds'
        round_test_rates    = test_success[c_i]
        round_both_rates    = both_success[c_i]

        # 1) Scatter data: each round's success rate as an individual point
        for r_i in range(rounds):
            all_cms_trigger.append(c_m_val)
            all_rates_trigger.append(round_trigger_rates[r_i])

            all_cms_test.append(c_m_val)
            all_rates_test.append(round_test_rates[r_i])

            all_cms_both.append(c_m_val)
            all_rates_both.append(round_both_rates[r_i])

        # 2) Compute means for each measure at this c_i
        mean_trigger = np.mean(round_trigger_rates) if round_trigger_rates else 0
        mean_test    = np.mean(round_test_rates)    if round_test_rates else 0
        mean_both    = np.mean(round_both_rates)    if round_both_rates else 0

        mean_trigger_by_c.append(mean_trigger)
        mean_test_by_c.append(mean_test)
        mean_both_by_c.append(mean_both)

    # ==============================
    # 5. Plot all data
    # ==============================
    plt.figure(figsize=(8, 5))

    # --- (A) Scatter all points ---
    plt.scatter(all_cms_trigger, all_rates_trigger, 
                color='blue', alpha=0.6, marker='o', label="Trigger Rates")
    plt.scatter(all_cms_test, all_rates_test, 
                color='green', alpha=0.6, marker='s', label="Test Rates")
    plt.scatter(all_cms_both, all_rates_both, 
                color='red', alpha=0.6, marker='^', label="Both Rates")

    # --- (B) Plot mean lines (piecewise linear) ---
    # These lines connect the average success rate at each c_m in order
    plt.plot(g_L_values, mean_trigger_by_c, '-o', color='blue',
             label="Mean Trigger Rate (connected)")
    plt.plot(g_L_values, mean_test_by_c, '-s', color='green',
             label="Mean Test Rate (connected)")
    plt.plot(g_L_values, mean_both_by_c, '-^', color='red',
             label="Mean Both Rate (connected)")

    plt.title("Success Rates vs. g_L Values (Scatter + Mean Lines)")
    plt.xlabel("CV Value")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()