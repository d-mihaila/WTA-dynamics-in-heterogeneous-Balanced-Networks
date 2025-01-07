import numpy as np
import matplotlib.pyplot as plt
import nest
import numpy as np
import build_network as Network
from collections import Counter
from scipy.optimize import curve_fit


'''This file is evaluating the persormance of the WTA dybamics of the more clusters'''

# 0. basic preliminary histograms per cluster just to make sure the cluster building is working.
# 1. make the overall current and the inputted current of the clusters as subplots using the previous functions from evaluate_balance and evaluate_input
# 1. visualise somehow the learning 
# 2. 
# 3. decision // inhibition of the other clusters
# 4. correlation with the input?

# just checking the code worked.

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


def run_CV_experiments(config):
    """
    Runs a series of experiments over multiple CV values,
    automatically computing and plotting success rates.
    """

    # Extract needed values from the config
    c_m_max       = config["CV"]["C_m"]
    n_increments  = config["repeats"]["CV_increments"]
    n_repeats     = config["repeats"]["per_round"]


    c_m_values = np.linspace(0, c_m_max, n_increments)  

    # Prepare 2D lists for winners
    correct_winners = [[] for _ in range(len(c_m_values))]
    trigger_winners = [[] for _ in range(len(c_m_values))]
    test_winners    = [[] for _ in range(len(c_m_values))]

    # Arrays to store the final success rates per C_m
    trigger_success = []
    test_success    = []
    both_success    = []

    # Iterate over each C_m in our linspace
    for i, c_m_val in enumerate(c_m_values):
        config["CV"]["C_m"] = c_m_val
        print(f"\n======= Running for C_m = {c_m_val} =======")

        for i in range(n_repeats):
            # Setup and run your simulation
            net = Network(config)   # <-- Adjust if your class name or init is different
            net.run_simulation()

            # Collect winners
            correct_winners[i].append(net.winner)

            # get_winner should return (trigger, test)
            trigger, test = get_winner(net)
            trigger_winners[i].append(trigger)
            test_winners[i].append(test)

            # Reset NEST for the next iteration
            nest.ResetKernel()

        # Compute success rates for this batch (this C_m)
        rate_trigger, rate_test, rate_both = success_rate(
            correct_winners[i],
            trigger_winners[i],
            test_winners[i]
        )

        # Store them for plotting or analysis
        trigger_success.append(rate_trigger)
        test_success.append(rate_test)
        both_success.append(rate_both)

    # Restore config["CV"]["C_m"] to its original max if needed
    config["CV"]["C_m"] = c_m_max

    # ==============================
    # Plot the three success curves
    # ==============================
    plt.figure(figsize=(8, 5))
    plt.plot(c_m_values, trigger_success, marker='o', label='Trigger Success (%)')
    plt.plot(c_m_values, test_success,    marker='s', label='Test Success (%)')
    plt.plot(c_m_values, both_success,    marker='^', label='Both Success (%)')
    plt.title("Success Rates vs. C_m Values")
    plt.xlabel("C_m Value")
    plt.ylabel("Success Rate (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()





# ---------------------------
def get_winner_extended(net):
    """
    Returns:
    - winner_correct: integer cluster index that was 'intended' to be correct
    - winner_trigger: integer cluster index with max excitatory spikes in trigger period
    - winner_test: integer cluster index with max excitatory spikes in test period
    - spikes_trigger: 1D array of length (N_clusters), total excitatory spikes in trigger period
    - spikes_test: 1D array of length (N_clusters), total excitatory spikes in test period
    """

    winner_correct = net.winner

    # Prepare lists
    spikes_trigger = []
    spikes_test = []

    # Time intervals
    duration = net.simulation['duration']
    delay = net.simulation['delay']
    trigger = net.simulation['trigger']
    test = net.simulation['test']

    # Define boundaries
    trigger_start = duration + delay
    trigger_end   = trigger_start + trigger
    test_start    = trigger_end
    test_end      = test_start + test

    # For each cluster, count excitatory spikes in trigger/test
    for cluster in net.clusters:
        pyr_data = nest.GetStatus(cluster.pyr_spikes, keys='events')[0]
        times = pyr_data['times']

        # Count spikes in trigger period
        trig_spikes = np.sum((times >= trigger_start) & (times < trigger_end))
        spikes_trigger.append(trig_spikes)

        # Count spikes in test period
        t_spikes = np.sum((times >= test_start) & (times < test_end))
        spikes_test.append(t_spikes)

    spikes_trigger = np.array(spikes_trigger)
    spikes_test    = np.array(spikes_test)

    # Determine winners
    winner_trigger = np.argmax(spikes_trigger)
    winner_test    = np.argmax(spikes_test)

    # Debug / logs
    print(f"Spikes in trigger period: {spikes_trigger.tolist()}")
    print(f"Spikes in test period: {spikes_test.tolist()}")
    print(f"Winner (Trigger): Cluster {winner_trigger}")
    print(f"Winner (Test): Cluster {winner_test}")
    print(f"Correct winner (input chosen cluster): Cluster {winner_correct}")

    return winner_correct, winner_trigger, winner_test, spikes_trigger, spikes_test

def summarize_round(
    round_index,
    winner_correct_list,  # shape = (per_round,) of correct winners
    winner_trigger_list,  # shape = (per_round,) of winners
    winner_test_list,     # shape = (per_round,) of winners
    all_trigger_spikes,   # shape = (per_round, N_clusters)
    all_test_spikes       # shape = (per_round, N_clusters)
):
    """
    Summarize the results for one round.
    
    - We find the cluster with the majority "wins" in trigger and test phases.
    - We also sum up total spikes across repeats for each cluster in trigger/test phases.
    - Then see which cluster is top.
    """

    
    per_round = len(winner_correct_list)
    # 1) The "majority" winners by counting how many times each cluster was the winner.
    trigger_counter = Counter(winner_trigger_list)
    test_counter    = Counter(winner_test_list)

    # which cluster had the most trigger wins overall?
    majority_trigger_cluster = max(trigger_counter, key=trigger_counter.get)
    # how many times did it win?
    majority_trigger_count = trigger_counter[majority_trigger_cluster]

    # which cluster had the most test wins overall?
    majority_test_cluster = max(test_counter, key=test_counter.get)
    majority_test_count   = test_counter[majority_test_cluster]

    # 2) The "majority" correct cluster: we assume net.winner is the same across repeats,
    #    but if it's not, you can similarly do a counter. 
    #    Typically net.winner might vary, but let's see which was correct the most. 
    correct_counter = Counter(winner_correct_list)
    majority_correct_cluster = max(correct_counter, key=correct_counter.get)
    majority_correct_count   = correct_counter[majority_correct_cluster]

    # 3) Sum up total spikes across repeats for each cluster
    sum_trigger_spikes = np.sum(all_trigger_spikes, axis=0)  # shape (N_clusters,)
    sum_test_spikes    = np.sum(all_test_spikes, axis=0)     # shape (N_clusters,)

    spike_trigger_winner = np.argmax(sum_trigger_spikes)
    spike_test_winner    = np.argmax(sum_test_spikes)

    # Print a neat table
    print(f"\n=== Round {round_index+1} Summary ===")
    print(f"  - Majority correct cluster   = {majority_correct_cluster} (count: {majority_correct_count})")
    print(f"  - Majority trigger winner    = {majority_trigger_cluster} (count: {majority_trigger_count}/{per_round})")
    print(f"  - Majority test winner       = {majority_test_cluster} (count: {majority_test_count}/{per_round})")
    print(f"  - Highest total trigger spk  = Cluster {spike_trigger_winner} (sum of {sum_trigger_spikes[spike_trigger_winner]} spikes)")
    print(f"  - Highest total test spk     = Cluster {spike_test_winner}    (sum of {sum_test_spikes[spike_test_winner]} spikes)")

    # Optionally return them for further use
    return {
        "majority_correct_cluster": majority_correct_cluster,
        "majority_trigger_cluster": majority_trigger_cluster,
        "majority_test_cluster": majority_test_cluster,
        "spike_trigger_winner": spike_trigger_winner,
        "spike_test_winner": spike_test_winner
    }