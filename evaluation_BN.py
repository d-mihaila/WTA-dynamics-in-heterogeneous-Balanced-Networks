import numpy as np
import matplotlib.pyplot as plt
import nest

'''This file is meant to evaluate the 'balanced nature' of the clusters'''

# 2. Coefficient of Variation (CV) of the Inter-Spike-Interval (ISI) distribution
# 3. Cross-correlation of the spike trains
# 1. Excitatory-Inhibitory currents balance (short time proportionality 10ms) -- as desribed in paper "Criteria on Balance, Stability, and Excitability in Cortical Networks for Constraining Computational Models"
# 4. Weight matrixes visualisation
# 5. Weight statistics mean and with IQR


def compute_EI_balance(pyr_data, pv_data, N_e, N_i, E_ex=0.0, E_in=-70.0):
    """
    Compute and plot average excitatory and inhibitory currents for the excitatory and inhibitory populations.
    """

    # Process both excitatory and inhibitory populations
    times_e, I_ex_mean_e, I_in_mean_e = process_population(pyr_data, N_e, E_ex, E_in)
    times_i, I_ex_mean_i, I_in_mean_i = process_population(pv_data, N_i, E_ex, E_in)

    # 3rd plot: Net currents of E-pop and I-pop together + their average
    I_net_e = I_ex_mean_e + I_in_mean_e
    I_net_i = I_ex_mean_i + I_in_mean_i
    I_net_avg = (I_net_e + I_net_i) / 2.0

    # Use times_e for plotting since times_e and times_i should be identical duration and resolution
    # If not identical, consider interpolation or just plotting the overlap.
    plt.figure(figsize=(10, 6))
    plt.plot(times_e, I_net_e, 'b', label='Net Current (E-pop)')
    plt.plot(times_e, I_net_i, 'r', label='Net Current (I-pop)')
    plt.plot(times_e, I_net_avg, 'k', label='Avg Net Current (E/I-pop)')
    plt.xlabel('Time (ms)')
    plt.ylabel('Current (pA)')
    plt.title('Net Currents of E and I Populations and their Average')
    plt.legend()
    plt.show()

    return times_e, I_ex_mean_e, I_in_mean_e, I_ex_mean_i, I_in_mean_i

# helper function for the e-i current function
def process_population(data, N, E_ex, E_in):
    # Extract raw arrays
    times_raw = data['times']
    V_m_raw = data['V_m']
    g_ex_raw = data['g_ex']
    g_in_raw = data['g_in']

    # Determine the number of time steps
    total_length = len(times_raw)
    num_steps = total_length // N

    # Reshape arrays: (num_steps, N)
    # Assuming data is recorded in chronological order for all N neurons at each timestep
    V_m = V_m_raw.reshape(num_steps, N)
    g_ex = g_ex_raw.reshape(num_steps, N)
    g_in = g_in_raw.reshape(num_steps, N)
    times = times_raw.reshape(num_steps, N)
    times = times[:, 0]  # Take one column since all should be identical per timestep

    # Compute currents
    I_ex = g_ex * (E_ex - V_m)
    I_in = g_in * (E_in - V_m)

    # Average across neurons
    I_ex_mean = I_ex.mean(axis=1)
    I_in_mean = I_in.mean(axis=1)

    return times, I_ex_mean, I_in_mean


def CV_ISI(pyr_data, pv_data):
    """
    Compute the average CV(ISI) for pyramidal and PV neuron populations,
    and plot the distribution of CV(ISI) values for individual neurons 
    in each population as histograms with the average value indicated.
    """

    def compute_population_CV_ISI(times, senders):
        unique_neurons = np.unique(senders)
        neuron_CVs = []
        for neuron_id in unique_neurons:
            # Extract spike times for this neuron
            neuron_spikes = times[senders == neuron_id]
            neuron_spikes.sort()

            # Compute ISIs if at least 2 spikes
            if len(neuron_spikes) > 1:
                ISIs = np.diff(neuron_spikes)
                mean_ISI = ISIs.mean()
                std_ISI = ISIs.std(ddof=1)  # sample std
                if mean_ISI > 0:
                    CV = std_ISI / mean_ISI
                    neuron_CVs.append(CV)
        
        return np.array(neuron_CVs)

    # Extract times and senders for both populations
    pyr_times = pyr_data['times']
    pyr_senders = pyr_data['senders']
    pv_times = pv_data['times']
    pv_senders = pv_data['senders']

    # Compute CV(ISI) arrays
    pyr_CVs = compute_population_CV_ISI(pyr_times, pyr_senders)
    pv_CVs = compute_population_CV_ISI(pv_times, pv_senders)

    # Compute mean CV(ISI)
    CV_ISI_pyr = np.nanmean(pyr_CVs) if pyr_CVs.size > 0 else np.nan
    CV_ISI_pv = np.nanmean(pv_CVs) if pv_CVs.size > 0 else np.nan

    # Plot histograms
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Excitatory (Pyramidal) Population
    if pyr_CVs.size > 0:
        axs[0].hist(pyr_CVs, bins=20, color='blue', alpha=0.7, edgecolor='black')
        axs[0].axvline(CV_ISI_pyr, color='red', linestyle='--', label=f'average = {CV_ISI_pyr:.2f}')
        axs[0].set_title('CV(ISI) Distribution - Excitatory')
        axs[0].set_xlabel('CV(ISI)')
        axs[0].set_ylabel('Count')
        axs[0].legend()
    else:
        axs[0].text(0.5, 0.5, 'No valid ISIs', ha='center', va='center')
        axs[0].set_title('Excitatory Population')

    # Inhibitory (PV) Population
    if pv_CVs.size > 0:
        axs[1].hist(pv_CVs, bins=20, color='green', alpha=0.7, edgecolor='black')
        axs[1].axvline(CV_ISI_pv, color='red', linestyle='--', label=f'average = {CV_ISI_pv:.2f}')
        axs[1].set_title('CV(ISI) Distribution - Inhibitory')
        axs[1].set_xlabel('CV(ISI)')
        axs[1].set_ylabel('Count')
        axs[1].legend()
    else:
        axs[1].text(0.5, 0.5, 'No valid ISIs', ha='center', va='center')
        axs[1].set_title('Inhibitory Population')

    plt.tight_layout()
    plt.show()

    return CV_ISI_pyr, CV_ISI_pv

def compute_ccf_zero_lag(pyr_data, pv_data, bin_size=10.0, fraction=0.5):
    """
    Compute a zero-lag correlation measure for randomly selected neuron pairs.
    
    Steps:
    1. From excitatory (pyr) and inhibitory (pv) data, select 50% of neurons randomly.
    2. Create neuron pairs of three types: E-E, E-I, I-I.
    3. Compute a correlation measure at zero-lag (Pearson correlation of binned spike trains).
    4. Plot each pair as a point on a scatter plot, color-coded by pair type.
    5. Show average correlation per pair type in the legend.
    """

    # Extract neuron IDs
    pyr_ids = np.unique(pyr_data['senders'])
    pv_ids = np.unique(pv_data['senders'])

    # Randomly select fraction of each population
    num_pyr_select = int(len(pyr_ids) * fraction)
    num_pv_select = int(len(pv_ids) * fraction)

    selected_pyr = np.random.choice(pyr_ids, size=num_pyr_select, replace=False) if num_pyr_select > 0 else []
    selected_pv = np.random.choice(pv_ids, size=num_pv_select, replace=False) if num_pv_select > 0 else []

    # Combine all selected neurons for binning
    all_selected = np.concatenate([selected_pyr, selected_pv])
    is_excitatory = {nid: (nid in selected_pyr) for nid in all_selected}

    times_all = np.concatenate([pyr_data['times'], pv_data['times']])
    senders_all = np.concatenate([pyr_data['senders'], pv_data['senders']])

    if len(times_all) == 0:
        print("No spikes recorded.")
        return {}

    # Determine total simulation time
    t_min = 0.0
    t_max = times_all.max()

    num_bins = int(np.ceil((t_max - t_min) / bin_size))
    time_bins = np.linspace(t_min, t_max, num_bins+1)

    # Create a dictionary of spike trains (binned) for selected neurons
    spike_trains = {}
    for nid in all_selected:
        neuron_spikes = times_all[senders_all == nid]
        counts, _ = np.histogram(neuron_spikes, bins=time_bins)
        spike_trains[nid] = counts

    # Form pairs
    # E-E pairs
    E_neurons = selected_pyr
    I_neurons = selected_pv

    # All pairs
    EE_pairs = []
    if len(E_neurons) > 1:
        EE_pairs = [(n1, n2) for i, n1 in enumerate(E_neurons) for n2 in E_neurons[i+1:]]

    II_pairs = []
    if len(I_neurons) > 1:
        II_pairs = [(n1, n2) for i, n1 in enumerate(I_neurons) for n2 in I_neurons[i+1:]]

    EI_pairs = [(e, i) for e in E_neurons for i in I_neurons]

    # Compute zero-lag correlation: Pearson correlation between the two spike count arrays
    def zero_lag_corr(tr1, tr2):
        # Compute Pearson correlation
        if np.std(tr1) == 0 or np.std(tr2) == 0:
            return 0.0  # If one train is flat, correlation is zero
        return np.corrcoef(tr1, tr2)[0, 1]

    EE_corrs = [zero_lag_corr(spike_trains[p[0]], spike_trains[p[1]]) for p in EE_pairs]
    EI_corrs = [zero_lag_corr(spike_trains[p[0]], spike_trains[p[1]]) for p in EI_pairs]
    II_corrs = [zero_lag_corr(spike_trains[p[0]], spike_trains[p[1]]) for p in II_pairs]

    # Prepare data for plotting
    results = {
        'E-E': EE_corrs,
        'E-I': EI_corrs,
        'I-I': II_corrs
    }

    # Compute averages for legend
    EE_mean = np.mean(EE_corrs) if len(EE_corrs) > 0 else np.nan
    EI_mean = np.mean(EI_corrs) if len(EI_corrs) > 0 else np.nan
    II_mean = np.mean(II_corrs) if len(II_corrs) > 0 else np.nan

    plt.figure(figsize=(10, 6))

    current_index = 0

    # Plot E-E pairs
    if len(EE_corrs) > 0:
        x_ee = np.arange(current_index, current_index + len(EE_corrs))
        plt.scatter(x_ee, EE_corrs, c='darkblue', alpha=0.7, label='E-E')
        current_index += len(EE_corrs)
        # Plot E-E mean as a darker dot slightly to the right of the E-E pairs
        plt.scatter(current_index - 0.5, EE_mean, color='yellow', edgecolors='blue', s=100, zorder=10, linewidth=3)

    # Plot E-I pairs
    if len(EI_corrs) > 0:
        x_ei = np.arange(current_index, current_index + len(EI_corrs))
        plt.scatter(x_ei, EI_corrs, c='green', alpha=0.7, label='E-I')
        current_index += len(EI_corrs)
        # Plot E-I mean as a darker dot
        plt.scatter(current_index - 0.5, EI_mean, color='yellow', edgecolors='green', s=100, zorder=10, linewidth=3)

    # Plot I-I pairs
    if len(II_corrs) > 0:
        x_ii = np.arange(current_index, current_index + len(II_corrs))
        plt.scatter(x_ii, II_corrs, c='red', alpha=0.7, label='I-I')
        current_index += len(II_corrs)
        # Plot I-I mean as a darker dot
        plt.scatter(current_index - 0.5, II_mean, color='yellow', edgecolors='red', s=100, zorder=10, linewidth=3)

    # Update legend to show averages
    legend_entries = []
    if len(EE_corrs) > 0:
        legend_entries.append(f'E-E (mean={EE_mean:.2f})')
    if len(EI_corrs) > 0:
        legend_entries.append(f'E-I (mean={EI_mean:.2f})')
    if len(II_corrs) > 0:
        legend_entries.append(f'I-I (mean={II_mean:.2f})')

    plt.xlabel('Pair Index')
    plt.ylabel('Zero-lag Correlation')
    plt.title('Pairwise Zero-lag Correlations for Selected Neurons')
    if legend_entries:
        plt.legend(legend_entries)
    else:
        plt.text(0.5, 0.5, 'No pairs to show', ha='center', va='center', transform=plt.gca().transAxes)

    plt.tight_layout()
    plt.show()

    return results





#### plotting the connection matrices 

def makeMatrix(sources, targets, weights, num_sources=None, num_targets=None):
    """
    Returns a matrix with the weights between the source and target indices.
    """
    if num_sources is None:
        num_sources = max(sources) + 1
    if num_targets is None:
        num_targets = max(targets) + 1

    aa = np.zeros((num_sources, num_targets))

    for src, trg, wght in zip(sources, targets, weights):
        aa[src, trg] += wght

    return aa

def plotMatrix(srcs, tgts, weights, title, pos, num_sources=None, num_targets=None, src_labels=None, tgt_labels=None):
    """
    Plots weight matrix.
    """
    plt.subplot(pos)
    aa = makeMatrix(srcs, tgts, weights, num_sources, num_targets)
    plt.matshow(aa, fignum=False, aspect='auto')
    plt.xlabel("Target Neuron")
    plt.ylabel("Source Neuron")
    plt.title(title)
    plt.colorbar(fraction=0.046, pad=0.04)
    if src_labels is not None:
        plt.yticks(range(num_sources), src_labels)
    if tgt_labels is not None:
        plt.xticks(range(num_targets), tgt_labels, rotation='vertical')

# TODO here -- make it so the 'heat map' always goes from 0 to max -- not sometimes the other way around.
def conn_weights(connections, title):
    '''
    Returns all of the weight matrices of all connections.
    '''
    plt.figure(figsize=(12, 10))
    plt.suptitle(title)

    # We expect exactly 4 connections to plot
    for (conn_title, src_pop, tgt_pop, pos) in connections:
        # Get the connections between the given source and target
        conns = nest.GetConnections(source=src_pop, target=tgt_pop)

        if len(conns) > 0:
            # Extract sources, targets, and weights
            sources, targets, weights = zip(*[
                (conn.source, conn.target, conn.weight) 
                for conn in conns
            ])
            # Plot the weight matrix
            plotMatrix(sources, targets, weights, conn_title, pos)
        else:
            # If no connections, create empty arrays
            sources, targets, weights = [], [], []
            plotMatrix(sources, targets, weights, conn_title + " (No Connections)", pos)

    plt.tight_layout()
    plt.show()


def plot_weight_statistics_with_iqr(connections, fig_title="Weight Statistics"):
    """
    Plots a bar chart of average weight for each of the four connection types,
    with error bars showing the standard deviation (as before) AND a second,
    thinner error bar showing the interquartile range (Q1 to Q3).
    """

    labels = []
    means = []
    stds = []
    q1s = []
    q3s = []

    # Distinct colors for each bar (4 connections)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    # 1) Gather stats (mean, std, Q1, Q3) for each connection
    for (conn_title, src_pop, tgt_pop, _) in connections:
        conns = nest.GetConnections(source=src_pop, target=tgt_pop)

        if len(conns) == 0:
            labels.append(conn_title)
            means.append(0.0)
            stds.append(0.0)
            q1s.append(0.0)
            q3s.append(0.0)
        else:
            weights = [c.weight for c in conns]
            wmean = np.mean(weights)
            wstd = np.std(weights)
            q1 = np.percentile(weights, 25)  # 25th percentile
            q3 = np.percentile(weights, 75)  # 75th percentile

            labels.append(conn_title)
            means.append(wmean)
            stds.append(wstd)
            q1s.append(q1)
            q3s.append(q3)

    # 2) Create bar chart for mean ± std
    plt.figure(figsize=(8, 6))
    plt.suptitle(fig_title, fontsize=14)

    x_positions = np.arange(len(labels))  # e.g., [0, 1, 2, 3]
    bars = plt.bar(
        x_positions,
        means,
        yerr=stds,
        capsize=5,  
        alpha=0.7,
        color=colors[:len(labels)]
    )

    # 3) Overlay vertical lines for IQR: from Q1 to Q3
    #    plus small horizontal lines ("caps") at each end
    for x, q1, q3 in zip(x_positions, q1s, q3s):
        # Draw a thin vertical line from Q1 to Q3
        plt.vlines(x, q1, q3, color='black', linewidth=1)
        # Optional "caps" at Q1, Q3 (small horizontal ticks)
        cap_width = 0.1  # half the width of the cap
        plt.hlines(q1, x - cap_width, x + cap_width, color='black', linewidth=1)
        plt.hlines(q3, x - cap_width, x + cap_width, color='black', linewidth=1)

    # Labeling
    plt.xticks(x_positions, labels, rotation=30, ha='right')
    plt.ylabel("Weight")
    plt.title("Mean ± STD plus IQR")

    # 4) Optional custom legend for clarity
    #    (the bar chart for mean ± std doesn't automatically get a legend label)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', lw=6, label='Mean ± STD'),
        Line2D([0], [0], color='black', lw=1, label='IQR (Q1–Q3)'),
    ]
    plt.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.show()