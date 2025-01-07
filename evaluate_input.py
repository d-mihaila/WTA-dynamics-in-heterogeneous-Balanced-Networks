import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import nest

# functions that help visualise the input
# also showing the sine curve of best fit and keeps it as information, and another function reconstructs the sine curve -- useful for later
# later will be a complexity measuring function too -- so we see what level of complexity the input has and how well it can be kept

# TODO much later: have superimposed sine curves. 


# first visualise 
def plot_histograms_of_inputs(
    num_inputs,
    parrot_ids,
    parrot_spikes,
    duration=1000.0,
    bin_size=10.0,
    title_prefix="Input"
):
    def sine_func(t, A, f, phi, offset):
        return A * np.sin(2.0 * np.pi * f * t + phi) + offset

    def guess_sine_params(t, y):
        offset_guess = np.mean(y)
        y_zeromean = y - offset_guess
        dt = (t[-1] - t[0]) / (len(t) - 1) if len(t) > 1 else 1.0
        freqs = np.fft.rfftfreq(len(y), d=dt)
        Y = np.fft.rfft(y_zeromean)
        i_max = np.argmax(np.abs(Y[1:])) + 1
        f_guess = freqs[i_max]
        A_guess = 2.0 * np.abs(Y[i_max]) / len(y)
        phi_guess = np.angle(Y[i_max])
        return A_guess, f_guess, phi_guess, offset_guess

    all_events = parrot_spikes.get("events")
    all_times = all_events["times"]
    all_senders = all_events["senders"]
    fig, axes = plt.subplots(nrows=num_inputs, ncols=1, figsize=(10, 2.5 * num_inputs), sharex=True)
    if num_inputs == 1:
        axes = [axes]

    bin_edges = np.arange(0.0, duration + bin_size, bin_size)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fit_info = {}

    for i in range(num_inputs):
        pid = parrot_ids[i]
        spike_times_i = all_times[all_senders == pid]
        hist_i, _ = np.histogram(spike_times_i, bins=bin_edges)

        axes[i].step(bin_centers, hist_i, where='mid', color=f'C{i}', label=f"{title_prefix} {i}")

        if hist_i.sum() > 0:
            p0 = guess_sine_params(bin_centers, hist_i)
            try:
                popt, _ = curve_fit(sine_func, bin_centers, hist_i, p0=p0, maxfev=50000)
                x_fit = np.linspace(0, duration, 1000)
                y_fit = sine_func(x_fit, *popt)
                axes[i].plot(x_fit, y_fit, 'r--', label="Sine fit")
                fit_info[i] = dict(params=popt, x_fit=x_fit, y_fit=y_fit)
            except RuntimeError:
                fit_info[i] = None
        else:
            fit_info[i] = None

        axes[i].set_ylabel("Spike Count")
        axes[i].legend(loc="best")

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()
    return fit_info

# reconstruct the sine curves (for later checking accuracy)
def plot_fitted_sine_curves(
    fit_info,
    rates,
    amplitudes,
    frequencies,
):
    num_inputs = len(fit_info)
    fig, axes = plt.subplots(nrows=num_inputs, ncols=1, figsize=(8, 3 * num_inputs), sharex=True)

    # y_limits = [np.min(amplitudes), np.max(amplitudes)]

    if num_inputs == 1:
        axes = [axes]

    for i in range(num_inputs):
        ax = axes[i]
        curve_data = fit_info[i]
        color = f"C{i % 10}"  # Pick a color from the default color cycle

        if curve_data is not None:
            x_fit = curve_data["x_fit"]
            y_fit = curve_data["y_fit"]
            label_str = (f"rate: {rates[i]}\n"
                         f"ampl: {amplitudes[i]}\n"
                         f"freq: {frequencies[i]}")

            ax.plot(x_fit, y_fit, color=color, label=label_str)
            ax.set_ylabel(f"Cluster {i} input")
            ax.legend(loc="upper right")
        else:
            ax.text(0.5, 0.5, "No fit available", ha="center", va="center", fontsize=12)
            ax.set_ylabel("No data")

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    plt.show()
