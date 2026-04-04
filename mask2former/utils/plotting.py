# Based on https://github.com/hollance/reliability-diagrams/blob/master/reliability_diagrams.py

import numpy as np
from matplotlib import pyplot as plt


def reliability_conf_diagram(accuracies: np.ndarray, confidences: np.ndarray, counts: np.ndarray, bins: np.ndarray,
                             ece=None, ace=None, mce=None, figsize=(5, 8), dpi=72, draw_bin_importance=False,
                             title="Reliability Diagram"):
    """
    Plots reliability and confidence diagram on one figure.
    :param counts: numpy array specifying the number of items per bin
    :param accuracies: mean accuracy of prediction in each bin
    :param confidences: mean confidence for prediction in each bin
    :param bins: thresholds for each bin (usually created via np.linspace)
    :param ece: Computed ECE value. Shown on plot not None.
    :param ace: Computed ACE value. Shown on plot not None.
    :param mce: Computed MCE value. Shown on plot not None.
    :param figsize: Figure size to use for matplotlbi figure.
    :param draw_bin_importance: whether to represent how much each bin contributes to the total accuracy: False, "alpha", "widths"
    :return:
    """

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=figsize, dpi=dpi,
                           gridspec_kw={"height_ratios": [4, 1]})
    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.1)

    reliability_diagram(ax[0], confidences, accuracies, counts, bins, ece=ece, ace=ace, mce=mce,
                        draw_bin_importance=draw_bin_importance, title=title)

    avg_accuracy = np.average(accuracies, weights=counts.astype(float) / np.sum(counts))
    avg_confidence = np.average(confidences, weights=counts.astype(float) / np.sum(counts))

    # -ve counts: draw upside down
    confidence_histogram(ax[1], -counts, bins, avg_accuracy, avg_confidence, draw_averages=True)
    # Also negate the ticks for the upside-down histogram.
    new_ticks = np.abs(ax[1].get_yticks()).astype(int)
    ax[1].set_yticks(ax[1].get_yticks())
    ax[1].set_yticklabels(new_ticks)

    return fig


def confidence_histogram(ax, counts: np.ndarray, bins: np.ndarray, avg_accuracy: float = None,
                         avg_confidence: float = None, draw_averages=True, title="Examples per bin",
                         xlabel="Confidence", ylabel="Count"):
    """
    Draw confidence histogram.
    :param ax: matplotlib axes to plot on
    :param counts: numpy array specifying the number of items per bin
    :param bins: thresholds for each bin (usually created via np.linspace)
    :param avg_accuracy: Accuracy across entire dataset. Only used if draw_averages is True.
    :param avg_confidence: Confidence across entire dataset. Only used if draw_averages is True.
    :param draw_averages: Draw lines showing average dataset confidence and accuracy.
    :param title: Plot title to show.
    :param xlabel: Plot x-axis label to show.
    :param ylabel: Plot y-axis label to show.
    """
    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    ax.bar(positions, counts, width=bin_size * 0.9)

    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if draw_averages:
        acc_plt = ax.axvline(x=avg_accuracy, ls="solid", lw=3,
                             c="black", label="Accuracy")
        conf_plt = ax.axvline(x=avg_confidence, ls="dotted", lw=3,
                              c="#444", label="Avg. confidence")
        ax.legend(handles=[acc_plt, conf_plt])


def reliability_diagram(ax, confidences: np.ndarray, accuracies: np.ndarray, counts: np.ndarray, bins: np.ndarray,
                        ece: float = None, ace: float = None, mce: float = None, draw_bin_importance=False, title="Reliability Diagram",
                        xlabel="Confidence", ylabel="Expected Accuracy"):
    """
    Plots reliability diagram.
    :param ax: matplotlib axes to plot on
    :param counts: numpy array specifying the number of items per bin
    :param accuracies: mean accuracy of prediction in each bin
    :param confidences: mean confidence for prediction in each bin
    :param bins: thresholds for each bin (usually created via np.linspace)
    :param ece: Computed ECE value. Shown on plot not None.
    :param ace: Computed ACE value. Shown on plot not None.
    :param mce: Computed MCE value. Shown on plot not None.
    :param draw_bin_importance: whether to represent how much each bin contributes to the total accuracy: False, "alpha", "widths"
    :param title: Plot title to show.
    :param xlabel: Plot x-axis label to show.
    :param ylabel: Plot y-axis label to show.
    """
    bin_size = 1.0 / len(counts)
    positions = bins[:-1] + bin_size / 2.0

    widths = bin_size
    alphas = 0.3
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)

    if draw_bin_importance == "alpha":
        alphas = 0.2 + 0.8 * normalized_counts
    elif draw_bin_importance == "width":
        widths = 0.1 * bin_size + 0.9 * bin_size * normalized_counts

    colors = np.zeros((len(counts), 4))
    colors[:, 0] = 240 / 255.
    colors[:, 1] = 60 / 255.
    colors[:, 2] = 60 / 255.
    colors[:, 3] = alphas

    gap_plt = ax.bar(positions, np.abs(accuracies - confidences),
                     bottom=np.minimum(accuracies, confidences), width=widths,
                     edgecolor=colors, color=colors, linewidth=1, label="Gap")

    acc_plt = ax.bar(positions, 0, bottom=accuracies, width=widths,
                     edgecolor="black", color="black", alpha=1.0, linewidth=3,
                     label="Accuracy")

    ax.set_aspect("equal")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    display_str = ""
    if ece is not None:
        ece = ece * 100
        display_str += f"ECE={ece:.2f}"
    if ace is not None:
        ace = ace * 100
        display_str += f"\nACE={ace:.2f}"
    if mce is not None:
        mce = mce * 100
        display_str += f"\nMCE={mce:.2f}"
    if len(display_str) > 0:
        ax.text(0.98, 0.02, display_str, color="black", ha="right", va="bottom", transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(handles=[gap_plt, acc_plt])
