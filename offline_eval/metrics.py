# Based on https://github.com/IML-DKFZ/values/blob/main/evaluation/metrics/aurc.py

import numpy as np


def rc_curve_stats(risks: np.array, confids: np.array) -> tuple[list[float], list[float], list[float]]:
    coverages = []
    selective_risks = []
    assert (len(risks.shape) == 1 and len(confids.shape) == 1 and len(risks) == len(confids))

    n_samples = len(risks)
    idx_sorted = np.argsort(confids)

    coverage = n_samples
    error_sum = sum(risks[idx_sorted])

    coverages.append(coverage / n_samples)
    selective_risks.append(error_sum / n_samples)

    weights = []

    tmp_weight = 0
    for i in range(0, len(idx_sorted) - 1):
        coverage = coverage - 1
        error_sum = error_sum - risks[idx_sorted[i]]
        tmp_weight += 1
        if i == 0 or confids[idx_sorted[i]] != confids[idx_sorted[i - 1]]:
            coverages.append(coverage / n_samples)
            selective_risks.append(error_sum / (n_samples - 1 - i))
            weights.append(tmp_weight / n_samples)
            tmp_weight = 0

    # add a well-defined final point to the RC-curve.
    if tmp_weight > 0:
        coverages.append(0)
        selective_risks.append(selective_risks[-1])
        weights.append(tmp_weight / n_samples)

    return coverages, selective_risks, weights


def aurc(risks: np.array, confids: np.array):
    _, risks, weights = rc_curve_stats(risks, confids)
    return sum([(risks[i] + risks[i + 1]) * 0.5 * weights[i] for i in range(len(weights))])


def eaurc(risks: np.array, confids: np.array):
    """Compute normalized AURC, i.e. subtract AURC of optimal CSF (given fixed risks)."""
    n = len(risks)
    # optimal confidence sorts risk. Asencding here because we start from coverage 1/n
    selective_risks = np.sort(risks).cumsum() / np.arange(1, n + 1)
    aurc_opt = selective_risks.sum() / n
    return aurc(risks, confids) - aurc_opt
