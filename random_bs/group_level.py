import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm, trange


def mixed_model(
    between_group_variance,
    within_group_variance,
    group_size,
    mean_diff,
    two_tailed=True,
    alpha=0.05,
    power=0.8,
):
    r"""
    Let \sigma_b^2 be the between-group variance, \sigma_w^2 be the within-group variance, and g be the group size.
    Let \alpha_j be the normally distributed random effect for the j-th group with variance \sigma_b^2.
    Let \epsilon_ij be the normally distributed random effect for each data sample with variance \sigma_w^2.
    Let \mu be the mean of the data samples and \delta be the mean difference between the groups. Let \beta be the power.
    Then the i-th data sample in the j-th group is given by:
    Y_ij = \mu + \alpha_j + \epsilon_ij

    The population intra-class correlation is then given by:
    ICC = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2}
    """

    # The population intra-class correlation is then given by:
    # ICC = \frac{\sigma_b^2}{\sigma_b^2 + \sigma_w^2}
    ICC = between_group_variance / (between_group_variance + within_group_variance)

    # The design effect is then given by:
    # DE = 1 + (g - 1) * ICC
    DE = 1 + (group_size - 1) * ICC

    # Compute Z score for alpha and power
    # Z = Z_{1-\alpha/2} + Z_{\beta}
    Z_alpha = (
        scipy.stats.norm.ppf(1 - alpha / 2)
        if two_tailed
        else scipy.stats.norm.ppf(1 - alpha)
    )
    Z_power = scipy.stats.norm.ppf(power)
    Z = Z_alpha + Z_power

    # Sample size formula for cluster-randomized trials
    # n = \frac{Z^2}{DE}
