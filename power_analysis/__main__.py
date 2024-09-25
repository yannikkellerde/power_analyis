import pandas as pd

from power_analysis.compute_data_statistics import (
    compute_data_statistics,
    compute_data_statistics_algoinst_format,
)
from power_analysis.sampling import pandas_transform, sample_groups
from power_analysis.significance_test import check_percent_significant

if __name__ == "__main__":
    dfp = pd.read_csv("data/pilot_old.csv")
    stats = compute_data_statistics_algoinst_format(dfp)
    print(stats)
    data = sample_groups(
        group_size=4,
        n_groups_per_condition=20,
        mean_diff=2,
        between_group_std=stats["between_group_std"],
        within_group_std=stats["within_group_std"],
        experiments=1000,
    )
    df = pandas_transform(data)
    new_stats = compute_data_statistics(df)
    print(new_stats)
    print(check_percent_significant(df))
