import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def estimate_pilot_params(df: pd.DataFrame):
    df = df[df["experiment_name"] == "trail_rounds_2"]
    out_dict = {}
    df["group_missing"] = pd.to_numeric(
        df.groupby(["session", "global_group_id", "round_number", "episode"])[
            "player_no_input"
        ].transform("sum")
    )
    df["payoff"] = (
        20
        - df["contribution"]
        - df["punishment"]
        + df["common_good"] / (4 - df["group_missing"])
    )

    dfg = (
        df.groupby(["global_group_id", "participant_code"])["payoff"]
        .mean()
        .reset_index()
    )

    md = smf.mixedlm("payoff ~ 1", dfg, groups=dfg["global_group_id"])
    mdf = md.fit(method=["lbfgs"])
    print(mdf.summary().tables[1].loc["Group Var", "Coef."])


if __name__ == "__main__":
    df = pd.read_csv("pilot_old.csv")
    estimate_pilot_params(df)
