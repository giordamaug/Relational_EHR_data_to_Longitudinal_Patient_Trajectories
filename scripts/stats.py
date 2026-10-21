import pandas as pd
import itertools
from scipy.stats import friedmanchisquare, wilcoxon
import statsmodels.stats.multitest as smm
import numpy as np

# ======================
# FRIEDMAN TEST
# ======================
def friedman_test(df, alpha = 0.05, debug=False):
    stat, p = friedmanchisquare(*[df[col] for col in df.columns], nan_policy='omit')

    if debug:
        print("\n=== FRIEDMAN TEST ===")
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {p:.6f}")

    if p < alpha:
        if debug: print("➡️ SIGNIFICANT global differences")
    else:
        if debug: print("➡️ No global differences")
    return stat, p

# ========================
# WILCOXON POST-HOC + HOLM
# ========================
def wilcoxon_test(df, alpha=0.05, debug=False):
    pairs = list(itertools.combinations(df.columns, 2))

    raw_p = []
    results = []

    for a, b in pairs:
        try:
            w_stat, pval = wilcoxon(df[a], df[b], zero_method="wilcox")
        except ValueError:
            w_stat, pval = np.nan, 1.0

        raw_p.append(pval)
        results.append((a, b, w_stat, pval))

    reject, p_corr, _, _ = smm.multipletests(
        raw_p, alpha=alpha, method="holm"
    )

    methods = list(df.columns)
    n = len(methods)

    p_matrix = np.ones((n, n))

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            p_matrix[i, j] = p_corr[k]
            p_matrix[j, i] = p_corr[k]
            k += 1

    return (
        pd.DataFrame(p_matrix, columns=methods, index=methods),
        reject,
        p_corr,
        results
    )

# ========================
# WILCOXON RANKING
# ========================

def wilcoxon_ranking(df, wilcoxon_reject, higher_is_better=True):
    methods = list(df.columns)
    n = len(methods)

    wins = {m: 0 for m in methods}
    losses = {m: 0 for m in methods}
    ties = {m: 0 for m in methods}

    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = methods[i]
            b = methods[j]

            if wilcoxon_reject[k]:
                mean_a = df[a].mean()
                mean_b = df[b].mean()

                if higher_is_better:
                    winner, loser = (a, b) if mean_a > mean_b else (b, a)
                else:
                    winner, loser = (a, b) if mean_a < mean_b else (b, a)

                wins[winner] += 1
                losses[loser] += 1
            else:
                ties[a] += 1
                ties[b] += 1

            k += 1

    ranking_df = pd.DataFrame({
        "mean": df.mean(),
        "wins": wins,
        "losses": losses,
        "ties": ties
    })

    ranking_df["score"] = ranking_df["wins"] - ranking_df["losses"]

    return ranking_df.sort_values(
        by=["score", "wins", "mean"],
        ascending=[False, False, not higher_is_better]
    )