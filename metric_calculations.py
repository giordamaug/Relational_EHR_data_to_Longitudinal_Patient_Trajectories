#!/usr/bin/env python
# coding: utf-8

### Set the experiment to process results for

from joblib import Parallel, delayed, parallel_config
from scripts.utils import highlight_best_values
import argparse
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


DEFAULT_METHODS = [
    ["STATIC"],
    ["STATIC", "BINARY"],
    ["STATIC", "FREQUENCY"],
    ["STATIC", "EVENT-CNT"],
    ["STATIC", "TF-IDF"],
    ["STATIC", "BOW"],
    ["STATIC", "BiPadLSTM"],
    ["STATIC", "Med2Vec"],
    ["STATIC", "Dipole"],
    ["STATIC", "CEHR-BERT"],
    ["STATIC", "DOME"],
    ["STATIC", "GRU"],
]

DEFAULT_METRICS = [
    "AUC", "AUPRC", "F1", "PPV", "Sensitivity",
    "Specificity", "NPV", "MCC", "Accuracy", "Brier",
    "Intercept", "Slope",
]

def parse_method(value):
    parts = value.split("+")
    if any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            "Use names like STATIC o STATIC+BINARY"
        )
    return [part.strip() for part in parts]


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Value must be greater than 0")
    return number


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validation analysis of results and metric computation."
    )

    parser.add_argument(
        "--dataset",
        default="INA",
        help="Dataset name (default: INA)",
    )
    parser.add_argument(
        "--iter",
        type=positive_int,
        default=10,
        help="Number of repetition of cross-validation (default: 10)",
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        dest="results_dir",
        default="experiment_main",
        help="result folder (default: experiment_icd)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        type=parse_method,
        default=DEFAULT_METHODS,
        help="Methods to consider in experiment (example: STATIC STATIC+BINARY)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=DEFAULT_METRICS,
        default=DEFAULT_METRICS,
        help="Metrics to calculate (default: all)",
    )
    parser.add_argument(
        "--save-summary",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--to-latex",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser.parse_args()

def load_metric_files(args):
    import pickle
    # Caricamento dei risultati
    results = {"+".join(m): {} for m in args.methods}

    for i in range(args.iter):
        for m in args.methods:
            meth = "+".join(m)
            path = (
                f"{args.results_dir}/results_{args.dataset}_"
                f"{'_'.join(m)}_ITER{i}.pkl"
            )
            with open(path, "rb") as f:
                results[meth][i] = pickle.load(f)
    return results

def metric_values_with_calibration(y, p, threshold=0.30):
    # Le metriche già presenti, nello stesso ordine
    values = metric_values(y, p, threshold)

    # Regressione di calibrazione: intercept e slope stimati insieme
    z = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)

    calibration_model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        max_iter=1000,
    )
    calibration_model.fit(z, y)

    return np.concatenate([
        values,
        [
            calibration_model.intercept_[0],
            calibration_model.coef_[0, 0],
        ],
    ])
    
def metric_values(y, p, threshold=0.30):
    """Metrics in a repetition, one value per subject."""
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, pred, labels=[0, 1]
    ).ravel()

    return np.array([
        roc_auc_score(y, p),
        average_precision_score(y, p),
        f1_score(y, pred, zero_division=0),
        precision_score(y, pred, zero_division=0),
        tp / (tp + fn),                         # Sensitivity
        tn / (tn + fp),                         # Specificity
        tn / (tn + fn) if (tn + fn) else 0.0,  # NPV
        matthews_corrcoef(y, pred),
        accuracy_score(y, pred),
        brier_score_loss(y, p),
    ])


def align_oof_runs(runs):
    """
    returns y (subjects,) e P (subjects, repetitions).
    """
    ids = pd.Index(runs[0]["y_idx"])
    if not ids.is_unique:
        raise ValueError("ID duplicati nella prima ripetizione")

    y = pd.Series(
        runs[0]["y_true_oof"], index=ids
    ).to_numpy()

    P = np.empty((len(ids), len(runs)), dtype=float)

    for r, run in enumerate(runs):
        run_ids = pd.Index(run["y_idx"])

        if (
            not run_ids.is_unique
            or len(run_ids) != len(ids)
            or not run_ids.difference(ids).empty
        ):
            raise ValueError(f"ID do not match, repetition {r}")

        y_r = pd.Series(
            run["y_true_oof"], index=run_ids
        ).reindex(ids).to_numpy()

        if not np.array_equal(y_r, y):
            raise ValueError(f"Results not coherent, repetition {r}")

        P[:, r] = pd.Series(
            run["y_prob_oof"], index=run_ids
        ).reindex(ids).to_numpy()

    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("Binary results expected 0/1 in both classes")
    if not np.isfinite(P).all() or ((P < 0) | (P > 1)).any():
        raise ValueError("Wrong probabilities")

    return y, P


def mean_calibration_curve(y, P, n_bins=10):
    """Average repetition-specific calibration curves using fixed bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    predicted = np.full((P.shape[1], n_bins), np.nan)
    observed = np.full((P.shape[1], n_bins), np.nan)

    for r in range(P.shape[1]):
        p = P[:, r]
        bin_ids = np.digitize(p, edges[1:-1])

        for b in range(n_bins):
            mask = bin_ids == b
            if mask.any():
                predicted[r, b] = p[mask].mean()
                observed[r, b] = y[mask].mean()

    # Empty bins do not contribute to the average
    valid_runs = np.isfinite(observed).sum(axis=0)

    def average_bins(values):
        return np.divide(
            np.nansum(values, axis=0),
            valid_runs,
            out=np.full(n_bins, np.nan),
            where=valid_runs > 0,
        )

    return (
        average_bins(predicted),
        average_bins(observed),
        valid_runs,
    )

def summarize_repeated_cv(
    runs,
    metrics_names,
    threshold=0.30,
    n_boot=1000,
    seed=42,
    n_bins=10,
):
    if n_bins < 1 or n_boot < 1:
        raise ValueError("n_bins and n_boot must be positive.")

    y, P = align_oof_runs(runs)
    rng = np.random.default_rng(seed)

    # Metrics for each CV repetition
    per_run = np.vstack([
        metric_values_with_calibration(y, P[:, r], threshold)
        for r in range(P.shape[1])
    ])

    if per_run.shape[1] != len(metrics_names):
        raise ValueError(
            f"Computed {per_run.shape[1]} metrics, "
            f"but received {len(metrics_names)} names."
        )

    mean = per_run.mean(axis=0)
    std = (
        per_run.std(axis=0, ddof=1)
        if P.shape[1] > 1
        else np.full(len(metrics_names), np.nan)
    )

    # Average calibration curve across repetitions
    predicted, observed, valid_runs = mean_calibration_curve(
        y, P, n_bins
    )

    boot_means = []
    boot_curves = []

    with tqdm(
        total=n_boot,
        desc="Bootstrap",
        unit="sample",
    ) as pbar:

        while len(boot_means) < n_boot:
            idx = rng.integers(0, len(y), len(y))
            y_b = y[idx]

            if np.unique(y_b).size < 2:
                continue

            P_b = P[idx, :]

            scores_b = np.vstack([
                metric_values_with_calibration(
                    y_b, P_b[:, r], threshold
                )
                for r in range(P_b.shape[1])
            ])

            boot_means.append(scores_b.mean(axis=0))

            # Recalculate the average curve on the same bootstrap sample
            _, observed_b, _ = mean_calibration_curve(
                y_b, P_b, n_bins
            )
            boot_curves.append(observed_b)

            pbar.update(1)

    ci_low, ci_high = np.percentile(
        np.asarray(boot_means), [2.5, 97.5], axis=0
    )

    summary = pd.DataFrame({
        "mean": mean,
        "std_between_runs": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }, index=metrics_names)

    per_run = pd.DataFrame(
        per_run,
        columns=metrics_names,
        index=pd.Index(range(len(runs)), name="iteration"),
    )

    # Pointwise confidence intervals for the observed frequencies
    boot_curves = np.asarray(boot_curves)
    curve_low = np.full(n_bins, np.nan)
    curve_high = np.full(n_bins, np.nan)
    valid_bootstraps = np.isfinite(boot_curves).sum(axis=0)

    for b in range(n_bins):
        values = boot_curves[:, b]
        values = values[np.isfinite(values)]

        # Omit intervals for bins frequently empty in bootstrap samples
        if len(values) >= 0.95 * n_boot:
            curve_low[b], curve_high[b] = np.percentile(
                values, [2.5, 97.5]
            )

    edges = np.linspace(0, 1, n_bins + 1)
    calibration = pd.DataFrame({
        "bin_left": edges[:-1],
        "bin_right": edges[1:],
        "prob_pred": predicted,
        "prob_true": observed,
        "ci95_low": curve_low,
        "ci95_high": curve_high,
        "valid_runs": valid_runs,
        "valid_bootstraps": valid_bootstraps,
    })

    return summary, per_run, calibration

def summarize_repeated_cv_old(
    runs, metrics_names, threshold=0.30, n_boot=1000, seed=42
):
    y, P = align_oof_runs(runs)
    rng = np.random.default_rng(seed)

    # One metric row for each CV repetition
    per_run = np.vstack([
        metric_values(y, P[:, r], threshold)
        for r in range(P.shape[1])
    ])

    mean = per_run.mean(axis=0)
    std = (
        per_run.std(axis=0, ddof=1)
        if P.shape[1] > 1
        else np.full(len(metrics_names), np.nan)
    )

    # Bootstrap: subject resampling, keeping joined predictions in all repetitions
    boot_means = []

    with tqdm(total=n_boot,initial=len(boot_means),desc="Bootstrap",unit="sample",) as pbar:
        while len(boot_means) < n_boot:
            idx = rng.integers(0, len(y), len(y))
            y_b = y[idx]

            # AUC e AUPRC require both classes
            if np.unique(y_b).size < 2:
                continue

            P_b = P[idx, :]
            scores_b = np.vstack([
                metric_values(y_b, P_b[:, r], threshold)
                for r in range(P_b.shape[1])
            ])
            boot_means.append(scores_b.mean(axis=0))
            pbar.update(1)
        
    ci_low, ci_high = np.percentile(
        np.asarray(boot_means), [2.5, 97.5], axis=0
    )

    summary = pd.DataFrame({
        "mean": mean,
        "std_between_runs": std,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
    }, index=metrics_names)

    per_run = pd.DataFrame(
        per_run,
        columns=metrics_names,
        index=pd.Index(range(len(runs)), name="iteration"),
    )

    return summary, per_run


def format_results_table(
    summaries,
    results,
    methods,
    n_iter,
    metric_names,
    threshold=0.30,
    output_path=None,
):
    """Format metric summaries and the mean confusion matrix for each model."""
    rows = []

    for method in methods:
        model_name = "+".join(method)
        summary = summaries[model_name]

        row = {"model": model_name}

        for metric in metric_names:
            stats = summary.loc[metric]
            row[metric] = (
                f"{stats['mean']:.3f}±{stats['std_between_runs']:.3f} "
                f"[{stats['ci95_low']:.3f}-{stats['ci95_high']:.3f}]"
            )

        # Align subjects across repetitions
        runs = [results[model_name][i] for i in range(n_iter)]
        y, probabilities = align_oof_runs(runs)

        # Average confusion matrices across repetitions
        matrices = []
        for i in range(n_iter):
            predicted = (probabilities[:, i] >= threshold).astype(int)

            # Requested layout: [[TN FP] [FN TP]]
            matrix = confusion_matrix(
                y, predicted, labels=[0, 1]
            )
            matrices.append(matrix)

        mean_matrix = np.mean(matrices, axis=0)

        row["Confusion matrix"] = (
            f"[[{mean_matrix[0, 0]:.1f} {mean_matrix[0, 1]:.1f}] "
            f"[{mean_matrix[1, 0]:.1f} {mean_matrix[1, 1]:.1f}]]"
        )

        rows.append(row)

    table = pd.DataFrame(rows).set_index("model")

    if output_path is not None:
        table.to_csv(output_path)

    return table

def split_results_table(formatted_results, nsplit=4):
    """Split metric columns into balanced groups and stack them vertically."""
    df = formatted_results.copy()

    for model_column in ("model", "Model"):
        if model_column in df.columns:
            df = df.set_index(model_column)
            break

    n_columns = len(df.columns)
    if not 1 <= nsplit <= n_columns:
        raise ValueError(
            f"nsplit must be between 1 and {n_columns}"
        )

    # Divide columns into balanced groups
    base_size, remainder = divmod(n_columns, nsplit)
    group_sizes = [
        base_size + (i < remainder)
        for i in range(nsplit)
    ]

    blocks = []
    start = 0

    for size in group_sizes:
        blocks.append(df.iloc[:, start:start + size].copy())
        start += size

    max_width = max(group_sizes)
    output_columns = ["Model"] + blocks[0].columns.tolist()

    sections = []

    for i, block in enumerate(blocks):
        if i > 0:
            header = (
                ["Model"]
                + block.columns.tolist()
                + [""] * (max_width - block.shape[1])
            )
            sections.append(
                pd.DataFrame([header], columns=output_columns)
            )

        rows = block.rename_axis("Model").reset_index()
        rows.columns = output_columns[:rows.shape[1]]
        rows = rows.reindex(columns=output_columns, fill_value="")
        sections.append(rows)

    return pd.concat(sections, ignore_index=True)

def summarize_method(meth, runs, metrics_names, threshold, n_boot, seed):
    summary, per_run, calibration = summarize_repeated_cv(
        runs,
        metrics_names=metrics_names,
        threshold=threshold,
        n_boot=n_boot,
        n_bins=seed,
    )
    #summary, per_run = summarize_repeated_cv(
    #    runs,
    #    metrics_names,
    #    threshold=threshold,
    #    n_boot=n_boot,
    #    seed=seed,
    #)
    return meth, summary, per_run, calibration

def main():
    args = parse_args()

    summaries = {}
    scores_by_run = {}
    results = load_metric_files(args)

    n_jobs = min(20, len(args.methods))

    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(n_jobs=n_jobs, pre_dispatch=n_jobs) as parallel:
            outputs = parallel(
                delayed(summarize_method)(
                    meth="+".join(m),
                    runs=[
                        results["+".join(m)][i]
                        for i in range(args.iter)
                    ],
                    metrics_names=args.metrics,
                    threshold=0.30,
                    n_boot=1000,
                    seed=42,
                )
                for m in args.methods
            )
        
    summaries = {
        meth: summary
        for meth, summary, per_run, calibration in outputs
    }
    calibrations = {
        meth: calibration
        for meth, summary, per_run, calibration in outputs
    }

    # save summaries of metrics
    if args.save_summary:
        for m in args.methods:
            meth = "+".join(m)
            print(f"\n{meth}")
            print(summaries[meth].round(3))
            summaries[meth].to_csv(f"{args.results_dir}/summary_{args.dataset}_{meth}_ITER{args.iter}.csv")
            calibrations[meth].to_csv(f"{args.results_dir}/calibration_{args.dataset}_{meth}_ITER{args.iter}.csv")

    formatted_results = format_results_table(
            summaries=summaries,
            results=results,
            methods=args.methods,
            n_iter=args.iter,
            metric_names=args.metrics,
            threshold=0.30,
            output_path=(
                f"{args.results_dir}/"
                f"result_table_{args.dataset}_ITER{args.iter}.csv"
            ),
    )
    if args.to_latex:
        table = split_results_table(formatted_results, nsplit=4)
        print(
            highlight_best_values(
                table.to_latex(index=False, escape=True)
            )
        )

if __name__ == "__main__":
    main()