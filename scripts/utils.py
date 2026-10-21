import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
import os
import random
import numpy as np
import torch
from tqdm.notebook import tqdm
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from .models import BOWModel, TFIDFModel, DOME
import psutil
import re

#-------------------------------------------------------------------------------------
# Sequence utility functions
#-------------------------------------------------------------------------------------

def group_events_by_visit(sequences):
    visit_sequences= {}
    for pid, events in sequences.items():
        grouped_by_date = defaultdict(list)
        for event, date in events:
            grouped_by_date[date].append(event)
        visit_sequences[pid] = [(grouped_by_date[date], date) for date in sorted(grouped_by_date.keys())]
    return visit_sequences

def count_events_by_type(event_sequences):
    edf = pd.DataFrame(columns=['cardinality', 'n. instances', 'set'], 
                       index=pd.Series([], name='type'))
    for id, events in event_sequences.items():
        for event in events:
            if event[2] not in edf.index:
                row = pd.DataFrame([{'cardinality': 1, 'n. instances':1, 'set': set([event[0]])}],
                                    index=pd.Series([event[2]], name='type'))
                edf = pd.concat([edf, row], axis=0)
            else:
                edf.loc[event[2]]['set'].add(event[0])
                edf.loc[event[2], 'cardinality'] = len(edf.loc[event[2]]['set'])
                edf.loc[event[2], 'n. instances'] += 1
    return edf

def cooccurring_to_target(sequences, targets):
    filtered_sequences = {}
    for id in tqdm(sequences.keys(), desc=f"Cooccurrence removal"):
        # convert dates to datetype
        parsed_set = [(el, datetime.strptime(date_str, "%Y-%m-%d")) for el,date_str in sequences[id]]
        # Find most recent event,date pair
        if len(parsed_set) > 0:   # if the sequence is not null 
            # get dates form sequences
            _, dates = zip(*sequences[id])
            if len(set(dates)) > 1:                         # if at least two different dates
                max_date = max(date for _,date in parsed_set)
                # filter tuples with max date and with event in targets
                filtered_seq = [
                    (el, date.strftime("%Y-%m-%d"))
                    for el, date in parsed_set
                    if not (date == max_date and el not in targets + ['followup'])
                ]
                filtered_sequences[id] = filtered_seq
            else:
                seq = [
                    (el, date.strftime("%Y-%m-%d"))
                    for el, date in parsed_set
                ]
                filtered_sequences[id] = seq
        else:
            filtered_sequences[id] = parsed_set
    return filtered_sequences

def extract_event_type_counters(event_sequences):
    rows = []
    for patient_id, events in event_sequences.items():
        counts = Counter(
            event_type
            for _, _, event_type in events
        )
        row = {"patient_id": patient_id}
        for event_type, n in counts.items():
            row[f"{event_type.lower()}"] = n
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .fillna(0)
        .set_index("patient_id")
        .astype(int)
    )

def extract_event_counters(event_sequences):
    rows = []
    for patient_id, events in event_sequences.items():
        counts = Counter(
            event
            for event, _, _ in events
        )
        row = {"patient_id": patient_id}
        for event_type, n in counts.items():
            row[f"{event_type.lower()}"] = n
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .fillna(0)
        .set_index("patient_id")
        .astype(int)
    )

def split_array(a):
    if not a:
        return []

    result = []
    current = [a[0]]

    for i in range(len(a) - 1):
        if a[i] > a[i + 1]:
            result.append(current)
            current = [a[i + 1]]
        else:
            current.append(a[i + 1])

    result.append(current)
    return result

#-------------------------------------------------------------------------------------
# Memory consumption utility functions
#-------------------------------------------------------------------------------------

def count_trainable_parameters(model):
    """
    Counts the parameters learned by the model.
    PyTorch:  Parameters with `requires_grad=True`.
    TFIDFModel: IDF weights + SVD coefficients.
    BOWModel: SVD coefficients, if present.
    Does not include the downstream classifier.
    """

    if model is None:
        return np.nan

    # -----------------------------------------
    # Deep learning models
    # -----------------------------------------
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ==========================================
    # DOME
    # ==========================================
    if isinstance(model, DOME):

        if (model.W is None or model.C_plus is None or model.C_minus is None):
            return np.nan

        return int(model.W.size + model.C_plus.size + model.C_minus.size)
        
    # -----------------------------------------
    # TF-IDF
    # -----------------------------------------
    if isinstance(model, TFIDFModel):
        if not model.is_fitted:
            return np.nan

        n_params = 0

        # IDF weights
        if hasattr(model.vectorizer, "idf_"):
            n_params += model.vectorizer.idf_.size

        # SVD coefficients
        if model.use_svd and hasattr(model.svd, "components_"):
            n_params += model.svd.components_.size

        return int(n_params)

    # -----------------------------------------
    # Bag-of-Words
    # -----------------------------------------
    if isinstance(model, BOWModel):

        if not model.is_fitted:
            return np.nan

        n_params = 0

        # SVD coefficients
        if model.use_svd and hasattr(model.svd, "components_"):
            n_params += model.svd.components_.size

        return int(n_params)

    return np.nan


def count_total_parameters(model):
    """
    Counts all the numerical parameters in the model.
    PyTorch: All parameters, including frozen ones.
    TFIDFModel: IDF weights + SVD coefficients.
    BOWModel: SVD coefficients.
    Does not include the downstream classifier.
    """

    if model is None:
        return np.nan

    # -----------------------------------------
    # Deep learning models
    # -----------------------------------------
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters())

    # ==========================================
    # DOME
    # ==========================================
    if isinstance(model, DOME):

        if (model.W is None or model.C_plus is None or model.C_minus is None):
            return np.nan

        return int(model.W.size + model.C_plus.size + model.C_minus.size)

    # -----------------------------------------
    # TF-IDF
    # -----------------------------------------
    if isinstance(model, TFIDFModel):

        if not model.is_fitted:
            return np.nan

        n_params = 0

        if hasattr(model.vectorizer, "idf_"):
            n_params += model.vectorizer.idf_.size

        if model.use_svd and hasattr(model.svd, "components_"):
            n_params += model.svd.components_.size

        return int(n_params)

    # -----------------------------------------
    # Bag-of-Words
    # -----------------------------------------
    if isinstance(model, BOWModel):

        if not model.is_fitted:
            return np.nan

        n_params = 0

        if model.use_svd and hasattr(model.svd, "components_"):
            n_params += model.svd.components_.size

        return int(n_params)

    return np.nan

import os
import time
import psutil
import threading


class MemoryProfiler:

    def __init__(self, interval=0.01):
        """
        interval: intervallo di campionamento in secondi.
        """
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.baseline_mb = 0.0
        self.peak_mb = 0.0
        self._running = False
        self._thread = None

    def _monitor(self):
        while self._running:
            try:
                rss = (
                    self.process.memory_info().rss
                    / 1024**2
                )

                self.peak_mb = max(
                    self.peak_mb,
                    rss
                )
            except psutil.Error:
                break
            time.sleep(self.interval)

    def start(self):
        self.baseline_mb = (
            self.process.memory_info().rss
            / 1024**2
        )
        self.peak_mb = self.baseline_mb
        self._running = True
        self._thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join()
        # Ultima lettura
        rss = (
            self.process.memory_info().rss
            / 1024**2
        )
        self.peak_mb = max(
            self.peak_mb,
            rss
        )
        return {
            "baseline_mb": self.baseline_mb,
            "peak_rss_mb": self.peak_mb,
            "peak_increase_mb": max(
                0.0,
                self.peak_mb - self.baseline_mb
            )
        }
        
def get_process_memory_mb():
    """
    Resident Set Size del processo Python.
    Include memoria Python + librerie native + PyTorch CPU.
    """
    process = psutil.Process(os.getpid())

    return (
        process.memory_info().rss /
        1024**2
    )


def reset_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_peak_gpu_memory_mb():
    if torch.cuda.is_available():
        return (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    return 0.0

def mean_std_ci(values, confidence=0.95):

    values = np.asarray(
        values,
        dtype=float
    )

    # elimina eventuali nan
    values = values[
        ~np.isnan(values)
    ]

    if len(values) == 0:
        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan
        )

    mean = np.mean(values)

    if len(values) < 2:
        return (
            mean,
            np.nan,
            np.nan,
            np.nan
        )

    std = np.std(
        values,
        ddof=1
    )

    sem = st.sem(values)

    h = st.t.ppf(
        (1 + confidence) / 2,
        df=len(values) - 1
    ) * sem

    return (
        mean,
        std,
        mean - h,
        mean + h
    )


#-------------------------------------------------------------------------------------
# Print results formatting
#-------------------------------------------------------------------------------------

def std_(a):
    N = len(a)
    d2 = abs(a - a.mean())**2  # abs is for complex `a`
    var = d2.sum() / N  # note use of `ddof`
    return var**0.5

def get_mean_std(dfs, tmetrics, method, niter):
    values_mean = []
    values_std = []
    valuesp = []
    for metric in tmetrics:
        if metric == "CM":
            cm_mean = (dfs[metric][method].apply(lambda x: np.array(eval(x))).sum() / niter)
            cm_std = (dfs[metric][method].apply(lambda x: std_(np.array(eval(x)))))
            values_mean.append(cm_mean)
            values_std.append(cm_std)
            valuesp.append(cm_mean)

        elif metric == "Brier":
            mean, std, lo, hi = mean_std_ci(dfs[metric][method].values)
            values_mean.append(1 - mean)
            values_std.append(std)
            valuesp.append(f"{mean:.3f} [{lo:.3f}-{hi:.3f}]")
        else:
            mean, std, lo, hi = mean_std_ci(dfs[metric][method].values)
            values_mean.append(mean)
            values_std.append(std)
            valuesp.append(f"{mean:.3f}±{std:.3f} [{lo:.3f}-{hi:.3f}]")
    return values_mean, values_std, valuesp

NUMBER = re.compile(r"^(\s*)(\d+(?:\.\d+)?)")
def highlight_best_values(text: str) -> str:
    lines = text.splitlines(keepends=True)
    metrics = []
    block = []

    def finish_block():
        for column, metric in enumerate(metrics, start=1):
            if metric == "CM":
                continue

            candidates = []
            for index in block:
                cells = lines[index].split("&")
                if column >= len(cells):
                    continue
                match = NUMBER.match(cells[column])
                if match:
                    candidates.append((index, float(match.group(2))))

            if not candidates:
                continue

            best = (
                min(value for _, value in candidates)
                if metric == "Brier"
                else max(value for _, value in candidates)
            )

            for index, value in candidates:
                if value == best:
                    cells = lines[index].split("&")
                    cell = cells[column]
                    match = NUMBER.match(cell)
                    start, end = match.span(2)
                    cells[column] = (
                        cell[:start]
                        + r"\textbf{"
                        + match.group(2)
                        + "}"
                        + cell[end:]
                    )
                    lines[index] = "&".join(cells)

    for index, line in enumerate(lines):
        cells = line.split("&")

        if cells[0].strip() == "Model":
            finish_block()
            block = []
            metrics = [cell.strip().rstrip("\\").strip() for cell in cells[1:]]
        elif metrics and len(cells) > 1 and NUMBER.match(cells[1]):
            block.append(index)

    finish_block()
    return "".join(lines)

#-------------------------------------------------------------------------------------
# CPU profile function
#-------------------------------------------------------------------------------------

def fill_profile(emb_train_time, emb_inference_time, train_idx, valid_idx, emb_cpu_mem, 
                 emb_peak_gpu_mb, emb_count_trainable_parameters, emb_count_total_parameters,
                 clf_train_time_s=None, clf_inference_time_s=None, clf_inference_ms_per_patient=None, clf_cpu_memory_delta_mb=None):
    return     {
        "emb_train_time_s": emb_train_time,
        "emb_inference_time_s": emb_inference_time,
        "emb_inference_ms_per_patient": (emb_inference_time / (len(train_idx) + len(valid_idx)) * 1000),
        "emb_cpu_memory_delta_mb": emb_cpu_mem,
        "emb_peak_gpu_memory_mb": emb_peak_gpu_mb,
        "emb_n_trainable_params": emb_count_trainable_parameters,
        "emb_n_total_params": emb_count_total_parameters,
        "clf_train_time_s": clf_train_time_s,
        "clf_inference_time_s": clf_inference_time_s,
        "clf_inference_ms_per_patient": clf_inference_ms_per_patient,
        "clf_cpu_memory_delta_mb": clf_cpu_memory_delta_mb,
    }

#-------------------------------------------------------------------------------------
# Table utility (split)
#-------------------------------------------------------------------------------------

def create_table_with_splits(formatted_results, nsplit=4):
    """Split columns into balanced blocks and stack them vertically."""
    if not isinstance(nsplit, int) or nsplit < 1:
        raise ValueError("nsplit must be a positive integer.")
    if formatted_results.shape[1] == 0:
        raise ValueError("The DataFrame must have at least one column.")

    # Avoid empty blocks
    n = len(formatted_results.columns)
    nsplit = min(nsplit, n)

    base, remainder = divmod(n, nsplit)
    sizes = [base + (i < remainder) for i in range(nsplit)]

    blocks = []
    start = 0
    for size in sizes:
        blocks.append(
            formatted_results.iloc[:, start:start + size].copy()
        )
        start += size

    width = max(sizes)
    final_columns = ["Model"] + blocks[0].columns.tolist()

    parts = []
    for i, block in enumerate(blocks):
        if i > 0:
            header = (
                ["Model"]
                + block.columns.tolist()
                + [""] * (width - block.shape[1])
            )
            parts.append(
                pd.DataFrame([header], columns=final_columns)
            )

        rows = block.rename_axis("Model").reset_index()
        rows.columns = final_columns[:rows.shape[1]]
        rows = rows.reindex(columns=final_columns, fill_value="")
        parts.append(rows)

    return pd.concat(parts, ignore_index=True)

