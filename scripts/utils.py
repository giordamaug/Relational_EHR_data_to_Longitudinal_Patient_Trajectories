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

def plot_group_distribution_with_event_boxplot(df, groupby="primary_disease_area", label_desc=None):
    df = df.copy()

    df['event_length'] = df['events'].apply(len)
    df = df.dropna(subset=[groupby, 'event_length'])
    if label_desc is not None:
        df[groupby] = df[groupby].map(label_desc)
        counts = df[groupby].value_counts()
        counts = counts.rename(label_desc).fillna(0)
    else:
        counts = df[groupby].value_counts()

    values = counts.values
    labels = counts.index
    print(labels)

    if values.sum() == 0:
        raise ValueError(
            f"Nessun dato valido per {groupby}. "
            f"Controlla il mapping e i valori originali della colonna."
        )

    palette = sns.color_palette("tab10", len(labels))
    palette_dict = dict(zip(labels, palette))

    def autopct_abs(pct):
        total = np.sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}' if val > 0 else ''

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    wedges, texts, autotexts = axes[0].pie(
        values,
        labels=None,
        colors=[palette_dict[l] for l in labels],
        autopct=autopct_abs,
        pctdistance=0.75,
        startangle=90
    )

    axes[0].set_title("(A) Cohort Distribution")
    axes[0].legend(
        wedges,
        labels,
        title=groupby,
        bbox_to_anchor=(1.05, 1),
        loc='upper left'
    )

    sns.boxplot(
        data=df,
        x=groupby,
        y='event_length',
        order=labels,
        hue=groupby,
        palette=palette_dict,
        dodge=False,
        legend=False,
        ax=axes[1]
    )

    axes[1].set_title("(B) Event Length Distribution")
    axes[1].tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

def pie_plot(dataset, pathology_field, pathologies):
    values = dataset[pathology_field].value_counts()
    legend_labels = values.index
    labels_short = [pathologies[l] for l in legend_labels]
    # 👉 funzione per mostrare valori assoluti
    def absolute_autopct(vals):
        def inner(pct):
            total = sum(vals)
            val = int(round(pct * total / 100.0))
            return f"{val}"
        return inner
    fig, ax = plt.subplots(figsize=(10, 3))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels_short,
        autopct=absolute_autopct(values)
    )
    # 👉 legenda separata
    ax.legend(
        wedges,
        legend_labels,
        title="Legend",
        loc="center left",
        bbox_to_anchor=(1.1, 0.5)
    )
    plt.tight_layout()
    plt.show()

def plot_event_boxplot(df, groupby="primary_disease", label_desc=None):
    df = df.copy()

    df['event_length'] = df['events'].apply(len)
    df = df.dropna(subset=[groupby, 'event_length'])
    if label_desc is not None:
        df[groupby] = df[groupby].map(label_desc)
        counts = df[groupby].value_counts()
        counts = counts.rename(label_desc).fillna(0)
    else:
        counts = df[groupby].value_counts()

    values = counts.values
    labels = counts.index
    print(labels)

    if values.sum() == 0:
        raise ValueError(
            f"Nessun dato valido per {groupby}. "
            f"Controlla il mapping e i valori originali della colonna."
        )

    palette = sns.color_palette("tab10", len(labels))
    palette_dict = dict(zip(labels, palette))

    def autopct_abs(pct):
        total = np.sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}' if val > 0 else ''

    fig, ax = plt.subplots(figsize=(10, 3))

    sns.boxplot(
        data=df,
        x=groupby,
        y='event_length',
        order=labels,
        hue=groupby,
        palette=palette_dict,
        dodge=False,
        legend=False,
        ax=ax
    )

    ax.set_title("(B) Event Length Distribution")
    ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------
# Sequence utility functions
#-------------------------------------------------------------------------------------

def truncevents(sequences, infection_list, max_inf=1, max_flwup=5, debug=False):
    trunc_sequences = {}
    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating to {max_inf}"):
        inf_cnt = 0
        flw_cnt = 0
        new_evset = set()
        for e, d in sequences[id]:
              if e in infection_list:
                if debug: print(f"INF[{id}] {e}")
                new_evset.add((e,d))
                inf_cnt += 1
                if inf_cnt >= max_inf: break
              elif e == "followup" :
                new_evset.add((e,d))
                flw_cnt += 1
                if flw_cnt >= max_flwup: break
              else:
                if debug: print(f"eve[{id}] {e}")
                new_evset.add((e,d))
        trunc_sequences[id] = sorted(list(new_evset), key=lambda x: x[1])
    return trunc_sequences

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

def truncate_events_on1st_target(sequences, target_list, max_occurrence=1, field_name='event', debug=False):
    trunc_sequences = {}
    zero_data = np.zeros((len(sequences.keys()),))
    y_df = pd.DataFrame(zero_data, columns=['target']).set_index(pd.Series(list(sequences.keys())))
    
    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating to {max_occurrence}"):
        inf_cnt = 0
        new_evset = []
        for event in sequences[id]:
                if event[0] in target_list:
                    if debug: print(f"INF[{id}] {event[0]}")
                    inf_cnt += 1
                    if inf_cnt >= max_occurrence: 
                        y_df.loc[id] = 1
                        break
                    new_evset.append(event)
                else:
                    new_evset.append(event)
        trunc_sequences[id] = new_evset
    return trunc_sequences, y_df

def truncate_events_on1st_infection(sequences, field_value='infection',  debug=False):
    trunc_sequences = {}
    zero_data = np.zeros((len(sequences.keys()),))
    y_df = pd.DataFrame(zero_data, columns=['target']).set_index(pd.Series(list(sequences.keys())))
    y_df.index.name = "id"

    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating"):
        inf_cnt = 0
        new_evset = []
        for event in sequences[id]:
                if event[2] == field_value:
                    if debug: print(f"INF[{id}] {event[2]}")
                    inf_cnt += 1
                    if inf_cnt >= 1:
                        y_df.loc[id] = 1
                        break
                    new_evset.append(event[0:2])
                else:
                    new_evset.append(event[0:2])
        trunc_sequences[id] = new_evset
    return trunc_sequences, y_df

def truncate_events_on1st_target_keepit(sequences, target_list, max_occurrence=1, field_name='event', debug=False):
    trunc_sequences = {}
    zero_data = np.zeros((len(sequences.keys()),))
    y_df = pd.DataFrame(zero_data, columns=['target']).set_index(pd.Series(list(sequences.keys())))
    
    # truncate event sequence to the k-th occurrence of target
    for id in tqdm(sequences.keys(), desc=f"Truncating to {max_occurrence}"):
        inf_cnt = 0
        new_evset = []
        for event in sequences[id]:
                if event[0] in target_list:
                    if debug: print(f"INF[{id}] {event[0]}")
                    new_evset.append(event)
                    inf_cnt += 1
                    if inf_cnt >= max_occurrence: 
                        y_df.loc[id] = 1
                        break
                else:
                    new_evset.append(event)
        trunc_sequences[id] = new_evset
    return trunc_sequences, y_df

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

def remove_target_from_sequences(sequences, targets):
    filtered_sequences = {}
    for id in tqdm(sequences.keys(), desc=f"Target removal"):
        # filter tuples with event not in targets
        filtered_seq = [
                (el, date_str)
                for el,date_str in sequences[id]
                if el not in targets
        ]
        filtered_sequences[id] = filtered_seq
    return filtered_sequences

def remove_target_from_visit_sequences(sequences, targets):
    filtered_sequences = {}
    for id in tqdm(sequences.keys(), desc=f"Target removal"):
        # filter tuples with event not in targets
        filtered_seq = [
                (list(set(els)-set(targets)), date_str)
                for els,date_str in sequences[id]
                if len(set(els)-set(targets)) > 0
        ]
        filtered_sequences[id] = filtered_seq
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
            row[f"n_{event_type}"] = n
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