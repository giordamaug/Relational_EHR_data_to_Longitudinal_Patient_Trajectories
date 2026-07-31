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