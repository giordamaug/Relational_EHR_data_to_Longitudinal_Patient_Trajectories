#!/usr/bin/env python
# coding: utf-8

### Libraries

import torch
import pandas as pd
import numpy as np
import re
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle
from scripts.cv_folding import lgbm_cv_iter, e2e_cv
from pathlib import Path
import time
from scripts.configure import configure
from scripts.utils import group_events_by_visit

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # to force BERT determinsm

def print_ascii_table(df, show_index=False):
    headers = [str(col) for col in df.columns]
    rows = df.astype(str).values.tolist()

    if show_index:
        headers.insert(0, str(df.index.name or "index"))
        rows = [[str(idx), *row] for idx, row in zip(df.index, rows)]

    # Evita che i ritorni a capo nelle celle rompano la tabella.
    rows = [[cell.replace("\n", "\\n") for cell in row] for row in rows]
    widths = [
        max(len(header), *(len(row[i]) for row in rows))
        for i, header in enumerate(headers)
    ]

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(values):
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(values, widths)
        ) + " |"

    print(border)
    print(format_row(headers))
    print(border)
    for row in rows:
        print(format_row(row))
    print(border)
    
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # per multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def check_methods(methods, defmethods):
    if methods is None:
        return ['STATIC', 'STATIC+BINARY', 'STATIC+FREQUENCY', 'STATIC+EVENT-CNT',  # baseline
                'STATIC+TF-IDF', 'STATIC+BOW', 'STATIC+BiPadLSTM', 'STATIC+Med2Vec', 
                'STATIC+Dipole', 'STATIC+CEHR-BERT', 'STATIC+DOME', 'STATIC+GRU']
    for meths in methods:
        for m in meths.split("+"):
            if not m in defmethods:
                raise ValueError(f"\033[31mWrong method {meths}!\033[0m")
    return methods

def shuffle_events(event_sequences, event_sequences_type, seed):
    if event_sequences.keys() != event_sequences_type.keys():
        raise ValueError(f"\033[31mThe two dictionary must have same ids\033[0m")

    rng = random.Random(seed)

    for id, events in event_sequences.items():
        types = event_sequences_type[id]

        if len(events) != len(types):
            raise ValueError(f"\033[31mLists must be same length for id! {id!r}\033[0m")

        indices = list(range(len(events)))
        rng.shuffle(indices)

        event_sequences[id] = [events[i] for i in indices]
        event_sequences_type[id] = [types[i] for i in indices]

    visit_sequences = group_events_by_visit(event_sequences)
    return event_sequences, event_sequences_type, visit_sequences 


### Load dataset

import sys, os

DATA_PATH = "data" # "your-path-to-data"
DEFAULT_METHODS = ['STATIC', 'BINARY', 'FREQUENCY', 'EVENT-CNT', 'TF-IDF', 
              'BOW', 'BiPadLSTM', 'Med2Vec', 'Dipole', 'CEHR-BERT',
               'DOME', 'GRU']
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Esegue l'esperimento sul dataset INA")
    parser.add_argument("--data-path", default="data", help="Cartella contenente il dataset")
    parser.add_argument("--datafile")
    parser.add_argument("--dataname", default="INA")
    parser.add_argument("--pathology-field", default="primary_disease")
    parser.add_argument(
        "--disease-list",
        nargs="+",
        default=[
            "CHA - Congenital Hemolytic Anemias",
            "THA - Thalassemia",
            "SCD - Sickle Cell Disease",
            "CA - Congenital Asplenia",
            "IM - Immunodeficiencies",
            "OD - Oncological Diseases",
            "OSV - Other Structural vascular",
            "OTI - Other Trauma Iatrogenic",
            "OTH - Other pathology",
        ],
    )
    parser.add_argument(
        "--static-vars",
        nargs="+",
        default=["gender", "age_group", "age_at_splenectomy", "primary_disease"],
    )
    parser.add_argument(
        "--pattern",
        default=r"Encounter for palliative|Do not resuscitate|Patient room in hospital|Unspecified place",
    )
    parser.add_argument("--target-var", default="is_alive?")
    parser.add_argument("--evfields", nargs="+", default=["code_descr"])
    parser.add_argument("--remove-events", nargs="+", default=["drug"])
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "AUC", "AUPRC", "F1", "PPV", "Sensitivity", "Specificity",
            "NPV", "MCC", "Accuracy", "Brier", "CM",
        ],
    )

    parser.add_argument("--enable-plot", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--enable-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--perf-save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--iter", type=int, default=10)
    parser.add_argument("--thresh", type=float, default=0.3)
    parser.add_argument("--results-dir", default="experiment_main")
    parser.add_argument("--folds-dir", default=None)
    parser.add_argument(
        "--onlysplenectomized",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--removelastadm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--mode", choices=["skip", "e2e", "embed"], default="skip")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method list with combination of methods with +",
    )
    return parser.parse_args()



def main(args):
    start = time.monotonic()
    cnt = 0
    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Loading the dataset {Path(args.data_path) / args.datafile}...\033[0m")
    args.dataset = pd.read_json(Path(args.data_path) / args.datafile)
    args.dataset.set_index("id", inplace=True)
    if args.verbose: print(args.dataset.head(), f"\033[33mPatients no: {len(args.dataset)}\033[0m")

    # ### Keep patients with the specific asplenic groups
    if args.pathology_field in args.dataset.columns:
        if args.verbose: print(print_ascii_table(args.dataset[args.pathology_field].value_counts(dropna=False).reset_index(name="count")))
        args.dataset = args.dataset[args.dataset[args.pathology_field].isin(args.disease_list)]
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Keep patients with the specific asplenic groups...\033[0m \033[33mPatients no: {len(args.dataset)}\033[0m")
    else:
        raise IOError(f"\033[31mWrong {args.pathology_field}!\033[0m")

    # ### Select splenectomize patients
    if args.onlysplenectomized:
        args.dataset = args.dataset[args.dataset["is_splenectomized?"] == 'YES']
        args.selected_patient_ids = args.dataset.index
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Select splenectomize patients...\033[0m \033[33mPatients no: {len(args.selected_patient_ids)}\033[0m")

    # ### Remove events in the mortal admission
    # For each diseased patient we remove all events related to the last hospital admission in which the death occurred.
    # Each event with admission ID equal to the one in which the patient dead are removed.

    if args.removelastadm and args.dataname == "MIMIC":
        for ind,row in args.dataset.iterrows():
            args.dataset.at[ind, "events"] = list(filter(lambda event: event['admission_id'] != row["admission_id_of_death"], row['events']))
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Remove events in the mortal admission...\033[0m \033[33mPatients no: {len(args.selected_patient_ids)}\033[0m")

    # ### Remove patients with specific cause of death
    # Remove othe rpatients with causes not related to the illness

    if "cause_of_death" in args.dataset.columns:
        words = ["accident", "trauma", "suicide", "fracture", "malignant", "myeloma"]
        args.dataset = args.dataset[~args.dataset["cause_of_death"].str.contains("|".join(words), case=False, na=False)]
        args.selected_patient_ids = args.dataset.index
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Remove patients with specific cause of death...\033[0m \033[33mPatients no: {len(args.selected_patient_ids)}\033[0m")

    # ## Get clinical trajectories
    from scripts.utils import count_events_by_type
    events_field, date_field, type_field = 'events', 'date', 'type'
    event_sequences = { id: [(" ".join([event[field] if isinstance(event[field], str) else "NONE" for field in args.evfields]), 
                            event[date_field], event[type_field]) for event in events] for (id,events) in args.dataset[[events_field]].to_dict()[events_field].items()} 
    if args.verbose: 
        print(f"\033[32m{(cnt := cnt + 1)} Extracting clinical trajectories by\033[0m {args.evfields}...\033[0m \033[33mPatients no {len(event_sequences)}\033[0m")

    # ### Remove events by types
    # get sequence of event names
    event_sequences = { id: [(" ".join([event[field] if isinstance(event[field], str) else "NONE" for field in args.evfields]), event[date_field]) for event in events if event[type_field] not in args.remove_events] for (id,events) in args.dataset[[events_field]].to_dict()[events_field].items()} 
    event_sequences_type = { id: [(" ".join([event[field] if isinstance(event[field], str) else "NONE" for field in args.evfields]), event[date_field], event[type_field]) for event in events if event[type_field] not in args.remove_events] for (id,events) in args.dataset[[events_field]].to_dict()[events_field].items()} 
    visit_sequences = group_events_by_visit(event_sequences)
    args.dataset['events'] = args.dataset.apply(lambda row: event_sequences[row.name], axis=1)
    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Remove events by types...\033[0m {args.remove_events} \033[33mPatients no:{len(event_sequences)}\033[0m")

    # ### Remove events by name
    if args.pattern != r"":
        event_sequences = { id: [event for event in events if not re.search(args.pattern, event[0])] for (id,events) in event_sequences.items()}
        event_sequences_type = { id: [event for event in events if not re.search(args.pattern, event[0])] for (id,events) in event_sequences_type.items()}
        visit_sequences = group_events_by_visit(event_sequences)
        if args.verbose:print(f"\033[32m{(cnt := cnt + 1)} Remove events by patterns...\033[0m\033[33mPatients no:{len(event_sequences)}\033[0m\n{args.pattern}")
    else:
        raise ValueError(f"\033[0mEmpty pattern for removal!\033[0m")
    
    # ### Remove patients with no events at all
    #args.dataset = args.dataset[args.dataset["events"].apply(lambda x: x != [])]
    #args.selected_patient_ids = args.dataset.index
    event_sequences = { id: events for (id,events) in event_sequences.items() if len(events) > 0 } 
    event_sequences_type = { id: events for (id,events) in event_sequences_type.items() if len(events) > 0 }
    args.selected_patient_ids = np.array(list(event_sequences.keys()))
    args.dataset = args.dataset.loc[args.selected_patient_ids]
    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Remove patients with no events...\033[0m \033[33mPatients no: {len(args.selected_patient_ids)}\033[0m")

    # ### Plot cohort distributions
    if args.enable_plot:
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Plotting cohort distributions...\033[0m")
        labels = [x.split(" - ")[0] for x in list(args.dataset[args.pathology_field].unique())]
        from scripts.plotting import plot_distribution, distrib_summary
        disease_mapper = dict(zip(list(args.dataset[args.pathology_field].unique()), labels))
        fig = plot_distribution(args.dataset, disease_mapper)
        distrib_summary(args.dataset, disease_mapper)
                        
    # ### count event by type and cardinality
    count_events_by_type(event_sequences_type)

    # ## Get static variables
    if len(args.static_vars) > 0:
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Get static variables {args.static_vars}...\033[0m")
        mapper = {'pediatric': 0, 'adolescent' : 1, 'adult': 2, 'ND' : -1, 'young': 0, 'mature' : 1, 'elder': 2, 'geriatric': 3}
        X_static = args.dataset.loc[args.selected_patient_ids][args.static_vars].copy()
        X_static.replace(mapper, inplace=True)
        cnames = list(X_static.select_dtypes(include=['object']).columns)
        for col in cnames:
            X_static[col], _ = pd.factorize(X_static[col])
        if args.verbose: print(X_static.head())
    else:
        X_static = pd.DataFrame(index=args.selected_patient_ids)

    # ## Get targets
    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Get targets...\033[0m")
    target_df = args.dataset[args.target_var].copy()
    target_df.replace({"n/a": "YES", np.nan: "YES"}, inplace=True)
    counts = target_df.value_counts(dropna=False)
    if args.enable_plot :
        if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Plotting targets pie...\033[0m")
        counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
        plt.title('Distribution of Targets')
        plt.ylabel('')
        plt.show()
    target_df.replace({"YES": 0, "NO": 1}, inplace=True)
    if args.enable_save:
        target_df.to_csv(os.path.join(DATA_PATH, f"{args.dataname}_targets.csv"))
    if args.verbose: print(print_ascii_table(target_df.value_counts(dropna=False).reset_index(name="count")))
    method_list = list(map(lambda x: x.split("+"), args.methods))
    print(args.methods)

    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Validation with mode {args.mode}...\033[0m")
    if args.mode in ["embed", "e2e"]:
        prob_pred = {m: [] for m in args.methods}
        prob_true = {m: [] for m in args.methods}
        all_runs = {m: ([], []) for m in args.methods}
        cpu_profiler = {m: {"emb_train_time_s": [],
            "emb_inference_time_s": [],
            "emb_inference_ms_per_patient": [],
            "emb_cpu_memory_delta_mb": [],
            "emb_peak_gpu_memory_mb": [],
            "emb_n_trainable_params": [],
            "emb_n_total_params": [],
            "clf_train_time_s": [],
            "clf_inference_time_s": [],
            "clf_inference_ms_per_patient": [],
            "clf_cpu_memory_delta_mb": [],} for xs in method_list for m in xs}
        # ## Validation (Iterated 5-fold)
        if args.mode == "embed":
            for i in range(args.iter):
                set_seed(args.random_state+i)
                if args.verbose: 
                    print(f"\033[95m####################\033[0m")
                    print(f"\033[95m# ITERATION:{i}      #\033[0m")
                    print(f"\033[95m####################\033[0m")
                if args.shuffle: 
                    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Shuffling sequences...\033[0m")
                    event_sequences, event_sequences_type, visit_sequences,= shuffle_events(event_sequences, event_sequences_type, args.random_state+i)
                embedder_configs = configure(event_sequences, visit_sequences, event_sequences_type, target_df.to_dict(), X_static, args)
                for m in args.methods:
                    _,_,_,_,_, profile, results = lgbm_cv_iter(embedder_configs, target_df, args.selected_patient_ids, i,
                                                                        methods=m.split("+"), threshold=args.thresh, random_state=args.random_state+i, 
                                                                        return_metrics=True, fold_indices_dir=args.folds_dir)
                    prob_pred[m] += list(results['calibration_prob_pred'])
                    prob_true[m] += list(results['calibration_prob_true'])
                    all_runs[m] = (all_runs[m][0] + list(results['y_true_oof']),
                                                all_runs[m][1] +  list(results['y_prob_oof']))
                    if args.enable_save: 
                        with open(f"{args.results_dir}/results_{args.dataname}_{m.replace('+','_')}_ITER{i}.pkl", 'wb') as f:
                            pickle.dump(results, f)
                    for method,record in profile.items():
                        for measure,lm in record.items():
                            cpu_profiler[method][measure] += lm
            if args.perf_save:
                dfmem = pd.DataFrame.from_dict(cpu_profiler, orient='index')
                dfmem = dfmem.map(lambda valori: f"{np.mean(valori):.3f}±{np.std(valori):.3f}" if len(valori) else np.nan)
                dfmem.to_csv(f"{args.results_dir}/perf_{args.dataname}_ITER{args.iter}.csv")

        elif args.mode == "e2e":
            args.methods = list(np.array(args.methods).ravel())
            for i in range(args.iter):
                set_seed(args.random_state+i)
                if args.verbose: 
                    print(f"\033[95m####################\033[0m")
                    print(f"\033[95m# ITERATION:{i}      #\033[0m")
                    print(f"\033[95m####################\033[0m")
                if args.shuffle: 
                    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Shuffling sequences...\033[0m")
                    event_sequences, event_sequences_type, visit_sequences,= shuffle_events(event_sequences, event_sequences_type, args.random_state+i)
                embedder_configs = configure(event_sequences, visit_sequences, event_sequences_type, target_df.to_dict(), X_static, args)
                for m in args.methods:
                    _,_, results = e2e_cv(embedder_configs, target_df, args.selected_patient_ids, method=m)
                    prob_pred[m] += list(results['calibration_prob_pred'])
                    prob_true[m] += list(results['calibration_prob_true'])
                    all_runs[m] = (all_runs[m][0] + list(results['y_true_oof']),
                                                all_runs[m][1] +  list(results['y_prob_oof']))
                    if args.enable_save: 
                        with open(f"{args.results_dir}/results_{args.dataname}_{m}_ITER{i}.pkl", 'wb') as f:
                            pickle.dump(results, f)
    elif args.mode == "skip":
        embedder_configs = configure(event_sequences, visit_sequences, event_sequences_type, target_df.to_dict(), X_static, args)
        train_idx = args.selected_patient_ids
        valid_idx = args.selected_patient_ids
        for method in list(map(lambda x: x.split("+"), args.methods)):
            for meth in method:
                cfg = embedder_configs[meth]
                func = cfg["func"]
                kwargs = (cfg["kwargs"].copy())
                kwargs["train_idx"] = train_idx
                kwargs["valid_idx"] = valid_idx
                func(**kwargs)
    else:
        raise ValueError(f"Wrong mode {args.mode}!")
    elapsed = int(time.monotonic() - start)        
    if args.verbose: print(f"\033[38;5;208mExperiments DONE!\033[0m in {elapsed // 60:02d}m{elapsed % 60:02d}s")

if __name__ == "__main__":
    args = parse_args()
    args.methods = check_methods(args.methods, DEFAULT_METHODS)
    main(args)
