#!/usr/bin/env python
# coding: utf-8

### Libraries

import io
import torch
import pandas as pd
import numpy as np
import re
import warnings
warnings.simplefilter(action='ignore')
import os
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pickle
from scripts.cv_folding import lgbm_cv_iter, e2e_cv
import time
from scripts.configure import configure
from scripts.utils import group_events_by_visit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
import lightgbm as lgb
from tqdm.auto import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"       # to force BERT determinsm

class TerminalOutput:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        return False
    def clear_output(self, wait=False):
        pass  # Nel terminale conserva i messaggi già stampati

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

DEFAULT_METHODS = ['STATIC', 'BINARY', 'FREQUENCY', 'EVENT-CNT', 'TF-IDF', 
                   'BOW', 'BiPadLSTM', 'Med2Vec', 'Dipole', 'CEHR-BERT',
                   'DOME', 'GRU']
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Esegue l'esperimento sul dataset INA")
    parser.add_argument("--datafile", help="Input dataset path (JSON)")
    parser.add_argument("--dataname", default="INA", help="Name of dataset")
    parser.add_argument("--pathology-field", default="primary_disease", help="Name of primary disease attibute")
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
        help="List of primary disease for patient selection"
    )
    parser.add_argument(
        "--static-vars",
        nargs="+",
        default=["gender", "age_group", "age_at_splenectomy", "primary_disease"],
        help="List of static variables (used by STATIC method)"
    )
    parser.add_argument(
        "--pattern",
        default=r"Encounter for palliative|Do not resuscitate|Patient room in hospital|Unspecified place",
        help="Patterns to match for event removals"
    )
    parser.add_argument("--target-var", default="is_alive?", help="Set target variable")
    parser.add_argument("--evfields", nargs="+", default=["code_descr"], help="Event description field used in processing of clinical sequences (\"event\", \"code\", \"code_descr\")")
    parser.add_argument("--remove-events", nargs="+", default=["drug"], help="Type of event to remove from clinical sequences")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=[
            "AUC", "AUPRC", "F1", "PPV", "Sensitivity", "Specificity",
            "NPV", "MCC", "Accuracy", "Brier", "CM",
        ], help="Metric list"
    )

    parser.add_argument("--enable-plot", action=argparse.BooleanOptionalAction, default=False, help="Enable training graph plotting")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds in cross-validation")
    parser.add_argument("--num-epochs", type=int, default=10, help="Embeddign param (number of epochs)")
    parser.add_argument("--batch-size", type=int, default=16, help="Embeddign param (batch size in training/testing)")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embeddign param (final embedding size)")
    parser.add_argument("--hidden-dim", type=int, default=300, help="Embeddign param (last hidden layer size)")
    parser.add_argument("--random-state", type=int, default=42, help="Set initial random seed (for reproducibility)")
    parser.add_argument("--enable-save", action=argparse.BooleanOptionalAction, default=True, help="Enable cv partition saving (if a valid dir)")
    parser.add_argument("--perf-save", action=argparse.BooleanOptionalAction, default=True, help="Enable saving to file of mehods memory/time consumptions")
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=False, help="Enable shuffling of clinical events")
    parser.add_argument("--iter", type=int, default=10, help="Number of repetition of cross-validation")
    parser.add_argument("--thresh", type=float, default=0.3, help="Set classification threshold")
    parser.add_argument("--results-dir", default="experiment_main", help="Folder for metric measure saving")
    parser.add_argument("--folds-dir", default=None, help="")
    parser.add_argument(
        "--onlysplenectomized",
        action=argparse.BooleanOptionalAction,
        default=True, 
        help="Select only splenectomized patients"
    )
    parser.add_argument(
        "--removelastadm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable removal of last admission events (only in MIMIC)"
    )
    parser.add_argument("--mode", choices=["skip", "e2e", "embed", "shap"], default="skip", 
                        help="Execution mode: \n- embed: embedding+LGBM classification\n- e2e: end-to-end classification (only for Med2Vec, Dipole, BiPadLSTM, GRU, CEHR-BERT)")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True, help="Enable verbosity")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Method list, combine methods with + (example, STATIC STATIC+DOME ...))",
    )
    return parser.parse_args()



def main(args):
    start = time.monotonic()
    cnt = 0
    if args.verbose: print(f"\033[32m{(cnt := cnt + 1)} Loading the dataset {args.datafile}...\033[0m")
    args.dataset = pd.read_json(args.datafile)
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
    if args.verbose: print(print_ascii_table(target_df.value_counts(dropna=False).reset_index(name="count")))
    method_list = list(map(lambda x: x.split("+"), args.methods))

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
    elif args.mode == "shap":
        import shap
        frame_top = TerminalOutput()
        frame_tqdm = TerminalOutput()
        frame_plot = TerminalOutput()
        frame_log = TerminalOutput()
        embedder_configs = configure(event_sequences, visit_sequences, event_sequences_type, target_df.to_dict(), X_static, args)
        lgb_params = {
            "objective": "binary",
            "metric": "None",
            "verbosity": -3,
            "is_unbalance": True,
            "seed": args.random_state,
            "deterministic": True,
            "force_col_wise": True,
        }
        def mcc_eval(y_pred, dataset):
            y_true = dataset.get_label()
            y_pred_labels = (y_pred > args.thresh).astype(int)
            mcc = matthews_corrcoef(y_true,y_pred_labels)
            return "MCC", mcc, True

        for m in args.methods:
            set_seed(args.random_state)
            skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
            y_df = target_df.loc[args.selected_patient_ids]
            y = (y_df.values.astype(np.float32).ravel())
            cvfolding = tqdm(skf.split(args.selected_patient_ids,y),total=args.n_splits,desc="Folds", position=0, leave=True)
            for fold, (tr_idx, val_idx) in enumerate(cvfolding):

                train_idx = args.selected_patient_ids[tr_idx]
                valid_idx = args.selected_patient_ids[val_idx]
                cfg = embedder_configs[m]
                func = cfg["func"]
                kwargs = (cfg["kwargs"].copy())
                kwargs["train_idx"] = train_idx
                kwargs["valid_idx"] = valid_idx
                X_emb_tr, X_emb_val, profile = func(**kwargs, frame_tqdm=frame_tqdm,frame_plot=frame_plot)                
                y_train = target_df.loc[train_idx]
                y_valid = target_df.loc[valid_idx]
                X_train = pd.concat([X_emb_tr, X_static.loc[train_idx]], axis=1)
                X_val = pd.concat([X_emb_val, X_static.loc[valid_idx]], axis=1)
                X = pd.concat([X_train, X_val], axis=0)
                shap_oof = np.zeros(X.shape)
                base_oof = np.zeros(len(X))
                train_data = lgb.Dataset(X_train,label=y_train)
                valid_data = lgb.Dataset(X_val,label=y_valid)
                model = lgb.train(lgb_params,train_data,num_boost_round=1000,
                    valid_sets=[valid_data],
                    feval=mcc_eval,
                    callbacks=[lgb.early_stopping(50, verbose=False),lgb.log_evaluation(0)])
                # Il background viene preso solo dal training del fold
                background = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.TreeExplainer(
                    model,
                    data=background,
                    feature_perturbation="interventional"
                )
                explanation = explainer(X_val)
                shap_oof[val_idx] = explanation.values
                base_oof[val_idx] = explanation.base_values
            # Spiegazioni out-of-fold per l'intero dataset
            explanation_oof = shap.Explanation(
                values=shap_oof,
                base_values=base_oof,
                data=X.to_numpy(),
                feature_names=X.columns.tolist()
            )
            # 1. Violin plot
            plt.figure()

            shap.summary_plot(
                shap_values=shap_oof,
                features=X,
                feature_names=X.columns.tolist(),
                plot_type="violin",
                max_display=15,
                plot_size=(12, 8),
                show=False
            )

            fig_violin = plt.gcf()
            buffer_violin = io.BytesIO()
            fig_violin.savefig(
                buffer_violin, format="png", dpi=300,
                bbox_inches="tight", facecolor="white"
            )
            buffer_violin.seek(0)
            image_violin = plt.imread(buffer_violin)
            plt.close(fig_violin)

            # 2. Bar plot
            plt.figure()

            shap.plots.bar(
                explanation_oof,
                max_display=15,
                show=False
            )

            fig_bar = plt.gcf()
            fig_bar.set_size_inches(12, 8)

            buffer_bar = io.BytesIO()
            fig_bar.savefig(
                buffer_bar, format="png", dpi=300,
                bbox_inches="tight", facecolor="white"
            )
            buffer_bar.seek(0)
            image_bar = plt.imread(buffer_bar)
            plt.close(fig_bar)

            # 3. Figura finale con due pannelli uno sotto l'altro
            fig, axes = plt.subplots(2, 1, figsize=(14, 20))

            axes[0].imshow(image_violin)
            axes[0].set_title("A. SHAP value distribution")

            axes[1].imshow(image_bar)
            axes[1].set_title("B. Global feature importance")

            for ax in axes:
                ax.axis("off")

            fig.tight_layout()

            fig.savefig(
                f"{args.results_dir}/{args.dataname}_{m}_shap.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white"
            )

            plt.close(fig)
            buffer_violin.close()
            buffer_bar.close()
            if args.verbose:print(f"\033[32m{(cnt := cnt + 1)} Saving SHAP plot\033[0m {args.results_dir}/{args.dataname}_{m}_shap.png")
    else:
        raise ValueError(f"Wrong mode {args.mode}!")
    elapsed = int(time.monotonic() - start)        
    if args.verbose: print(f"\033[38;5;208mExperiments DONE!\033[0m in {elapsed // 60:02d}m{elapsed % 60:02d}s")

if __name__ == "__main__":
    args = parse_args()
    args.methods = check_methods(args.methods, DEFAULT_METHODS)
    main(args)
