import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, confusion_matrix, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, brier_score_loss, average_precision_score
)
import lightgbm as lgb
import ipywidgets as widgets
from contextlib import nullcontext
from IPython import get_ipython
from IPython.display import clear_output, display
from tqdm.auto import tqdm
from .utils import MemoryProfiler, get_process_memory_mb
import time

def in_notebook():
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"

class TerminalOutput:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def clear_output(self, wait=False):
        pass  # Nel terminale conserva i messaggi già stampati

#----------------------------------------------------------------------------------
# Iterated CV folding with LGBM classification
#----------------------------------------------------------------------------------

def lgbm_cv_iter_early_stopping_problem(
    embedder_configs,
    y_df,
    selected_patient_ids,
    iteration,
    n_splits=5,
    threshold=0.5,
    methods=[],
    return_metrics=False,
    random_state=42
):
    import numpy as np
    import pandas as pd
    import scipy.stats as st

    import lightgbm as lgb

    from tqdm.auto import tqdm
    from IPython.display import display, clear_output
    import ipywidgets as widgets

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        matthews_corrcoef,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        brier_score_loss
    )
    from sklearn.calibration import calibration_curve

    # =========================================================
    # UI
    # =========================================================

    notebook = in_notebook()

    if notebook:
        import ipywidgets as widgets

        clear_output(wait=True)

        frame_top = widgets.Output()
        frame_tqdm = widgets.Output()
        frame_plot = widgets.Output()
        frame_log = widgets.Output()

        box_cv = widgets.Tab(children=[frame_top])
        box_cv.set_title(0, "🌀 Cross Validation")

        a_plot = widgets.Tab(children=[frame_plot])
        a_tqdm = widgets.Tab(children=[frame_tqdm])

        a_plot.set_title(0, "📉 Training Plot")
        a_tqdm.set_title(0, "📉 Training Loop")

        a_out = widgets.Tab(children=[frame_log])
        a_out.set_title(0, "📝 Output")

        box_train = widgets.VBox([a_plot, a_tqdm])

        display(box_cv, box_train, a_out)
    else:
        frame_top = TerminalOutput()
        frame_tqdm = TerminalOutput()
        frame_plot = TerminalOutput()
        frame_log = TerminalOutput()
    # =========================================================
    # LIGHTGBM PARAMETERS
    # =========================================================

    lgb_params = {
        "objective": "binary",
        "metric": "None",
        "verbosity": -3,
        "is_unbalance": True,
    }

    # =========================================================
    # CUSTOM MCC FOR LIGHTGBM
    # =========================================================

    def mcc_eval(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred_labels = (y_pred > threshold).astype(int)
        mcc = matthews_corrcoef(y_true,y_pred_labels)
        return "MCC", mcc, True

    # =========================================================
    # 95% CONFIDENCE INTERVAL
    # =========================================================

    def mean_std_ci(values, confidence=0.95):

        values = np.asarray(values,dtype=float)
        # elimina eventuali nan
        values = values[~np.isnan(values)]

        if len(values) == 0:
            return (np.nan,np.nan,np.nan,np.nan)
        mean = np.mean(values)
        if len(values) < 2:
            return (mean,np.nan,np.nan,np.nan)
        std = np.std(values,ddof=1)
        sem = st.sem(values)
        h = st.t.ppf((1 + confidence) / 2, df=len(values) - 1) * sem
        return (mean,std,mean - h,mean + h)

    # =========================================================
    # METRIC LISTS
    # =========================================================

    mcc_scores = []
    acc_scores = []

    rocauc_scores = []
    auprc_scores = []

    recall_scores = []
    f1_scores = []

    specificity_scores = []
    ppv_scores = []
    npv_scores = []

    # OOF predictions
    y_idx_all = []
    y_valid_all = []
    y_pred_all = []
    y_prob_all = []
    computational_profiles = {
        method: {
            "emb_train_time_s": [],
            "emb_inference_time_s": [],
            "emb_inference_ms_per_patient": [],
            "emb_cpu_memory_delta_mb": [],
            "emb_peak_gpu_memory_mb": [],
            "emb_n_trainable_params": [],
            "emb_n_total_params": [],
            "clf_train_time_s": [],
            "clf_inference_time_s": [],
            "clf_inference_ms_per_patient": [],
            "clf_cpu_memory_delta_mb": [],
        }
        for method in methods
    }

    # =========================================================
    # CROSS VALIDATION
    # =========================================================

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    selected_patient_ids = np.asarray(selected_patient_ids)
    y_df = y_df.loc[selected_patient_ids]
    y = (y_df.values.astype(np.float32).ravel())

    with frame_top:

        cvfolding = tqdm(skf.split(selected_patient_ids,y),total=n_splits,desc="Folds", position=0, leave=True)
        for fold, (t_idx, v_idx) in enumerate(cvfolding):

            train_idx = selected_patient_ids[t_idx]
            valid_idx = selected_patient_ids[v_idx]
            train_features_list = []
            valid_features_list = []

            # =================================================
            # EMBEDDERS
            # =================================================

            for method in methods:

                cfg = embedder_configs[method]
                func = cfg["func"]
                kwargs = (cfg["kwargs"].copy())
                kwargs["train_idx"] = train_idx
                kwargs["valid_idx"] = valid_idx
                result = func(**kwargs,frame_tqdm=frame_tqdm,frame_plot=frame_plot)

                if len(result) == 3:
                    train_emb_df, valid_emb_df, profile = result
                else:
                    train_emb_df, valid_emb_df = result
                    profile = {
                        "emb_train_time_s": np.nan,
                        "emb_inference_time_s": np.nan,
                        "emb_inference_ms_per_patient": np.nan,
                        "emb_cpu_memory_delta_mb": np.nan,
                        "emb_peak_gpu_memory_mb": np.nan,
                        "emb_n_trainable_params": np.nan,
                        "emb_n_total_params": np.nan,
                        "clf_train_time_s": np.nan,
                        "clf_inference_time_s": np.nan,
                        "clf_inference_ms_per_patient": np.nan,
                        "clf_cpu_memory_delta_mb": np.nan,
                    }

                train_features_list.append(train_emb_df)
                valid_features_list.append(valid_emb_df)
                

            # =================================================
            # CONCAT FEATURES
            # =================================================

            train_df = pd.concat(train_features_list,axis=1)
            valid_df = pd.concat(valid_features_list,axis=1)
            y_train = (y_df.loc[train_idx].values.astype(np.float32).ravel())
            y_valid = (y_df.loc[valid_idx].values.astype(np.float32).ravel())

            # =================================================
            # LIGHTGBM DATASET
            # =================================================

            train_data = lgb.Dataset(train_df,label=y_train)
            valid_data = lgb.Dataset(valid_df,label=y_valid)

            # =================================================
            # TRAIN MODEL
            # =================================================

            with frame_log:
                cpu_mem_before = get_process_memory_mb()    # Perf measure
                t0 = time.perf_counter()                    # Perf measure
                model = lgb.train(lgb_params,train_data,num_boost_round=1000,
                    valid_sets=[valid_data],
                    feval=mcc_eval,
                    callbacks=[lgb.early_stopping(50, verbose=False),lgb.log_evaluation(0)])
                train_time = time.perf_counter() - t0                      # Perf measure

            # =================================================
            # PREDICTIONS
            # =================================================

            profiler = MemoryProfiler(interval=0.01)
            profiler.start()
            t0 = time.perf_counter()     # Perf measure
            y_prob = model.predict(valid_df)
            y_pred_labels = (y_prob > threshold).astype(int)
            inference_time = time.perf_counter() - t0                  # Perf measure
            #cpu_mem_after = get_process_memory_mb()                    # Perf measure
            memory_stats = profiler.stop()
            peak_memory_mb = memory_stats["peak_rss_mb"]
            profile["clf_train_time_s"] = train_time,
            profile["clf_inference_time_s"] = inference_time,
            profile["clf_inference_ms_per_patient"] = (inference_time / (len(train_idx) + len(valid_idx)) * 1000),
            profile["clf_cpu_memory_delta_mb"] = peak_memory_mb #cpu_mem_after - cpu_mem_before,

            for key in computational_profiles[method]:
                computational_profiles[method][key].append(profile.get(key,np.nan))
            # =================================================
            # STANDARD METRICS
            # =================================================

            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid,y_pred_labels)
            rocauc = roc_auc_score(y_valid,y_prob)
            auprc = average_precision_score(y_valid,y_prob)
            recall = recall_score(y_valid,y_pred_labels,zero_division=0)
            f1 = f1_score(y_valid,y_pred_labels,zero_division=0)

            # =================================================
            # CONFUSION-MATRIX METRICS
            # =================================================

            tn, fp, fn, tp = confusion_matrix(y_valid,y_pred_labels,labels=[0, 1]).ravel()
            specificity = ( tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            ppv = (tp / (tp + fp) if (tp + fp) > 0 else np.nan)
            npv = (tn / (tn + fn) if (tn + fn) > 0 else np.nan)

            # =================================================
            # STORE FOLD METRICS
            # =================================================

            mcc_scores.append(mcc)
            acc_scores.append(acc)
            rocauc_scores.append(rocauc)
            auprc_scores.append(auprc)
            recall_scores.append(recall)
            f1_scores.append(f1)
            specificity_scores.append(specificity)
            ppv_scores.append(ppv)
            npv_scores.append(npv)

            # =================================================
            # STORE OOF PREDICTIONS
            # =================================================

            y_idx_all.extend(valid_idx)
            y_valid_all.extend(y_valid)
            y_pred_all.extend(y_pred_labels)
            y_prob_all.extend(y_prob)

            # =================================================
            # PROGRESS BAR
            # =================================================

            cvfolding.set_postfix({
                "Fold": fold + 1,
                "AUC": f"{rocauc:.4f}",
                "AUPRC": f"{auprc:.4f}",
                "Sensitivity": f"{recall:.4f}",
                "Spec": f"{specificity:.4f}",
                "PPV": f"{ppv:.4f}",
                "NPV": f"{npv:.4f}",
                "F1": f"{f1:.4f}",
                "MCC": f"{mcc:.4f}"
            })

    # =========================================================
    # FINAL OOF ANALYSIS
    # =========================================================

    with frame_log:

        print(f"Train data {train_df.shape} - "f"Test data {valid_df.shape}")
        y_valid_all = np.asarray(y_valid_all)
        y_pred_all = np.asarray(y_pred_all)
        y_prob_all = np.asarray(y_prob_all)

        # =====================================================
        # AGGREGATED CONFUSION MATRIX
        # =====================================================

        cm_final = confusion_matrix(y_valid_all,y_pred_all,labels=[0, 1])
        tn, fp, fn, tp = (cm_final.ravel())
        specificity_oof = (tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        ppv_oof = (tp / (tp + fp) if (tp + fp) > 0 else np.nan)
        npv_oof = (tn / (tn + fn) if (tn + fn) > 0 else np.nan)
        recall_oof = (tp / (tp + fn) if (tp + fn) > 0 else np.nan)

        # =====================================================
        # OOF AUC / AUPRC
        # =====================================================

        try:
            auc_oof = roc_auc_score(y_valid_all,y_prob_all)
        except Exception:
            auc_oof = np.nan
        try:
            auprc_oof = average_precision_score(y_valid_all,y_prob_all)
        except Exception:
            auprc_oof = np.nan

        # =====================================================
        # CALIBRATION
        # =====================================================

        prob_true, prob_pred = calibration_curve(y_valid_all,y_prob_all,n_bins=10,strategy="uniform")
        brier = brier_score_loss(y_valid_all,y_prob_all)

        # =====================================================
        # MEAN / STD / CI95
        # =====================================================

        (aucmean,aucstd,auc_lo,auc_hi) = mean_std_ci(rocauc_scores)
        (auprcmean,auprcstd,auprc_lo,auprc_hi) = mean_std_ci(auprc_scores)
        (f1mean,f1std,f1_lo,f1_hi) = mean_std_ci(f1_scores)
        (recmean,recstd,rec_lo,rec_hi) = mean_std_ci(recall_scores)
        (specmean,specstd,spec_lo,spec_hi) = mean_std_ci(specificity_scores)
        (ppvmean,ppvstd,ppv_lo,ppv_hi) = mean_std_ci(ppv_scores)
        (npvmean,npvstd,npv_lo,npv_hi) = mean_std_ci(npv_scores)
        (mccmean,mccstd,mcc_lo,mcc_hi) = mean_std_ci(mcc_scores)
        (accmean,accstd,acc_lo,acc_hi) = mean_std_ci(acc_scores)

        # =====================================================
        # OUTPUT
        # =====================================================

        print(f"\n📊 Experiment {iteration}. Risultati medi su {n_splits} fold:")
        print(f"📈 AUC:         "f"{aucmean:.4f} ± {aucstd:.4f} "f"[95% CI {auc_lo:.4f}-{auc_hi:.4f}]")
        print(f"📉 AUPRC:       "f"{auprcmean:.4f} ± {auprcstd:.4f} "f"[95% CI {auprc_lo:.4f}-{auprc_hi:.4f}]")
        print(f"🧪 F1-score:    "f"{f1mean:.4f} ± {f1std:.4f} "f"[95% CI {f1_lo:.4f}-{f1_hi:.4f}]")
        print(f"🔁 Sensitivity:      "f"{recmean:.4f} ± {recstd:.4f} "f"[95% CI {rec_lo:.4f}-{rec_hi:.4f}]")
        print(f"🛡️ Specificity: "f"{specmean:.4f} ± {specstd:.4f} "f"[95% CI {spec_lo:.4f}-{spec_hi:.4f}]")
        print(f"➕ PPV:         "f"{ppvmean:.4f} ± {ppvstd:.4f} "f"[95% CI {ppv_lo:.4f}-{ppv_hi:.4f}]")
        print(f"➖ NPV:         "f"{npvmean:.4f} ± {npvstd:.4f} "f"[95% CI {npv_lo:.4f}-{npv_hi:.4f}]")
        print(f"🧮 MCC:         "f"{mccmean:.4f} ± {mccstd:.4f} "f"[95% CI {mcc_lo:.4f}-{mcc_hi:.4f}]")
        print(f"🎯 Accuracy:    "f"{accmean:.4f} ± {accstd:.4f} "f"[95% CI {acc_lo:.4f}-{acc_hi:.4f}]")
        print(f"🎲 Brier:       "f"{brier:.4f}")
        print("\n📊 OOF metrics:")
        print(f"AUC OOF:         {auc_oof:.4f}")

        print(f"AUPRC OOF:       {auprc_oof:.4f}")
        print(f"Sensitivity OOF:      {recall_oof:.4f}")
        print(f"Specificity OOF: {specificity_oof:.4f}")
        print(f"PPV OOF:         {ppv_oof:.4f}")
        print(f"NPV OOF:         {npv_oof:.4f}")
        print(f"\n🧩 Confusion Matrix finale (OOF):\n"f"{cm_final}")
        print("\n📉 Calibration curve:")
        print(f"prob_pred = {prob_pred}")
        print(f"prob_true = {prob_true}")

    # =========================================================
    # RESULTS DICTIONARY
    # =========================================================

    results = {
        "AUC": rocauc_scores,
        "AUPRC": auprc_scores,
        "F1": f1_scores,
        "Sensitivity": recall_scores,
        "Specificity": specificity_scores,
        "PPV": ppv_scores,
        "NPV": npv_scores,
        "MCC": mcc_scores,
        "Accuracy": acc_scores,
        "Brier": brier,
        "CM": cm_final,
        "OOF AUC": auc_oof,
        "OOF AUPRC": auprc_oof,
        "OOF Sensitivity": recall_oof,
        "OOF Specificity": specificity_oof,
        "OOF PPV": ppv_oof,
        "OOF NPV": npv_oof,
        "CI95": {
            "AUC": (auc_lo,auc_hi),
            "AUPRC": (auprc_lo,auprc_hi),
            "F1": (f1_lo,f1_hi),
            "Sensitivity": (rec_lo,rec_hi),
            "Specificity": (spec_lo,spec_hi),
            "PPV": (ppv_lo,ppv_hi),
            "NPV": (npv_lo,npv_hi),
            "MCC": (mcc_lo,mcc_hi),
            "Accuracy": (acc_lo, acc_hi)
        },
        "y_true_oof": y_valid_all,
        "y_pred_oof": y_pred_all,
        "y_prob_oof": y_prob_all,
        "y_idx" : y_idx_all,
        "calibration_prob_pred": prob_pred,
        "calibration_prob_true": prob_true
    }

    # =========================================================
    # RETURN
    # =========================================================
    if return_metrics:
        return (model,train_df,valid_df,y_df.loc[train_idx],y_df.loc[valid_idx],computational_profiles, results)
    else:
        return (model,train_df,valid_df,y_df.loc[train_idx],y_df.loc[valid_idx],computational_profiles)
        

def lgbm_cv_iter(
    embedder_configs,
    y_df,
    selected_patient_ids,
    iteration,
    n_splits=5,
    threshold=0.5,
    methods=[],
    return_metrics=False,
    random_state=42,
    validation_size=0.20,
    fold_indices_dir=None
):
    import numpy as np
    import pandas as pd
    import scipy.stats as st
    import json
    import time
    from pathlib import Path

    import lightgbm as lgb

    from tqdm.auto import tqdm
    from IPython.display import display, clear_output
    import ipywidgets as widgets

    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.metrics import (
        matthews_corrcoef,
        accuracy_score,
        roc_auc_score,
        average_precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        brier_score_loss
    )
    from sklearn.calibration import calibration_curve

    # =========================================================
    # UI
    # =========================================================

    notebook = in_notebook()

    if notebook:
        import ipywidgets as widgets

        clear_output(wait=True)

        frame_top = widgets.Output()
        frame_tqdm = widgets.Output()
        frame_plot = widgets.Output()
        frame_log = widgets.Output()

        box_cv = widgets.Tab(children=[frame_top])
        box_cv.set_title(0, "🌀 Cross Validation")

        a_plot = widgets.Tab(children=[frame_plot])
        a_tqdm = widgets.Tab(children=[frame_tqdm])

        a_plot.set_title(0, "📉 Training Plot")
        a_tqdm.set_title(0, "📉 Training Loop")

        a_out = widgets.Tab(children=[frame_log])
        a_out.set_title(0, "📝 Output")

        box_train = widgets.VBox([a_plot, a_tqdm])

        display(box_cv, box_train, a_out)
    else:
        frame_top = TerminalOutput()
        frame_tqdm = TerminalOutput()
        frame_plot = TerminalOutput()
        frame_log = TerminalOutput()
    # =========================================================
    # LIGHTGBM PARAMETERS
    # =========================================================

    lgb_params = {
        "objective": "binary",
        "metric": "None",
        "verbosity": -3,
        "is_unbalance": True,
        "seed": random_state,
        "deterministic": True,
        "force_col_wise": True,
    }

    # =========================================================
    # CUSTOM MCC FOR LIGHTGBM
    # =========================================================

    def mcc_eval(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred_labels = (y_pred > threshold).astype(int)
        mcc = matthews_corrcoef(y_true,y_pred_labels)
        return "MCC", mcc, True

    # =========================================================
    # 95% CONFIDENCE INTERVAL
    # =========================================================

    def mean_std_ci(values, confidence=0.95):

        values = np.asarray(values,dtype=float)
        # elimina eventuali nan
        values = values[~np.isnan(values)]

        if len(values) == 0:
            return (np.nan,np.nan,np.nan,np.nan)
        mean = np.mean(values)
        if len(values) < 2:
            return (mean,np.nan,np.nan,np.nan)
        std = np.std(values,ddof=1)
        sem = st.sem(values)
        h = st.t.ppf((1 + confidence) / 2, df=len(values) - 1) * sem
        return (mean,std,mean - h,mean + h)

    # =========================================================
    # METRIC LISTS
    # =========================================================

    mcc_scores = []
    acc_scores = []

    rocauc_scores = []
    auprc_scores = []

    recall_scores = []
    f1_scores = []

    specificity_scores = []
    ppv_scores = []
    npv_scores = []

    # OOF predictions
    y_idx_all = []
    y_valid_all = []
    y_pred_all = []
    y_prob_all = []
    computational_profiles = {
        method: {
            "emb_train_time_s": [],
            "emb_inference_time_s": [],
            "emb_inference_ms_per_patient": [],
            "emb_cpu_memory_delta_mb": [],
            "emb_peak_gpu_memory_mb": [],
            "emb_n_trainable_params": [],
            "emb_n_total_params": [],
            "clf_train_time_s": [],
            "clf_inference_time_s": [],
            "clf_inference_ms_per_patient": [],
            "clf_cpu_memory_delta_mb": [],
        }
        for method in methods
    }

    # =========================================================
    # CROSS VALIDATION
    # =========================================================

    if not methods:
        raise ValueError("Specificare almeno un metodo in methods.")
    if not 0 < validation_size < 1:
        raise ValueError("validation_size deve essere compreso tra 0 e 1.")
    if not isinstance(iteration, (int, np.integer)) or iteration < 0:
        raise ValueError("iteration deve essere un intero non negativo.")
    if not isinstance(random_state, (int, np.integer)) or random_state < 0:
        raise ValueError("random_state deve essere un intero non negativo.")

    # Seed diversi per ripetizione, riproducibili a parità di input e ordine.
    repetition_seed = int(np.random.SeedSequence(
        [int(random_state), int(iteration)]
    ).generate_state(1)[0])
    lgb_params["seed"] = repetition_seed
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repetition_seed)
    selected_patient_ids = np.asarray(selected_patient_ids)
    if not pd.Index(selected_patient_ids).is_unique:
        raise ValueError("selected_patient_ids must contains unique ids.")
    if not y_df.index.is_unique:
        raise ValueError("Index of y_df must be unique.")
    y_df = y_df.loc[selected_patient_ids]
    y = (y_df.values.astype(np.float32).ravel())
    if len(y) != len(selected_patient_ids) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_df deve contenere una sola etichetta binaria per paziente.")
    if np.unique(y).size != 2 or np.min(np.bincount(y.astype(int))) < n_splits:
        raise ValueError("Ogni classe deve contenere almeno n_splits pazienti.")

    # Precalcola e salva tutti gli indici prima dell'addestramento.
    fold_splits = list(skf.split(selected_patient_ids,y))
    inner_splits = []
    fold_indices = {
        "iteration": int(iteration),
        "random_state": int(random_state),
        "repetition_seed": repetition_seed,
        "n_splits": int(n_splits),
        "validation_size": validation_size,
        "position_reference": "selected_patient_ids nell'ordine fornito",
        "selected_patient_ids": selected_patient_ids.tolist(),
        "folds": []
    }
    for fold, (t_idx, v_idx) in enumerate(fold_splits):
        inner_seed = int(np.random.SeedSequence(
            [int(random_state), int(iteration), fold + 1]
        ).generate_state(1)[0])
        try:
            fit_pos, early_stop_pos = train_test_split(t_idx, test_size=validation_size,
                                                    random_state=inner_seed,stratify=y[t_idx])
        except ValueError as exc:
            raise ValueError(
                f"Fold {fold + 1}: split interno stratificato impossibile. "
                "Controllare numerosità delle classi e validation_size."
            ) from exc
        inner_splits.append((fit_pos, early_stop_pos, inner_seed))
        fold_indices["folds"].append({
            "fold": fold + 1,
            "inner_seed": inner_seed,
            "outer_train_positions": t_idx.tolist(),
            "outer_test_positions": v_idx.tolist(),
            "lgb_train_positions": fit_pos.tolist(),
            "early_stopping_positions": early_stop_pos.tolist(),
            "outer_train_ids": selected_patient_ids[t_idx].tolist(),
            "outer_test_ids": selected_patient_ids[v_idx].tolist(),
            "lgb_train_ids": selected_patient_ids[fit_pos].tolist(),
            "early_stopping_ids": selected_patient_ids[early_stop_pos].tolist()
        })

    with frame_top:

        cvfolding = tqdm(fold_splits,total=n_splits,desc="Folds", position=0, leave=True)
        for fold, (t_idx, v_idx) in enumerate(cvfolding):

            train_idx = selected_patient_ids[t_idx]
            valid_idx = selected_patient_ids[v_idx]
            train_features_list = []
            valid_features_list = []
            fold_profiles = {}
            fit_pos, early_stop_pos, inner_seed = inner_splits[fold]
            fit_idx = selected_patient_ids[fit_pos]
            early_stop_idx = selected_patient_ids[early_stop_pos]

            # =================================================
            # EMBEDDERS
            # =================================================

            for method in methods:

                cfg = embedder_configs[method]
                func = cfg["func"]
                kwargs = (cfg["kwargs"].copy())
                kwargs["train_idx"] = train_idx
                kwargs["valid_idx"] = valid_idx
                result = func(**kwargs,frame_tqdm=frame_tqdm,frame_plot=frame_plot)

                if len(result) == 3:
                    train_emb_df, valid_emb_df, profile = result
                else:
                    train_emb_df, valid_emb_df = result
                    profile = {
                        "emb_train_time_s": np.nan,
                        "emb_inference_time_s": np.nan,
                        "emb_inference_ms_per_patient": np.nan,
                        "emb_cpu_memory_delta_mb": np.nan,
                        "emb_peak_gpu_memory_mb": np.nan,
                        "emb_n_trainable_params": np.nan,
                        "emb_n_total_params": np.nan,
                        "clf_train_time_s": np.nan,
                        "clf_inference_time_s": np.nan,
                        "clf_inference_ms_per_patient": np.nan,
                        "clf_cpu_memory_delta_mb": np.nan,
                    }

                fold_profiles[method] = dict(profile)
                train_features_list.append(train_emb_df)
                valid_features_list.append(valid_emb_df)
                

            # =================================================
            # CONCAT FEATURES
            # =================================================

            train_df = pd.concat(train_features_list,axis=1).loc[train_idx]
            valid_df = pd.concat(valid_features_list,axis=1).loc[valid_idx]
            y_train = (y_df.loc[train_idx].values.astype(np.float32).ravel())
            y_valid = (y_df.loc[valid_idx].values.astype(np.float32).ravel())

            # =================================================
            # SAVE TRAIN / TEST MATRICES
            # =================================================

            if fold_indices_dir is not None:
                train_export = train_df.copy()
                test_export = valid_df.copy()
                train_export["target"] = y_train
                test_export["target"] = y_valid
                train_export.index = pd.MultiIndex.from_arrays(
                    [[fold + 1] * len(train_export), train_export.index],
                    names=["fold", "patient_id"]
                )
                test_export.index = pd.MultiIndex.from_arrays(
                    [[fold + 1] * len(test_export), test_export.index],
                    names=["fold", "patient_id"]
                )
                train_export.to_csv(f"{fold_indices_dir}/train_{'+'.join(methods)}_{iteration}.csv",
                                    mode="w" if fold == 0 else "a",header=(fold == 0),index=True)
                test_export.to_csv(f"{fold_indices_dir}/test_{'+'.join(methods)}_{iteration}.csv",
                                    mode="w" if fold == 0 else "a",header=(fold == 0),index=True)
                        
            # =================================================
            # LIGHTGBM DATASET
            # =================================================

            # Solo il training esterno viene suddiviso per LightGBM.
            train_fit = train_df.loc[fit_idx]
            valid_test = train_df.loc[early_stop_idx]
            y_train_fit = y_df.loc[fit_idx].values.astype(np.float32).ravel()
            y_valid_test = y_df.loc[early_stop_idx].values.astype(np.float32).ravel()

            lgb_params["seed"] = inner_seed
            train_data = lgb.Dataset(train_fit,label=y_train_fit)
            valid_data = lgb.Dataset(valid_test,label=y_valid_test,reference=train_data)

            # =================================================
            # TRAIN MODEL
            # =================================================

            with frame_log:
                cpu_mem_before = get_process_memory_mb()    # Perf measure
                t0 = time.perf_counter()                    # Perf measure
                model = lgb.train(lgb_params,train_data,num_boost_round=1000,
                    valid_sets=[valid_data],
                    feval=mcc_eval,
                    callbacks=[lgb.early_stopping(50, verbose=False),lgb.log_evaluation(0)])
                train_time = time.perf_counter() - t0                      # Perf measure

            # =================================================
            # PREDICTIONS
            # =================================================

            t0 = time.perf_counter()     # Perf measure

            y_prob = model.predict(valid_df,num_iteration=model.best_iteration)
            y_pred_labels = (y_prob > threshold).astype(int)
            inference_time = time.perf_counter() - t0                  # Perf measure
            cpu_mem_after = get_process_memory_mb()                    # Perf measure
            # Il classificatore è condiviso dai metodi concatenati.
            for method in methods:
                profile = fold_profiles[method]
                profile["clf_train_time_s"] = train_time
                profile["clf_inference_time_s"] = inference_time
                profile["clf_inference_ms_per_patient"] = inference_time / len(valid_idx) * 1000
                profile["clf_cpu_memory_delta_mb"] = cpu_mem_after - cpu_mem_before

                for key in computational_profiles[method]:
                    computational_profiles[method][key].append(profile.get(key,np.nan))
            # =================================================
            # STANDARD METRICS

            # =================================================

            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid,y_pred_labels)
            rocauc = roc_auc_score(y_valid,y_prob)
            auprc = average_precision_score(y_valid,y_prob)
            recall = recall_score(y_valid,y_pred_labels,zero_division=0)
            f1 = f1_score(y_valid,y_pred_labels,zero_division=0)

            # =================================================
            # CONFUSION-MATRIX METRICS
            # =================================================

            tn, fp, fn, tp = confusion_matrix(y_valid,y_pred_labels,labels=[0, 1]).ravel()
            specificity = ( tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            ppv = (tp / (tp + fp) if (tp + fp) > 0 else np.nan)
            npv = (tn / (tn + fn) if (tn + fn) > 0 else np.nan)

            # =================================================
            # STORE FOLD METRICS

            # =================================================

            mcc_scores.append(mcc)
            acc_scores.append(acc)
            rocauc_scores.append(rocauc)
            auprc_scores.append(auprc)
            recall_scores.append(recall)
            f1_scores.append(f1)
            specificity_scores.append(specificity)
            ppv_scores.append(ppv)
            npv_scores.append(npv)

            # =================================================
            # STORE OOF PREDICTIONS
            # =================================================

            y_idx_all.extend(valid_idx)
            y_valid_all.extend(y_valid)
            y_pred_all.extend(y_pred_labels)
            y_prob_all.extend(y_prob)

            # =================================================
            # PROGRESS BAR
            # =================================================

            cvfolding.set_postfix({
                "Fold": fold + 1,
                "AUC": f"{rocauc:.4f}",
                "AUPRC": f"{auprc:.4f}",
                "Sensitivity": f"{recall:.4f}",
                "Spec": f"{specificity:.4f}",
                "PPV": f"{ppv:.4f}",
                "NPV": f"{npv:.4f}",
                "F1": f"{f1:.4f}",
                "MCC": f"{mcc:.4f}"
            })
    # =========================================================
    # FINAL OOF ANALYSIS
    # =========================================================

    with frame_log:

        print(f"Train data {train_df.shape} - "f"Test data {valid_df.shape}")
        y_valid_all = np.asarray(y_valid_all)
        y_pred_all = np.asarray(y_pred_all)
        y_prob_all = np.asarray(y_prob_all)

        # =====================================================
        # AGGREGATED CONFUSION MATRIX
        # =====================================================

        cm_final = confusion_matrix(y_valid_all,y_pred_all,labels=[0, 1])
        tn, fp, fn, tp = (cm_final.ravel())
        specificity_oof = (tn / (tn + fp) if (tn + fp) > 0 else np.nan)
        ppv_oof = (tp / (tp + fp) if (tp + fp) > 0 else np.nan)
        npv_oof = (tn / (tn + fn) if (tn + fn) > 0 else np.nan)
        recall_oof = (tp / (tp + fn) if (tp + fn) > 0 else np.nan)

        # =====================================================
        # OOF AUC / AUPRC
        # =====================================================

        try:
            auc_oof = roc_auc_score(y_valid_all,y_prob_all)
        except Exception:
            auc_oof = np.nan
        try:
            auprc_oof = average_precision_score(y_valid_all,y_prob_all)
        except Exception:
            auprc_oof = np.nan

        # =====================================================
        # CALIBRATION
        # =====================================================

        prob_true, prob_pred = calibration_curve(y_valid_all,y_prob_all,n_bins=10,strategy="uniform")
        brier = brier_score_loss(y_valid_all,y_prob_all)

        # =====================================================
        # MEAN / STD / CI95
        # =====================================================

        (aucmean,aucstd,auc_lo,auc_hi) = mean_std_ci(rocauc_scores)
        (auprcmean,auprcstd,auprc_lo,auprc_hi) = mean_std_ci(auprc_scores)
        (f1mean,f1std,f1_lo,f1_hi) = mean_std_ci(f1_scores)
        (recmean,recstd,rec_lo,rec_hi) = mean_std_ci(recall_scores)
        (specmean,specstd,spec_lo,spec_hi) = mean_std_ci(specificity_scores)
        (ppvmean,ppvstd,ppv_lo,ppv_hi) = mean_std_ci(ppv_scores)
        (npvmean,npvstd,npv_lo,npv_hi) = mean_std_ci(npv_scores)
        (mccmean,mccstd,mcc_lo,mcc_hi) = mean_std_ci(mcc_scores)
        (accmean,accstd,acc_lo,acc_hi) = mean_std_ci(acc_scores)

        # =====================================================
        # OUTPUT
        # =====================================================

        print(f"\n📊 Experiment {iteration}. Risultati medi su {n_splits} fold:")
        print(f"📈 AUC:         "f"{aucmean:.4f} ± {aucstd:.4f} "f"[95% CI {auc_lo:.4f}-{auc_hi:.4f}]")
        print(f"📉 AUPRC:       "f"{auprcmean:.4f} ± {auprcstd:.4f} "f"[95% CI {auprc_lo:.4f}-{auprc_hi:.4f}]")
        print(f"🧪 F1-score:    "f"{f1mean:.4f} ± {f1std:.4f} "f"[95% CI {f1_lo:.4f}-{f1_hi:.4f}]")
        print(f"🔁 Sensitivity:      "f"{recmean:.4f} ± {recstd:.4f} "f"[95% CI {rec_lo:.4f}-{rec_hi:.4f}]")
        print(f"🛡️ Specificity: "f"{specmean:.4f} ± {specstd:.4f} "f"[95% CI {spec_lo:.4f}-{spec_hi:.4f}]")
        print(f"➕ PPV:         "f"{ppvmean:.4f} ± {ppvstd:.4f} "f"[95% CI {ppv_lo:.4f}-{ppv_hi:.4f}]")
        print(f"➖ NPV:         "f"{npvmean:.4f} ± {npvstd:.4f} "f"[95% CI {npv_lo:.4f}-{npv_hi:.4f}]")
        print(f"🧮 MCC:         "f"{mccmean:.4f} ± {mccstd:.4f} "f"[95% CI {mcc_lo:.4f}-{mcc_hi:.4f}]")
        print(f"🎯 Accuracy:    "f"{accmean:.4f} ± {accstd:.4f} "f"[95% CI {acc_lo:.4f}-{acc_hi:.4f}]")
        print(f"🎲 Brier:       "f"{brier:.4f}")
        print("\n📊 OOF metrics:")
        print(f"AUC OOF:         {auc_oof:.4f}")

        print(f"AUPRC OOF:       {auprc_oof:.4f}")
        print(f"Sensitivity OOF:      {recall_oof:.4f}")
        print(f"Specificity OOF: {specificity_oof:.4f}")
        print(f"PPV OOF:         {ppv_oof:.4f}")
        print(f"NPV OOF:         {npv_oof:.4f}")
        print(f"\n🧩 Confusion Matrix finale (OOF):\n"f"{cm_final}")
        print("\n📉 Calibration curve:")
        print(f"prob_pred = {prob_pred}")
        print(f"prob_true = {prob_true}")

    # =========================================================
    # RESULTS DICTIONARY
    # =========================================================

    results = {
        "AUC": rocauc_scores,
        "AUPRC": auprc_scores,
        "F1": f1_scores,
        "Sensitivity": recall_scores,
        "Specificity": specificity_scores,
        "PPV": ppv_scores,
        "NPV": npv_scores,
        "MCC": mcc_scores,
        "Accuracy": acc_scores,
        "Brier": brier,
        "CM": cm_final,
        "OOF AUC": auc_oof,
        "OOF AUPRC": auprc_oof,
        "OOF Sensitivity": recall_oof,
        "OOF Specificity": specificity_oof,
        "OOF PPV": ppv_oof,
        "OOF NPV": npv_oof,
        "CI95": {
            "AUC": (auc_lo,auc_hi),
            "AUPRC": (auprc_lo,auprc_hi),
            "F1": (f1_lo,f1_hi),
            "Sensitivity": (rec_lo,rec_hi),
            "Specificity": (spec_lo,spec_hi),
            "PPV": (ppv_lo,ppv_hi),
            "NPV": (npv_lo,npv_hi),
            "MCC": (mcc_lo,mcc_hi),
            "Accuracy": (acc_lo, acc_hi)
        },
        "y_true_oof": y_valid_all,
        "y_pred_oof": y_pred_all,
        "y_prob_oof": y_prob_all,
        "y_idx" : y_idx_all,
        "calibration_prob_pred": prob_pred,
        "calibration_prob_true": prob_true
    }

    # =========================================================
    # RETURN
    # =========================================================
    if return_metrics:
        return (model,train_df,valid_df,y_df.loc[train_idx],y_df.loc[valid_idx],computational_profiles, results)
    else:
        return (model,train_df,valid_df,y_df.loc[train_idx],y_df.loc[valid_idx],computational_profiles)
            
            
#----------------------------------------------------------------------------------
# CV folding with E2E classifier
#----------------------------------------------------------------------------------

def e2e_cv(embedder_configs, 
            y_df, 
            selected_patient_ids,
            n_splits=5,
            return_metrics = True,
            method="LSTM"):

    notebook = in_notebook()
    if notebook:
        import ipywidgets as widgets

        clear_output(wait=True)

        frame_top = widgets.Output()
        frame_tqdm = widgets.Output()
        frame_plot = widgets.Output()
        frame_log = widgets.Output()

        box_cv = widgets.Tab(children=[frame_top])
        box_cv.set_title(0, "🌀 Cross Validation")

        a_plot = widgets.Tab(children=[frame_plot])
        a_tqdm = widgets.Tab(children=[frame_tqdm])

        a_plot.set_title(0, "📉 Training Plot")
        a_tqdm.set_title(0, "📉 Training Loop")

        a_out = widgets.Tab(children=[frame_log])
        a_out.set_title(0, "📝 Output")

        box_train = widgets.VBox([a_plot, a_tqdm])

        display(box_cv, box_train, a_out)
    else:
        frame_top = TerminalOutput()
        frame_tqdm = TerminalOutput()
        frame_plot = TerminalOutput()
        frame_log = TerminalOutput()
        
    mcc_scores = []
    acc_scores = []
    rocauc_scores = []
    prec_scores = []
    recall_scores = []
    f1_scores = []
    auprc_scores = []
    specificity_scores = []
    ppv_scores = []
    npv_scores = []
            
    y_valid_all = []
    y_pred_all = []
    y_prob_all = []
    y_idx_all = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_df = y_df.loc[selected_patient_ids]
    y = y_df.values.astype(np.float32).ravel()
    with frame_top:
        cvfolding = tqdm(skf.split(selected_patient_ids, y), total=n_splits, desc="Folds", position=0, leave=True)
        for fold, (t_idx, v_idx) in enumerate(cvfolding):
            train_idx = selected_patient_ids[t_idx]
            valid_idx = selected_patient_ids[v_idx]

            y_valid = y_df.loc[valid_idx].values.astype(np.float32).ravel()
            cfg = embedder_configs[method]
            func = cfg["clf"]
            kwargs = cfg["kwargs"].copy()  # facciamo una copia per sicurezza

            # Aggiungiamo train/valid idx dinamicamente
            kwargs["train_idx"] = train_idx
            kwargs["valid_idx"] = valid_idx

            # =================================================
            # TRAIN and PREDICTIONS
            # =================================================
            y_pred_labels, y_prob = func(**kwargs, frame_tqdm=frame_tqdm, frame_plot=frame_plot)

            # =================================================
            # STANDARD METRICS
            # =================================================

            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid,y_pred_labels)
            rocauc = roc_auc_score(y_valid,y_prob[:,1])
            auprc = average_precision_score(y_valid,y_prob[:,1])
            recall = recall_score(y_valid,y_pred_labels)
            f1 = f1_score(y_valid,y_pred_labels)

            # =================================================
            # CONFUSION-MATRIX METRICS
            # =================================================

            tn, fp, fn, tp = confusion_matrix(y_valid,y_pred_labels,labels=[0, 1]).ravel()
            specificity = ( tn / (tn + fp) if (tn + fp) > 0 else np.nan)
            ppv = (tp / (tp + fp) if (tp + fp) > 0 else np.nan)
            npv = (tn / (tn + fn) if (tn + fn) > 0 else np.nan)

            # =================================================
            # STORE FOLD METRICS
            # =================================================

            mcc_scores.append(mcc)
            acc_scores.append(acc)
            rocauc_scores.append(rocauc)
            auprc_scores.append(auprc)
            recall_scores.append(recall)
            f1_scores.append(f1)
            specificity_scores.append(specificity)
            ppv_scores.append(ppv)
            npv_scores.append(npv)

            # =================================================
            # STORE OOF PREDICTIONS
            # =================================================

            # Metriche fold
            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid, y_pred_labels)
            rocauc = roc_auc_score(y_valid, y_prob[:, 1])
            prec = precision_score(y_valid, y_pred_labels)
            recall = recall_score(y_valid, y_pred_labels)
            f1 = f1_score(y_valid, y_pred_labels)

            # Salva
            mcc_scores.append(mcc)
            acc_scores.append(acc)
            rocauc_scores.append(rocauc)
            prec_scores.append(prec)
            recall_scores.append(recall)
            f1_scores.append(f1)
            y_idx_all.extend(valid_idx)
            y_valid_all.extend(y_valid)
            y_pred_all.extend(y_pred_labels)
            y_prob_all.extend(y_prob[:, 1])   # <-- AGGIUNTO: serve per calibration curve

            # 🧠 Aggiorna tqdm
            cvfolding.set_postfix({
                "Fold": fold + 1,
                "MCC": f"{mcc:.4f}",
                "AUC": f"{rocauc:.4f}",
                "Accuracy": f"{acc:.4f}",
                "F1": f"{f1:.4f}"
            })

    # Final confusion matrix
    with frame_log:
        cm_final = confusion_matrix(y_valid_all, y_pred_all)
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_valid_all, y_prob_all, n_bins=10,
            strategy="uniform"   # oppure "quantile"
        )
        # Brier score
        brier = brier_score_loss(y_valid_all, y_prob_all)
        aucmean, aucstd = np.mean(rocauc_scores), np.std(rocauc_scores)
        f1mean, f1std = np.mean(f1_scores), np.std(f1_scores)
        precmean, precstd = np.mean(prec_scores), np.std(prec_scores)
        recmean, recstd = np.mean(recall_scores), np.std(recall_scores)
        mccmean, mccstd = np.mean(mcc_scores), np.std(mcc_scores)
        accmean, accstd = np.mean(acc_scores), np.std(acc_scores)
        print(f"\n📊 Risultati medi su {n_splits} fold:")
        print(f"📈 AUC:      {aucmean:.4f} ± {aucstd:.4f}")
        print(f"🧪 F1-score: {f1mean:.4f} ± {f1std:.4f}")
        print(f"⚖️ Precision:{precmean:.4f} ± {precstd:.4f}")
        print(f"🔁 Sensitivity:   {recmean:.4f} ± {recstd:.4f}")
        print(f"🧮 MCC:      {mccmean:.4f} ± {mccstd:.4f}")
        print(f"🎯 Accuracy: {accmean:.4f} ± {accstd:.4f}")
        print(f"🎲 Brier:     {brier:.4f}")

        print(f"\n🧩 Confusion Matrix finale (aggregata):\n{cm_final}")
        print(f"\n📉 Calibration curve:")
        print(f"prob_pred = {prob_pred}")
        print(f"prob_true = {prob_true}")

    results = {
        "AUC": rocauc_scores,
        "AUPRC": auprc_scores,
        "F1": f1_scores,
        "Sensitivity": recall_scores,
        "Specificity": specificity_scores,
        "PPV": ppv_scores,
        "NPV": npv_scores,
        "MCC": mcc_scores,
        "Accuracy": acc_scores,
        "Brier": brier,
        "CM": cm_final,
        "y_true_oof": y_valid_all,
        "y_pred_oof": y_pred_all,
        "y_prob_oof": y_prob_all,
        "y_idx" : y_idx_all,
        "calibration_prob_pred": prob_pred,
        "calibration_prob_true": prob_true
    }

    if return_metrics:
        return (y_df.loc[train_idx],y_df.loc[valid_idx],results)
    else:
        return (y_df.loc[train_idx],y_df.loc[valid_idx])
