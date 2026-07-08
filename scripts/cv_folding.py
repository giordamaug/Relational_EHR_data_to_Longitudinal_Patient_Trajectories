from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, confusion_matrix, accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score, brier_score_loss
)
import lightgbm as lgb
import numpy as np
from tqdm.notebook import tqdm
import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

#----------------------------------------------------------------------------------
# CV folding with LGBM classification
#----------------------------------------------------------------------------------

def lgbm_cv(embedder_configs, 
            y_df, 
            selected_patient_ids,
            n_splits=5,
            threshold=0.5,
            methods=[]):

    clear_output(wait=True)
    frame_top = widgets.Output() #layout={"border": "2px solid #888", "padding": "10px"}
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
    display(box_cv,box_train,a_out)
    # LGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'None',
        'verbosity': -3,
        'is_unbalance': True
        #"scale_pos_weight" : 900 / 100,
    }

    def mcc_eval(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred_labels = (y_pred > 0.5).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_labels)
        return 'MCC', mcc, True

    mcc_scores = []
    acc_scores = []
    rocauc_scores = []
    prec_scores = []
    recall_scores = []
    f1_scores = []

    y_valid_all = []
    y_pred_all = []
    y_prob_all = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_df = y_df.loc[selected_patient_ids]
    y = y_df.values.astype(np.float32).ravel()
    with frame_top:
        cvfolding = tqdm(skf.split(selected_patient_ids, y), total=n_splits, desc="Folds")
        for fold, (t_idx, v_idx) in enumerate(cvfolding):
            train_idx = selected_patient_ids[t_idx]
            valid_idx = selected_patient_ids[v_idx]

            train_features_list = []
            valid_features_list = []
            # Loop su ogni embedder
            for method in methods:
                cfg = embedder_configs[method]
                func = cfg["func"]
                kwargs = cfg["kwargs"].copy()  # facciamo una copia per sicurezza

                # Aggiungiamo train/valid idx dinamicamente
                kwargs["train_idx"] = train_idx
                kwargs["valid_idx"] = valid_idx

                # Chiamiamo la funzione in modo generico
                train_df, valid_df = func(**kwargs, frame_tqdm=frame_tqdm, frame_plot=frame_plot)

                # Appendiamo i risultati
                train_features_list.append(train_df)
                valid_features_list.append(valid_df)

            # Concatenazione di tutti gli embedding in un unico DataFrame
            train_df = pd.concat(train_features_list, axis=1)
            test_df  = pd.concat(valid_features_list, axis=1)
            y_train, y_valid = y_df.loc[train_idx].values.astype(np.float32).ravel(), y_df.loc[valid_idx].values.astype(np.float32).ravel()
            train_data = lgb.Dataset(train_df, label=y_train)
            valid_data = lgb.Dataset(test_df, label=y_valid)
            with frame_log:
                model = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[valid_data],
                    feval=mcc_eval,
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0)
                    ]
                )

            y_pred = model.predict(test_df)
            y_pred_labels = (y_pred > threshold).astype(int)

            # Metriche fold
            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid, y_pred_labels)
            rocauc = roc_auc_score(y_valid, y_pred)
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
            y_valid_all.extend(y_valid)
            y_pred_all.extend(y_pred_labels)
            y_prob_all.extend(y_pred)   # <-- AGGIUNTO: serve per calibration curve

            # 🧠 Aggiorna tqdm
            cvfolding.set_postfix({
                "Fold": fold + 1,
                "MCC": f"{mcc:.4f}",
                "AUC": f"{rocauc:.4f}",
                "Acc": f"{acc:.4f}",
                "F1": f"{f1:.4f}"
            })
            #cvfolding.update(1)

    # Final confusion matrix
    with frame_log:
        print(f"Train data {train_df.shape} - Test data {test_df.shape}")
        y_valid_all = np.array(y_valid_all)
        y_pred_all = np.array(y_pred_all)
        y_prob_all = np.array(y_prob_all)
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
        results_df = pd.DataFrame([[aucmean, aucstd, f1mean, f1std, precmean, 
                                    precstd, recmean, recstd, mccmean, mccstd, accmean, accstd, brier, cm_final]], 
                                    columns=['AUC mean', 'AUC std',
                                            'F1 mean', 'F1 std',
                                            'Prec mean', 'Prec std',
                                            'Recall mean', 'Recall std',
                                            'MCC mean', 'MCC std',
                                            'Acc mean', 'Acc std', 
                                            'Brier', "CM"], index=np.array(['+'.join(methods)]))
        print(f"\n📊 Risultati medi su {n_splits} fold:")
        print(f"📈 AUC:      {aucmean:.4f} ± {aucstd:.4f}")
        print(f"🧪 F1-score: {f1mean:.4f} ± {f1std:.4f}")
        print(f"⚖️ Precision:{precmean:.4f} ± {precstd:.4f}")
        print(f"🔁 Recall:   {recmean:.4f} ± {recstd:.4f}")
        print(f"🧮 MCC:      {mccmean:.4f} ± {mccstd:.4f}")
        print(f"🎯 Accuracy: {accmean:.4f} ± {accstd:.4f}")
        print(f"🎲 Brier:     {brier:.4f}")

        print(f"\n🧩 Confusion Matrix finale (aggregata):\n{cm_final}")
        print(f"\n📉 Calibration curve:")
        print(f"prob_pred = {prob_pred}")
        print(f"prob_true = {prob_true}")

    return results_df, model, train_df, test_df, y_df.loc[train_idx], y_df.loc[valid_idx] 

#----------------------------------------------------------------------------------
# Iterated CV folding with LGBM classification
#----------------------------------------------------------------------------------

def lgbm_cv_iter(embedder_configs, 
            y_df, 
            selected_patient_ids,
            n_splits=5,
            threshold=0.5,
            methods=[], 
            return_metrics=False, 
            random_state=42):

    clear_output(wait=True)
    frame_top = widgets.Output() #layout={"border": "2px solid #888", "padding": "10px"}
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
    display(box_cv,box_train,a_out)
    # LGBM parameters
    lgb_params = {
        'objective': 'binary',
        'metric': 'None',
        'verbosity': -3,
        'is_unbalance': True
        #"scale_pos_weight" : 900 / 100,
    }

    def mcc_eval(y_pred, dataset):
        y_true = dataset.get_label()
        y_pred_labels = (y_pred > 0.5).astype(int)
        mcc = matthews_corrcoef(y_true, y_pred_labels)
        return 'MCC', mcc, True

    mcc_scores = []
    acc_scores = []
    rocauc_scores = []
    prec_scores = []
    recall_scores = []
    f1_scores = []

    y_valid_all = []
    y_pred_all = []
    y_prob_all = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_df = y_df.loc[selected_patient_ids]
    y = y_df.values.astype(np.float32).ravel()
    with frame_top:
        cvfolding = tqdm(skf.split(selected_patient_ids, y), total=n_splits, desc="Folds")
        for fold, (t_idx, v_idx) in enumerate(cvfolding):
            train_idx = selected_patient_ids[t_idx]
            valid_idx = selected_patient_ids[v_idx]

            train_features_list = []
            valid_features_list = []
            # Loop su ogni embedder
            for method in methods:
                cfg = embedder_configs[method]
                func = cfg["func"]
                kwargs = cfg["kwargs"].copy()  # facciamo una copia per sicurezza

                # Aggiungiamo train/valid idx dinamicamente
                kwargs["train_idx"] = train_idx
                kwargs["valid_idx"] = valid_idx

                # Chiamiamo la funzione in modo generico
                train_df, valid_df = func(**kwargs, frame_tqdm=frame_tqdm, frame_plot=frame_plot)

                # Appendiamo i risultati
                train_features_list.append(train_df)
                valid_features_list.append(valid_df)

            # Concatenazione di tutti gli embedding in un unico DataFrame
            train_df = pd.concat(train_features_list, axis=1)
            test_df  = pd.concat(valid_features_list, axis=1)
            y_train, y_valid = y_df.loc[train_idx].values.astype(np.float32).ravel(), y_df.loc[valid_idx].values.astype(np.float32).ravel()
            train_data = lgb.Dataset(train_df, label=y_train)
            valid_data = lgb.Dataset(test_df, label=y_valid)
            with frame_log:
                model = lgb.train(
                    lgb_params,
                    train_data,
                    num_boost_round=1000,
                    valid_sets=[valid_data],
                    feval=mcc_eval,
                    callbacks=[
                        lgb.early_stopping(50),
                        lgb.log_evaluation(0)
                    ]
                )

            y_pred = model.predict(test_df)
            y_pred_labels = (y_pred > threshold).astype(int)

            # Metriche fold
            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid, y_pred_labels)
            rocauc = roc_auc_score(y_valid, y_pred)
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
            y_valid_all.extend(y_valid)
            y_pred_all.extend(y_pred_labels)
            y_prob_all.extend(y_pred)   # <-- AGGIUNTO: serve per calibration curve

            # 🧠 Aggiorna tqdm
            cvfolding.set_postfix({
                "Fold": fold + 1,
                "MCC": f"{mcc:.4f}",
                "AUC": f"{rocauc:.4f}",
                "Acc": f"{acc:.4f}",
                "F1": f"{f1:.4f}"
            })
            #cvfolding.update(1)

    # Final confusion matrix
    with frame_log:
        print(f"Train data {train_df.shape} - Test data {test_df.shape}")
        y_valid_all = np.array(y_valid_all)
        y_pred_all = np.array(y_pred_all)
        y_prob_all = np.array(y_prob_all)
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
        results_df = pd.DataFrame([[aucmean, aucstd, f1mean, f1std, precmean, 
                                    precstd, recmean, recstd, mccmean, mccstd, accmean, accstd, brier, cm_final]], 
                                    columns=['AUC mean', 'AUC std',
                                            'F1 mean', 'F1 std',
                                            'Prec mean', 'Prec std',
                                            'Recall mean', 'Recall std',
                                            'MCC mean', 'MCC std',
                                            'Acc mean', 'Acc std', 
                                            'Brier', "CM"], index=np.array(['+'.join(methods)]))
        print(f"\n📊 Risultati medi su {n_splits} fold:")
        print(f"📈 AUC:      {aucmean:.4f} ± {aucstd:.4f}")
        print(f"🧪 F1-score: {f1mean:.4f} ± {f1std:.4f}")
        print(f"⚖️ Precision:{precmean:.4f} ± {precstd:.4f}")
        print(f"🔁 Recall:   {recmean:.4f} ± {recstd:.4f}")
        print(f"🧮 MCC:      {mccmean:.4f} ± {mccstd:.4f}")
        print(f"🎯 Accuracy: {accmean:.4f} ± {accstd:.4f}")
        print(f"🎲 Brier:     {brier:.4f}")

        print(f"\n🧩 Confusion Matrix finale (aggregata):\n{cm_final}")
        print(f"\n📉 Calibration curve:")
        print(f"prob_pred = {prob_pred}")
        print(f"prob_true = {prob_true}")

    results = {
        "AUC": rocauc_scores,
        "F1": f1_scores,
        "Prec": prec_scores,
        "Recall": recall_scores,
        "MCC": mcc_scores,
        "Acc": acc_scores,
        "Brier": brier,
        "CM": cm_final,
        "y_true_oof": y_valid_all,
        "y_pred_oof": y_pred_all,
        "y_prob_oof": y_prob_all,
        "calibration_prob_pred": prob_pred,
        "calibration_prob_true": prob_true
    }

    if return_metrics:
        return results_df, model, train_df, test_df, y_df.loc[train_idx], y_df.loc[valid_idx], results
    else:
        return results_df, model, train_df, test_df, y_df.loc[train_idx], y_df.loc[valid_idx]

#----------------------------------------------------------------------------------
# CV folding with E2E classifier
#----------------------------------------------------------------------------------

def e2e_cv(embedder_configs, 
            y_df, 
            selected_patient_ids,
            n_splits=5,
            method="LSTM"):

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
    display(box_cv,box_train,a_out)

    mcc_scores = []
    acc_scores = []
    rocauc_scores = []
    prec_scores = []
    recall_scores = []
    f1_scores = []

    y_valid_all = []
    y_pred_all = []
    y_prob_all = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_df = y_df.loc[selected_patient_ids]
    y = y_df.values.astype(np.float32).ravel()
    with frame_top:
        cvfolding = tqdm(skf.split(selected_patient_ids, y), total=n_splits, desc="Folds")
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

            # Chiamiamo la funzione in modo generico
            y_pred_labels, y_prob = func(**kwargs, frame_tqdm=frame_tqdm, frame_plot=frame_plot)

            # Metriche fold
            mcc = matthews_corrcoef(y_valid, y_pred_labels)
            acc = accuracy_score(y_valid, y_pred_labels)
            rocauc = roc_auc_score(y_valid, y_prob[:, 1])
            #rocauc = roc_auc_score(y_valid, y_prob, multi_class='ovr')
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
            y_valid_all.extend(y_valid)
            y_pred_all.extend(y_pred_labels)
            y_prob_all.extend(y_prob[:, 1].flatten())   # <-- AGGIUNTO: serve per calibration curve

            # 🧠 Aggiorna tqdm
            cvfolding.set_postfix({
                "Fold": fold + 1,
                "MCC": f"{mcc:.4f}",
                "AUC": f"{rocauc:.4f}",
                "Acc": f"{acc:.4f}",
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
        results_df = pd.DataFrame([[aucmean, aucstd, f1mean, f1std, precmean, 
                                    precstd, recmean, recstd, mccmean, mccstd, accmean, accstd, brier, cm_final]], 
                                    columns=['AUC mean', 'AUC std',
                                            'F1 mean', 'F1 std',
                                            'Prec mean', 'Prec std',
                                            'Recall mean', 'Recall std',
                                            'MCC mean', 'MCC std',
                                            'Acc mean', 'Acc std', "Brier", "CM"], index=np.array([method]))
        print(f"\n📊 Risultati medi su {n_splits} fold:")
        print(f"📈 AUC:      {aucmean:.4f} ± {aucstd:.4f}")
        print(f"🧪 F1-score: {f1mean:.4f} ± {f1std:.4f}")
        print(f"⚖️ Precision:{precmean:.4f} ± {precstd:.4f}")
        print(f"🔁 Recall:   {recmean:.4f} ± {recstd:.4f}")
        print(f"🧮 MCC:      {mccmean:.4f} ± {mccstd:.4f}")
        print(f"🎯 Accuracy: {accmean:.4f} ± {accstd:.4f}")
        print(f"🎲 Brier:     {brier:.4f}")

        print(f"\n🧩 Confusion Matrix finale (aggregata):\n{cm_final}")
        print(f"\n📉 Calibration curve:")
        print(f"prob_pred = {prob_pred}")
        print(f"prob_true = {prob_true}")

    return results_df, y_df.loc[train_idx], y_df.loc[valid_idx] 
