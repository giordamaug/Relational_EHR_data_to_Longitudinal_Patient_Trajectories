import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Patch
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
import itertools
from sklearn.metrics import brier_score_loss

# Radar plot
def plot_radar(df, metrics):
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for idx, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=idx)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)])

    ax.grid(True)
    #plt.title("Model Comparison Radar Plot", pad=20)
    #ax.legend(loc='best')
    plt.legend(loc="upper right", bbox_to_anchor=(0.75, 1))
    plt.tight_layout()
    plt.show()
    return fig

def plot_group_distribution_with_event_boxplot(df, groupby="primary_disease_area", label_desc=None):
    df = df.copy()

    if label_desc is not None:
        mapping = {float(i): label for i, label in enumerate(label_desc)}
        label_order = [label_desc[int(i)] for i in df[groupby].dropna().unique().tolist()]
        df[groupby] = df[groupby].astype(float).map(mapping)
    else:
        label_order = df[groupby].dropna().unique().tolist()

    df['event_length'] = df['events'].apply(len)

    df = df.dropna(subset=[groupby, 'event_length'])

    counts = df[groupby].value_counts().reindex(label_order, fill_value=0)

    values = counts.values
    labels = counts.index

    if values.sum() == 0:
        raise ValueError(
            f"Nessun dato valido per {groupby}. "
            f"Controlla il mapping e i valori originali della colonna."
        )

    palette = sns.color_palette("tab10", len(label_order))
    palette_dict = dict(zip(label_order, palette))

    def autopct_abs(pct):
        total = np.sum(values)
        val = int(round(pct * total / 100.0))
        return f'{val}' if val > 0 else ''

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

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
        order=label_order,
        hue=groupby,
        palette=palette_dict,
        dodge=False,
        legend=False,
        ax=axes[1]
    )

    axes[1].set_title("(B) Event Length Distribution")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    
def plot_heatmap(X_df, y_df, Xv_df, ):
    mpl.rcParams['font.size'] = 20  # Dimensione base
    mpl.rcParams['axes.titlesize'] = 22  # Titolo degli assiimport matplotlib as mpl
    mpl.rcParams['axes.labelsize'] = 22  # Etichette assi
    mpl.rcParams['xtick.labelsize'] = 20  # Etichette asse x
    mpl.rcParams['ytick.labelsize'] = 14.5  # Etichette asse y
    mpl.rcParams['legend.fontsize'] = 16  # Font della legenda
    mpl.rcParams['legend.title_fontsize'] = 16  # Titolo legenda

    # Step 2: Rinomina le colonne nel DataFrame
    X_df_traslated = pd.concat([X_df, Xv_df], axis=0)

    # Aggiungi la colonna target temporaneamente per colorare le righe
    X_df_traslated['target'] = y_df.loc[X_df_traslated.index, 'target']
    X_sorted = X_df_traslated.sort_values('target', ascending=False)
    targets_sorted = X_sorted['target'].values
    X_sorted = X_sorted.drop(columns='target')


    X_norm = X_sorted.copy()
    X_norm_t = X_norm.T
    scaler = MinMaxScaler()
    X_norm_t.iloc[:, :] = scaler.fit_transform(X_norm_t)


    # Applica la selezione sulle righe (trasponi, applica, poi ritrasponi)
    X_filtered = X_norm_t[~(X_norm_t == 0).all(axis=1)]


    # 2. (Facoltativo) Rimuovi righe duplicate
    # len(X_final) = X_filtered.drop_duplicates()


    # Crea una palette per distinguere classi (ad esempio 1=red, 0=blue)
    row_colors = ['yellow' if y_df.loc[idx, 'target'] == 1 else 'orange' for idx in X_filtered.columns]


    legend_elements = [Patch(facecolor='yellow', edgecolor='k', label='Class 1 = With Infectious Events'),
                    Patch(facecolor='orange', edgecolor='k', label='Class 0 = Without Infectious Events')]

    g = sns.clustermap(
        X_filtered,
        cmap='RdBu_r',
        cbar_pos=(0.03, 0.1, 0.02, 0.18),
        figsize=(16, 14),
        col_colors=row_colors,
        xticklabels=False,
        yticklabels=True,
        col_cluster=False,
        row_cluster=True,
        metric='euclidean',
        method='ward'
    )


    # Calcola quante righe hanno target 1 (sono state messe in alto)
    n_positive = sum(y_df.loc[X_filtered.columns, 'target'] == 1)

    # Aggiungi linea orizzontale tra target 1 e 0
    g.ax_heatmap.axvline(x=n_positive, color='white', linestyle='-', linewidth=2)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels())
    # Aggiungi legenda
    plt.gcf().legend(handles=legend_elements, title='Target class',
                    loc='upper center', bbox_to_anchor=(0.4, 0.5))
    plt.show()
    

def plot_calibration(prob_pred, prob_true, all_runs, title="Calibration curves"):

    methods = list(prob_pred.keys())
    n_models = len(methods)

    # 🎨 colori consistenti
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(methods)}

    # Crea figura e asse
    fig, ax_cal = plt.subplots(figsize=(10, 6))
    
    # =========================
    # CALIBRATION CURVE
    # =========================
    ax_cal.plot([0, 1], [0, 1], '--', label='Perfect calibration')

    for i, m in enumerate(methods):
        x = np.array(prob_pred[m])
        y = np.array(prob_true[m])
        y_true_oof, y_prob_oof = np.array(all_runs[m][0]), np.array(all_runs[m][1]) #all_runs[m]
        brier = brier_score_loss(y_true_oof, y_prob_oof)

        color = colors[m]

        ax_cal.plot(
            x, y,
            marker='o',
            color=color,
            label=f"{m} (Brier={brier:.3f})"
        )

    ax_cal.set_title(title)
    ax_cal.set_ylabel("Fraction of positives")
    ax_cal.legend(loc="upper left")
    ax_cal.grid(True)
    plt.tight_layout()
    plt.show()
    return fig