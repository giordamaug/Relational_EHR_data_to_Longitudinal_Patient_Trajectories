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

def distrib_summary(df_orig, disease_mapper):

    df = df_orig.copy()

    df.replace(disease_mapper, inplace=True)
    df.replace({"is_alive?": {"n/a": "YES"}}, inplace=True)

    df["dead"] = df["is_alive?"].eq("NO")
    df["splen"] = df["is_splenectomized?"].eq("YES")
    df["nosplen"] = df["is_splenectomized?"].ne("YES")

    summary = (
        df.groupby("base_pathology_area")
        .agg(
            Total=("base_pathology_area", "size"),

            Splen=("splen", "sum"),
            NoSplen=("nosplen", "sum"),

            SplenDeath=("dead", lambda x: (x & df.loc[x.index, "splen"]).sum()),
            NoSplenDeath=("dead", lambda x: (x & df.loc[x.index, "nosplen"]).sum()),

            Male=("gender", lambda x: x.eq("M").sum()),
            Female=("gender", lambda x: x.eq("F").sum()),

            MaleDeath=("dead", lambda x: (x & df.loc[x.index, "gender"].eq("M")).sum()),
            FemaleDeath=("dead", lambda x: (x & df.loc[x.index, "gender"].eq("F")).sum()),

            TotalDeaths=("dead", "sum")
        )
        .reset_index()
        .rename(columns={"base_pathology_area": "Disease"})
    )

    summary["MortSplen"] = summary["SplenDeath"] / summary["Splen"].replace(0, np.nan) * 100
    summary["MortNoSplen"] = summary["NoSplenDeath"] / summary["NoSplen"].replace(0, np.nan) * 100

    summary["MortMale"] = (
        summary["MaleDeath"] /
        summary["Male"].replace(0, np.nan) * 100
    )

    summary["MortFemale"] = (
        summary["FemaleDeath"] /
        summary["Female"].replace(0, np.nan) * 100
    )

    summary["Mortality"] = (
        summary["TotalDeaths"] /
        summary["Total"] * 100
    )

    return summary.sort_values("Total", ascending=False)


def plot_distrib(df, disease_mapper):
    
    summary = distrib_summary(df, disease_mapper)
    palette = {
        "Male": "#4C72B0",
        "Female": "#DD8452",
        "Splenectomized": "#55A868",
        "Not splenectomized": "#C44E52"
    }
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.35
    )
    sns.despine()
    fig, axs = plt.subplots(
        2,2,
        figsize=(13,10),
        constrained_layout=True
    )
    # -------------------------
    # PANEL A
    # Cohort composition + splenectomy status
    # -------------------------
    tmp = summary.sort_values("Total", ascending=False).copy()

    axs[0,0].bar(
        tmp["Disease"],
        tmp["Splen"],
        color=palette["Splenectomized"],
        edgecolor="white",
        linewidth=0.7,
        label="Splenectomized"
    )

    axs[0,0].bar(
        tmp["Disease"],
        tmp["NoSplen"],
        bottom=tmp["Splen"],
        color=palette["Not splenectomized"],
        edgecolor="white",
        linewidth=0.7,
        label="Not splenectomized"
    )

    # segment labels + total labels
    for i, row in enumerate(tmp.itertuples()):
        if row.Splen > 0:
            axs[0,0].text(
                i,
                row.Splen / 2,
                f"{int(row.Splen)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold"
            )
        if row.NoSplen > 0:
            axs[0,0].text(
                i,
                row.Splen + row.NoSplen / 2,
                f"{int(row.NoSplen)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold"
            )

        axs[0,0].text(
            i,
            row.Total + tmp["Total"].max() * 0.02,
            f"{int(row.Total)}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

    axs[0,0].set_title("A. Cohort composition and splenectomy status")
    axs[0,0].set_xlabel("")
    axs[0,0].set_ylabel("Number of patients")
    axs[0,0].tick_params(axis="x", rotation=45)

    axs[0,0].set_ylim(
        0,
        tmp["Total"].max() * 1.15
    )

    axs[0,0].legend(
        title="",
        frameon=True,
        facecolor="white",
        edgecolor="lightgray",
        framealpha=0.85
    )

    axs[0,0].grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )

    axs[0,0].grid(axis="x", visible=False)
    axs[0,0].set_axisbelow(True)

    # -------------------------
    # PANEL B
    # Event burden
    # -------------------------

    tmp = df.copy()

    tmp["event_length"] = tmp["events"].apply(len)

    tmp = tmp.dropna(
        subset=["base_pathology_area", "event_length"]
    )

    # Usa il mapping solo se serve
    if disease_mapper is not None:
        tmp["base_pathology_area"] = tmp["base_pathology_area"].map(disease_mapper)

    order = summary["Disease"].tolist()

    sns.boxplot(
        data=tmp,
        x="base_pathology_area",
        y="event_length",
        order=order,
        color=palette["Male"],
        width=0.55,
        showfliers=False,
        ax=axs[0,1]
    )

    sns.stripplot(
        data=tmp,
        x="base_pathology_area",
        y="event_length",
        order=order,
        color="black",
        alpha=0.25,
        size=2,
        jitter=0.25,
        ax=axs[0,1]
    )

    axs[0,1].set_title("B. Event burden")
    axs[0,1].set_xlabel("")
    axs[0,1].set_ylabel("Number of events")
    axs[0,1].tick_params(axis="x", rotation=45)

    axs[0,1].grid(axis="y", linestyle="--", alpha=0.35)
    axs[0,1].grid(axis="x", visible=False)
    axs[0,1].set_axisbelow(True)

    # Counts above each boxplot
    counts = (
        tmp["base_pathology_area"]
        .value_counts()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )

    ylim = axs[0,1].get_ylim()[1]
    #ymin, _ = axs[0,1].get_ylim()
    axs[0,1].set_ylim(0, ylim)

    for i, disease in enumerate(order):
        axs[0,1].text(
            i,
            ylim * 0.98,
            f"n={counts[disease]}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold"
        )

    sns.despine(ax=axs[0,1])

    # -------------------------
    # PANEL C
    # Mortality by sex
    # -------------------------

    sex_long = pd.melt(
        summary,
        id_vars=["Disease", "Male", "Female", "MaleDeath", "FemaleDeath"],
        value_vars=["MortMale", "MortFemale"],
        var_name="Group",
        value_name="MortalityRate"
    )

    sex_long["Group"] = sex_long["Group"].replace({
        "MortMale": "Male",
        "MortFemale": "Female"
    })

    sns.barplot(
        data=sex_long,
        x="Disease",
        y="MortalityRate",
        hue="Group",
        ax=axs[1,0],
        palette=palette
    )

    axs[1,0].set_title("C. Mortality by sex")
    axs[1,0].set_xlabel("")
    axs[1,0].set_ylabel("Mortality (%)")
    axs[1,0].tick_params(axis="x", rotation=45)
    axs[1,0].legend(title="", frameon=False)

    # ---------- labels ----------

    labels = []
    for _, row in summary.iterrows():
        labels.append(f"{int(row['MaleDeath'])}/{int(row['Male'])}")
    for _, row in summary.iterrows():
        labels.append(f"{int(row['FemaleDeath'])}/{int(row['Female'])}")
    k = 0

    for container in axs[1,0].containers:

        texts = labels[k:k+len(container)]

        axs[1,0].bar_label(
            container,
            labels=texts,
            padding=3,
            fontsize=8
        )
        k += len(container)

    axs[1,0].set_ylim(
        0,
        max(sex_long["MortalityRate"])*1.20
    )

    # -------------------------
    # PANEL D
    # Mortality by splenectomy
    # -------------------------

    long = pd.melt(
        summary,
        id_vars=[
            "Disease",
            "Splen",
            "NoSplen",
            "SplenDeath",
            "NoSplenDeath"
        ],
        value_vars=[
            "MortSplen",
            "MortNoSplen"
        ],
        var_name="Group",
        value_name="MortalityRate"
    )

    long["Group"] = long["Group"].replace({
        "MortSplen": "Splenectomized",
        "MortNoSplen": "Not splenectomized"
    })

    sns.barplot(
        data=long,
        x="Disease",
        y="MortalityRate",
        hue="Group",
        ax=axs[1,1],
        palette=palette
    )

    axs[1,1].set_title("D. Mortality by splenectomy status")
    axs[1,1].set_xlabel("")
    axs[1,1].set_ylabel("Mortality (%)")
    axs[1,1].tick_params(axis="x", rotation=45)
    axs[1,1].legend(title="", frameon=False)

    # ---------- labels ----------

    labels = []

    for _, row in summary.iterrows():
        if not(row["Splen"] == 0 or row["Splen"] == np.nan):
            labels.append(f"{int(row['SplenDeath'])}/{int(row['Splen'])}")

    for _, row in summary.iterrows():
        if not(row["NoSplen"] == 0 or row["NoSplen"] == np.nan):
            labels.append(f"{int(row['NoSplenDeath'])}/{int(row['NoSplen'])}")

    k = 0

    for container in axs[1,1].containers:

        texts = labels[k:k+len(container)]

        axs[1,1].bar_label(
            container,
            labels=texts,
            padding=3,
            fontsize=8
        )

        k += len(container)

    axs[1,1].set_ylim(
        0,
        max(long["MortalityRate"])*1.20
    )
    for ax in axs.flat:

        sns.despine(ax=ax)

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35
        )

        ax.grid(axis="x", visible=False)

        ax.tick_params(
            axis="both",
            labelsize=10
        )

        ax.set_axisbelow(True)
    return fig
