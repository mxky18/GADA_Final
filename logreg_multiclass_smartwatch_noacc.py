from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score)

from logreg_multiclass import load_data, build_pipeline, run_variant
from logreg_multiclass_smartwatch import run_variant_smartwatch, EDA_COLS, TEMP_COLS
from xgboost_multiclass import (
    ID_COLS, LEAK_COLS, DESIGN_COLS, EMPTY_COLS, ACC_COLS, GROUP,
    SCORING, SCORING_LABELS,
)

DATA_PATH = Path(__file__).parent / "master_features_clean.csv"

VARIANT_COLORS = {
    "alle Features":  "#4C72B0",
    "ohne ACC":        "#C44E52",
    "nur HR+ACC+HRV":  "#55A868",
    "nur HR+HRV":      "#8172B3",
}


def build_features_smartwatch_noacc(df):
    """HR + HRV (ohne LF_peak). Kein ACC, kein EDA, kein TEMP."""
    drop = set(ID_COLS + LEAK_COLS + DESIGN_COLS + EMPTY_COLS + EDA_COLS + TEMP_COLS + ACC_COLS)
    feature_cols = [c for c in df.columns if c not in drop]
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    groups = df[GROUP].to_numpy()
    return X, y, groups, feature_cols, le


def run_variant_smartwatch_noacc(df):
    X, y, groups, feature_cols, le = build_features_smartwatch_noacc(df)
    pipe = build_pipeline()
    cv = LeaveOneGroupOut()

    y_pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv)
    y_prob = cross_val_predict(pipe, X, y, groups=groups, cv=cv, method="predict_proba")

    means = {
        "accuracy":        accuracy_score(y, y_pred),
        "roc_auc":         roc_auc_score(y, y_prob, multi_class="ovr", average="macro"),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro":        f1_score(y, y_pred, average="macro"),
    }
    per_subj_acc = [accuracy_score(y[groups == g], y_pred[groups == g]) for g in np.unique(groups)]

    pipe.fit(X, y)
    coefs = pipe.named_steps["model"].coef_
    order = np.argsort(np.abs(coefs).max(axis=0))[::-1]

    return {
        "name": "nur HR+HRV", "X": X, "y": y, "groups": groups,
        "feature_cols": feature_cols, "le": le, "means": means,
        "per_subj_acc": np.array(per_subj_acc),
        "y_pred": y_pred, "y_prob": y_prob, "coefs": coefs, "order": order,
    }


def print_summary(variants):
    le = variants[0]["le"]
    y0 = variants[0]["y"]
    counts = {le.classes_[k]: int((y0 == k).sum()) for k in range(4)}
    print(f"\nDatensatz: {len(y0)} Zeilen, 4 Klassen (bereinigt)")
    print(f"Klassenverteilung: {counts}\n")
    print(f"{'Variante':16s} {'#Feat':>6} " +
          " ".join(f"{SCORING_LABELS[m]:>22}" for m in SCORING) + f"{'acc/Pers.':>14}")
    for v in variants:
        row = f"{v['name']:16s} {len(v['feature_cols']):6d} "
        row += " ".join(f"{v['means'][m]:22.3f}" for m in SCORING)
        row += f"{v['per_subj_acc'].mean():9.3f}±{v['per_subj_acc'].std():.2f}"
        print(row)
    print("\n(Metriken = gepoolt ueber Out-of-fold-Vorhersagen, Leave-One-Subject-Out CV)")


def make_figure(variants, out_path):
    """variants = [alle Features, ohne ACC, nur HR+ACC+HRV, nur HR+HRV]."""
    le = variants[0]["le"]
    class_names = list(le.classes_)
    v_all, v_noacc, v_sw, v_swnoacc = variants
    recs = [recall_score(v["y"], v["y_pred"], average=None, labels=np.arange(4)) for v in variants]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle("Multinomiale LogReg – 4-Klassen (bereinigt): Smartwatch mit vs. ohne ACC\n"
                 "(HR+ACC+HRV vs. nur HR+HRV) im Vergleich zu alle Features / ohne ACC (voller Datensatz)",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.4)

    ax = fig.add_subplot(gs[0, 0:2])
    x = np.arange(len(SCORING)); w = 0.19
    for i, v in enumerate(variants):
        means = [v["means"][m] for m in SCORING]
        bars = ax.bar(x + (i - 1.5) * w, means, w,
                      label=v["name"], color=VARIANT_COLORS[v["name"]], alpha=0.88)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([SCORING_LABELS[m] for m in SCORING], rotation=20, ha="right")
    ax.set_ylim(0, 1.22); ax.set_ylabel("Score")
    ax.set_title("LOSO-Metriken (gepoolt)"); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    i_aer, i_sprint, i_str = class_names.index("aerobic"), class_names.index("sprint"), class_names.index("stress")
    lines = ["Differenz (nur HR+HRV - Vergleich):", ""]
    for label, v_ref, rec_ref in [("alle Features", v_all, recs[0]),
                                   ("ohne ACC", v_noacc, recs[1]),
                                   ("nur HR+ACC+HRV", v_sw, recs[2])]:
        d_f1 = v_swnoacc["means"]["f1_macro"] - v_ref["means"]["f1_macro"]
        lines.append(f"vs. {label}: F1 {d_f1:+.3f}")
    lines.append("")
    lines.append("aerobic-Recall: " + " -> ".join(f"{r[i_aer]:.2f}" for r in recs))
    lines.append("sprint-Recall:  " + " -> ".join(f"{r[i_sprint]:.2f}" for r in recs))
    lines.append("stress-Recall:  " + " -> ".join(f"{r[i_str]:.2f}" for r in recs))
    lines.append("(Reihenfolge: alle / ohneACC / HR+ACC+HRV / HR+HRV)")
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=8.2)

    ax = fig.add_subplot(gs[0, 3])
    x = np.arange(4); w = 0.19
    for i, (v, rec) in enumerate(zip(variants, recs)):
        ax.bar(x + (i - 1.5) * w, rec, w, label=v["name"], color=VARIANT_COLORS[v["name"]], alpha=0.88)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall")
    ax.set_title("Recall pro Klasse"); ax.legend(fontsize=7)

    for col, v in enumerate(variants):
        ax = fig.add_subplot(gs[1, col])
        cm = confusion_matrix(v["y"], v["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"CM – {v['name']}", fontsize=10)
        ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("Tatsaechlich")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualisierung gespeichert: {out_path}")


def main():
    df = load_data(DATA_PATH)
    print(f"Lade {len(df)} Zeilen aus master_features_clean.csv (bereinigt) ...")

    v_all = run_variant(df, include_acc=True); v_all["name"] = "alle Features"
    v_noacc = run_variant(df, include_acc=False); v_noacc["name"] = "ohne ACC"
    v_sw = run_variant_smartwatch(df)
    v_swnoacc = run_variant_smartwatch_noacc(df)

    variants = [v_all, v_noacc, v_sw, v_swnoacc]
    print_summary(variants)

    for v in variants:
        print(f"\n=== Classification Report – {v['name']} ===")
        print(classification_report(v["y"], v["y_pred"],
                                     target_names=list(v["le"].classes_), digits=3))

    make_figure(variants, Path(__file__).parent / "logreg_multiclass_smartwatch_noacc_results.png")


if __name__ == "__main__":
    main()
