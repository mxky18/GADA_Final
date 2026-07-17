import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, recall_score

from logreg_multiclass import load_data, run_variant, print_summary, make_comparison_figure

DATA_PATH_OLD = Path(__file__).parent / "master_features.csv"
DATA_PATH_NEW = Path(__file__).parent / "master_features_clean.csv"


def make_before_after_figure(old_variants, new_variants, out_path):
    le = new_variants[0]["le"]
    class_names = list(le.classes_)
    metrics = ["accuracy", "roc_auc", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "ROC-AUC", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]

    pairs = {"mit ACC": (old_variants[0], new_variants[0]),
             "ohne ACC": (old_variants[1], new_variants[1])}

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle("Multinomiale LogReg – 4-Klassen: unbereinigt vs. bereinigt (rest-Kontamination entfernt)",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    colors = {"unbereinigt": "#B0B0B0", "bereinigt": "#4C72B0"}

    for col, (variant_name, (v_old, v_new)) in enumerate(pairs.items()):
        ax = fig.add_subplot(gs[0, col])
        x = np.arange(len(metrics)); w = 0.38
        vals_old = [v_old["means"][m] for m in metrics]
        vals_new = [v_new["means"][m] for m in metrics]
        bars_old = ax.bar(x - w / 2, vals_old, w, label="unbereinigt", color=colors["unbereinigt"], alpha=0.9)
        bars_new = ax.bar(x + w / 2, vals_new, w, label="bereinigt", color=colors["bereinigt"], alpha=0.9)
        for bars in (bars_old, bars_new):
            for bar, val in zip(bars, [b.get_height() for b in bars]):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels(metric_labels, fontsize=9)
        ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
        ax.set_title(f"LOSO-Metriken – {variant_name}"); ax.legend()

    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    lines = ["Differenz (bereinigt - unbereinigt):", ""]
    for variant_name, (v_old, v_new) in pairs.items():
        lines.append(f"{variant_name}:")
        for m, lbl in zip(metrics, ["accuracy", "roc_auc", "prec_macro", "rec_macro", "f1_macro"]):
            d = v_new["means"][m] - v_old["means"][m]
            lines.append(f"  {lbl:12s}: {d:+.3f}")
        rec_old = recall_score(v_old["y"], v_old["y_pred"], average=None, labels=np.arange(4))
        rec_new = recall_score(v_new["y"], v_new["y_pred"], average=None, labels=np.arange(4))
        i_aer = class_names.index("aerobic")
        lines.append(f"  aerobic-Recall: {rec_old[i_aer]:.3f} -> {rec_new[i_aer]:.3f} "
                     f"({rec_new[i_aer]-rec_old[i_aer]:+.3f})")
        lines.append("")
    ax.text(0.0, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=9.5)

    v_old_no, v_new_no = pairs["ohne ACC"]
    for col, (label, v) in enumerate([("unbereinigt", v_old_no), ("bereinigt", v_new_no)]):
        ax = fig.add_subplot(gs[1, col])
        cm = confusion_matrix(v["y"], v["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f"Confusion Matrix (ohne ACC) – {label}")
        ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("Tatsaechlich")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    ax = fig.add_subplot(gs[1, 2])
    rec_old = recall_score(v_old_no["y"], v_old_no["y_pred"], average=None, labels=np.arange(4))
    rec_new = recall_score(v_new_no["y"], v_new_no["y_pred"], average=None, labels=np.arange(4))
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, rec_old, w, label="unbereinigt", color=colors["unbereinigt"], alpha=0.9)
    ax.bar(x + w / 2, rec_new, w, label="bereinigt", color=colors["bereinigt"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Recall")
    ax.set_title("Recall pro Klasse (ohne ACC)\naerobic sollte am staerksten profitieren")
    ax.legend()

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nVergleichs-Visualisierung gespeichert: {out_path}")


def main():
    df_old = load_data(DATA_PATH_OLD)
    df_new = load_data(DATA_PATH_NEW)
    print(f"Unbereinigt: {len(df_old)} Zeilen | Bereinigt: {len(df_new)} Zeilen")
    print(f"Klassenverteilung unbereinigt: {df_old['label'].value_counts().to_dict()}")
    print(f"Klassenverteilung bereinigt:   {df_new['label'].value_counts().to_dict()}\n")

    print("=== Bereinigter Datensatz ===")
    new_variants = [run_variant(df_new, include_acc=True),
                    run_variant(df_new, include_acc=False)]
    print_summary(new_variants)
    for v in new_variants:
        print(f"\n=== Classification Report – {v['name']} (bereinigt) ===")
        print(classification_report(v["y"], v["y_pred"],
                                     target_names=list(v["le"].classes_), digits=3))

    make_comparison_figure(new_variants, Path(__file__).parent / "logreg_multiclass_clean_results.png")

    print("\n=== Unbereinigter Datensatz (fuer Vorher/Nachher-Vergleich) ===")
    old_variants = [run_variant(df_old, include_acc=True),
                    run_variant(df_old, include_acc=False)]
    print_summary(old_variants)
    for v in old_variants:
        print(f"\n=== Classification Report – {v['name']} (unbereinigt) ===")
        print(classification_report(v["y"], v["y_pred"],
                                     target_names=list(v["le"].classes_), digits=3))

    make_before_after_figure(old_variants, new_variants,
                              Path(__file__).parent / "logreg_multiclass_before_after.png")


if __name__ == "__main__":
    main()
