import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from interpret.glassbox import ExplainableBoostingClassifier

from features_4class import (
    load_data, build_features, rename, CLASS_ORDER, CLASS_NAMES_DE,
    PALETTE_CLASS,
)

HERE = Path(__file__).parent
DATA_PATH_N593 = HERE / "Bereinigte_Daten" / "master_features_clean_n593.csv"
OUT_DIR = HERE / "EBM_4Klassen"
CACHE_DIR = OUT_DIR / "_cache"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

N_CLASSES = len(CLASS_ORDER)
SCORING = ["accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc_ovr_macro"]
SCORING_LABELS = {
    "accuracy": "Accuracy", "f1_macro": "F1-macro",
    "precision_macro": "Precision-macro", "recall_macro": "Recall-macro",
    "roc_auc_ovr_macro": "ROC-AUC (OvR, macro)",
}
VARIANT_COLOR = {"mit ACC": "#4C72B0", "ohne ACC": "#C44E52"}
TAG_TITLE = {"mit_acc": "mit ACC", "ohne_acc": "ohne ACC"}



def build_pipeline():
    imputer = SimpleImputer(strategy="median").set_output(transform="pandas")
    scaler = StandardScaler().set_output(transform="pandas")
    model = ExplainableBoostingClassifier(
        max_rounds=100, interactions=0, outer_bags=4, random_state=42, n_jobs=1,
    )
    return Pipeline(steps=[("imputer", imputer), ("scaler", scaler), ("model", model)])


def run_loso(X, y, groups):
    cv = LeaveOneGroupOut()
    y_arr = y.values
    y_pred = np.empty_like(y_arr)
    y_proba = np.empty((len(y_arr), N_CLASSES), dtype=float)

    for train_idx, test_idx in cv.split(X, y_arr, groups):
        pipe = build_pipeline()
        sw = compute_sample_weight("balanced", y_arr[train_idx])
        pipe.fit(X.iloc[train_idx], y_arr[train_idx], model__sample_weight=sw)
        y_pred[test_idx] = pipe.predict(X.iloc[test_idx])
        y_proba[test_idx] = pipe.predict_proba(X.iloc[test_idx])

    return y_pred, y_proba


def stage_loso(tag):
    include_acc = (tag == "mit_acc")
    title = TAG_TITLE[tag]
    print(f"\n=== EBM 4-Klassen – Variante: {title} ===")

    df = load_data(path=DATA_PATH_N593)
    X, y, groups, feature_cols = build_features(df, include_acc=include_acc)
    y_pred, y_proba = run_loso(X, y, groups)

    metrics = {
        "accuracy":          accuracy_score(y, y_pred),
        "f1_macro":          f1_score(y, y_pred, average="macro"),
        "precision_macro":   precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro":      recall_score(y, y_pred, average="macro"),
        "roc_auc_ovr_macro": roc_auc_score(y, y_proba, multi_class="ovr", average="macro"),
    }
    per_class_f1 = f1_score(y, y_pred, average=None, labels=range(N_CLASSES), zero_division=0)
    per_subj_acc = np.array([
        accuracy_score(y[groups == g], y_pred[groups == g]) for g in np.unique(groups)
    ])
    cm = confusion_matrix(y, y_pred, labels=range(N_CLASSES))

    print(f"  {len(feature_cols)} Features | " +
          " | ".join(f"{SCORING_LABELS[m]}={metrics[m]:.3f}" for m in SCORING))
    print("  Classification Report (LOSOCV, out-of-fold):")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES_DE,
                                 labels=range(N_CLASSES), digits=3, zero_division=0))

    final_pipe = build_pipeline()
    sw_full = compute_sample_weight("balanced", y.values)
    final_pipe.fit(X, y.values, model__sample_weight=sw_full)

    v = {
        "tag": tag, "title": title, "include_acc": include_acc,
        "X": X, "y": y, "groups": groups, "feature_cols": feature_cols,
        "metrics": metrics, "per_class_f1": per_class_f1, "per_subj_acc": per_subj_acc,
        "y_pred": y_pred, "y_proba": y_proba, "cm": cm,
        "final_pipe": final_pipe,
    }
    joblib.dump(v, CACHE_DIR / f"{tag}.joblib")
    print(f"  Cache gespeichert: {tag}.joblib")
    return v


def load_cached(tag):
    path = CACHE_DIR / f"{tag}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache fuer '{tag}' — zuerst 'python3 ebm_4class.py --stage loso --variant {tag}' ausfuehren.")
    return joblib.load(path)



def make_comparison_figure(variants, out_path):
    fig = plt.figure(figsize=(19, 12))
    fig.suptitle("EBM 4-Klassen (Ruhe / Stress / Aerob / Anaerob) – LOSOCV: mit vs. ohne ACC",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(SCORING)); w = 0.38
    for i, v in enumerate(variants):
        means = [v["metrics"][m] for m in SCORING]
        bars = ax.bar(x + (i - 0.5) * w, means, w, label=v["title"],
                      color=VARIANT_COLOR[v["title"]], alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([SCORING_LABELS[m] for m in SCORING], rotation=30, ha="right")
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score")
    ax.set_title("LOSOCV-Metriken (gepoolt)"); ax.legend()

    ax = fig.add_subplot(gs[0, 1])
    xw = np.arange(N_CLASSES)
    for i, v in enumerate(variants):
        bars = ax.bar(xw + (i - 0.5) * w, v["per_class_f1"], w, label=v["title"],
                      color=VARIANT_COLOR[v["title"]], alpha=0.85)
        for bar, val in zip(bars, v["per_class_f1"]):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xw); ax.set_xticklabels(CLASS_NAMES_DE)
    ax.set_ylim(0, 1.15); ax.set_ylabel("F1"); ax.set_title("F1 je Klasse"); ax.legend()

    ax = fig.add_subplot(gs[0, 2])
    data = [v["per_subj_acc"] for v in variants]
    names = [v["title"] for v in variants]
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(VARIANT_COLOR[name]); patch.set_alpha(0.7)
    ax.set_ylabel("Accuracy pro Subjekt"); ax.set_title("Streuung ueber Subjekte (LOSOCV)")
    ax.set_ylim(0, 1.05)

    for i, v in enumerate(variants):
        ax = fig.add_subplot(gs[1, i])
        sns.heatmap(v["cm"], annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                    xticklabels=CLASS_NAMES_DE, yticklabels=CLASS_NAMES_DE)
        ax.set_title(f"Konfusionsmatrix – {v['title']}")
        ax.set_xlabel("Vorhergesagt"); ax.set_ylabel("Tatsaechlich")

    ax = fig.add_subplot(gs[1, 2])
    v = variants[1]
    ebm = v["final_pipe"].named_steps["model"]
    explanation = ebm.explain_global()
    d = explanation.data()
    names_g, scores_g = d["names"], np.array(d["scores"])
    order = np.argsort(scores_g)[::-1][:8]
    ax.barh(rename([names_g[i] for i in order])[::-1], scores_g[order][::-1],
            color="#8172B3", alpha=0.85)
    ax.set_title("Top-Features (ohne ACC)\nEBM Global Importance")
    ax.set_xlabel("Mittlere absolute EBM-Contribution")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualisierung gespeichert: {out_path}")


def save_model(v):
    path = OUT_DIR / f"ebm_4class_{v['tag']}_n593.joblib"
    payload = {
        "pipe": v["final_pipe"],
        "feature_cols": v["feature_cols"],
        "include_acc": v["include_acc"],
        "class_order": CLASS_ORDER,
        "class_names_de": CLASS_NAMES_DE,
        "metrics_losocv": v["metrics"],
        "per_class_f1_losocv": list(v["per_class_f1"]),
    }
    joblib.dump(payload, path)

    meta = {
        "variant": v["title"],
        "n_features": len(v["feature_cols"]),
        "feature_cols": v["feature_cols"],
        "class_order": CLASS_ORDER,
        "class_names_de": CLASS_NAMES_DE,
        "metrics_losocv": {k: float(val) for k, val in v["metrics"].items()},
        "per_class_f1_losocv": dict(zip(CLASS_NAMES_DE, [float(x) for x in v["per_class_f1"]])),
        "n_samples": int(len(v["y"])),
        "n_subjects": int(v["groups"].nunique()),
    }
    with open(OUT_DIR / f"ebm_4class_{v['tag']}_n593_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Modell + Metadaten gespeichert: {path.name}")


def stage_figure():
    variants = [load_cached("mit_acc"), load_cached("ohne_acc")]
    print(f"\n{'Variante':10s} " + " ".join(f"{SCORING_LABELS[m]:>20}" for m in SCORING))
    for v in variants:
        print(f"{v['title']:10s} " + " ".join(f"{v['metrics'][m]:20.3f}" for m in SCORING))
    make_comparison_figure(variants, OUT_DIR / "ebm_4class_results_n593.png")
    print("\nSpeichere Modelle...")
    for v in variants:
        save_model(v)



def plot_global_importance(v):
    ebm = v["final_pipe"].named_steps["model"]
    explanation = ebm.explain_global()
    d = explanation.data()
    names_g, scores_g = d["names"], np.array(d["scores"])
    order = np.argsort(scores_g)[::-1][:12]
    top_names = rename([names_g[i] for i in order])
    top_scores = scores_g[order]

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#C44E52" if s > np.median(top_scores) else "#4C72B0" for s in top_scores]
    ax.barh(top_names[::-1], top_scores[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mittlere absolute EBM-Contribution (ueber alle Klassen aggregiert)")
    ax.set_title(f"EBM 4-Klassen – Global Feature Importance ({v['title']})")
    plt.tight_layout()
    out = OUT_DIR / f"ebm_global_importance_{v['tag']}_n593.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {out.name}")
    return names_g, scores_g, order


def plot_shape_functions(v, n_features=6):
    """Shape Functions der Top-N Features, je eine Linie pro Klasse
    (Multiclass-EBM liefert scores der Form (n_bins, n_klassen) pro Feature)."""
    ebm = v["final_pipe"].named_steps["model"]
    explanation = ebm.explain_global()
    d = explanation.data()
    names_g, scores_g = d["names"], np.array(d["scores"])
    order = np.argsort(scores_g)[::-1][:n_features]

    ncols = 3
    nrows = int(np.ceil(n_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle(f"EBM 4-Klassen – Shape Functions Top-{n_features} Features ({v['title']})\n"
                 "y-Achse = additiver Beitrag zur Klassen-Log-Odds (Feature standardisiert); "
                 "je eine Linie pro Klasse",
                 fontsize=14, fontweight="bold")

    for plot_idx, feat_idx in enumerate(order):
        ax = axes[plot_idx]
        feat_name = names_g[feat_idx]
        feat_data = explanation.data(feat_idx)
        x_vals = feat_data["names"]
        y_vals = np.array(feat_data["scores"])

        if y_vals.ndim == 1:
            y_vals = y_vals[:, None]

        is_categorical = isinstance(x_vals[0], str)
        for k in range(y_vals.shape[1]):
            color = PALETTE_CLASS[CLASS_NAMES_DE[k]]
            yk = y_vals[:, k]
            if is_categorical:
                xpos = np.arange(len(x_vals)) + (k - (y_vals.shape[1] - 1) / 2) * 0.15
                ax.bar(xpos, yk, width=0.15, color=color, alpha=0.85, label=CLASS_NAMES_DE[k])
            else:
                if len(x_vals) == len(yk) + 1:
                    x_mid = [(x_vals[i] + x_vals[i + 1]) / 2 for i in range(len(yk))]
                else:
                    x_mid = list(x_vals[:len(yk)])
                ax.step(x_mid, yk, where="mid", color=color, lw=2, label=CLASS_NAMES_DE[k])

        if is_categorical:
            ax.set_xticks(range(len(x_vals))); ax.set_xticklabels(x_vals, rotation=45, ha="right")
        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"{rename([feat_name])[0]}\n(importance={scores_g[feat_idx]:.3f})", fontsize=11)
        ax.set_ylabel("EBM Score", fontsize=9)
        if plot_idx == 0:
            ax.legend(fontsize=8)

    for j in range(n_features, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out = OUT_DIR / f"ebm_shape_functions_{v['tag']}_n593.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {out.name}")


def stage_global(tag):
    v = load_cached(tag)
    plot_global_importance(v)
    plot_shape_functions(v, n_features=6)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["loso", "figure", "global", "all"], default="all")
    parser.add_argument("--variant", choices=["mit_acc", "ohne_acc"], default=None)
    args = parser.parse_args()

    if args.stage == "loso":
        stage_loso(args.variant)
    elif args.stage == "figure":
        stage_figure()
    elif args.stage == "global":
        stage_global(args.variant)
    elif args.stage == "all":
        stage_loso("mit_acc")
        stage_loso("ohne_acc")
        stage_figure()
        stage_global("mit_acc")
        stage_global("ohne_acc")
        print(f"\nFertig. Alle Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
