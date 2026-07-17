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
import shap

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier

from features_4class import (
    load_data, build_features, rename, CLASS_ORDER, CLASS_NAMES_DE,
    PALETTE_CLASS,
)

HERE = Path(__file__).parent
DATA_PATH_N593 = HERE / "Bereinigte_Daten" / "master_features_clean_n593.csv"
OUT_DIR = HERE / "XGBoost_4Klassen"
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
    imputer = SimpleImputer(strategy="median")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=N_CLASSES,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        importance_type="gain",
    )
    return Pipeline(steps=[("imputer", imputer), ("model", model)])


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
    print(f"\n=== XGBoost 4-Klassen – Variante: {title} ===")

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

    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    sw_full = compute_sample_weight("balanced", y.values)
    final_model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=N_CLASSES, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, importance_type="gain",
    )
    final_model.fit(X_imp, y.values, sample_weight=sw_full)
    importances = final_model.feature_importances_
    order = np.argsort(importances)[::-1]

    v = {
        "tag": tag, "title": title, "include_acc": include_acc,
        "X_imp": X_imp, "y": y, "groups": groups, "feature_cols": feature_cols,
        "metrics": metrics, "per_class_f1": per_class_f1, "per_subj_acc": per_subj_acc,
        "y_pred": y_pred, "y_proba": y_proba, "cm": cm,
        "imputer": imp, "final_model": final_model,
        "importances": importances, "order": order,
    }
    joblib.dump(v, CACHE_DIR / f"{tag}.joblib")
    print(f"  Cache gespeichert: {tag}.joblib")
    return v


def load_cached(tag):
    path = CACHE_DIR / f"{tag}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Kein Cache fuer '{tag}' — zuerst 'python3 xgboost_4class.py --stage loso --variant {tag}' ausfuehren.")
    return joblib.load(path)



def make_comparison_figure(variants, out_path):
    fig = plt.figure(figsize=(19, 12))
    fig.suptitle("XGBoost 4-Klassen (Ruhe / Stress / Aerob / Anaerob) – LOSOCV: mit vs. ohne ACC",
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
    idx = v["order"][:8]
    names_f = rename([v["feature_cols"][j] for j in idx])
    vals = [v["importances"][j] for j in idx]
    ax.barh(names_f[::-1], vals[::-1], color="#8172B3", alpha=0.85)
    ax.set_title("Top-Features (ohne ACC)\nXGBoost Gain-Importance")
    ax.set_xlabel("Gain (relative Wichtigkeit)")

    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nVisualisierung gespeichert: {out_path}")


def save_model(v):
    path = OUT_DIR / f"xgboost_4class_{v['tag']}_n593.joblib"
    payload = {
        "imputer": v["imputer"],
        "model": v["final_model"],
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
    with open(OUT_DIR / f"xgboost_4class_{v['tag']}_n593_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Modell + Metadaten gespeichert: {path.name}")


def stage_figure():
    variants = [load_cached("mit_acc"), load_cached("ohne_acc")]
    print(f"\n{'Variante':10s} " + " ".join(f"{SCORING_LABELS[m]:>20}" for m in SCORING))
    for v in variants:
        print(f"{v['title']:10s} " + " ".join(f"{v['metrics'][m]:20.3f}" for m in SCORING))
    make_comparison_figure(variants, OUT_DIR / "xgboost_4class_results_n593.png")
    print("\nSpeichere Modelle...")
    for v in variants:
        save_model(v)



def explain_variant(v):
    explainer = shap.TreeExplainer(v["final_model"])
    sv = explainer(v["X_imp"])
    sv.feature_names = rename(v["feature_cols"])
    v["sv"] = sv
    return v


def plot_shap_overview(v):
    sv = v["sv"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for k, ax in enumerate(axes.ravel()):
        plt.sca(ax)
        shap.plots.beeswarm(sv[:, :, k], max_display=10, show=False, plot_size=None)
        ax.set_title(f"Klasse: {CLASS_NAMES_DE[k]}", fontsize=12, fontweight="bold")
    fig.suptitle(f"XGBoost 4-Klassen – SHAP Beeswarm je Klasse ({v['title']})\n"
                 "rot = hoher Feature-Wert, blau = niedrig; rechts von 0 = schiebt Richtung dieser Klasse",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / f"shap_overview_{v['tag']}_n593.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {out.name}")


def plot_shap_importance_bar(v):
    sv = v["sv"]
    mean_abs = np.abs(sv.values).mean(axis=0)
    overall = mean_abs.mean(axis=1)
    order = np.argsort(overall)[::-1][:10]
    feat_names = [sv.feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos = np.arange(len(order))
    h = 0.8 / N_CLASSES
    for k in range(N_CLASSES):
        vals = mean_abs[order, k]
        ax.barh(y_pos + (k - (N_CLASSES - 1) / 2) * h, vals, h,
                label=CLASS_NAMES_DE[k], color=PALETTE_CLASS[CLASS_NAMES_DE[k]], alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels(feat_names)
    ax.invert_yaxis()
    ax.set_xlabel("Mittlerer |SHAP|-Wert")
    ax.set_title(f"XGBoost 4-Klassen – SHAP-Feature-Importance je Klasse ({v['title']})")
    ax.legend()
    plt.tight_layout()
    out = OUT_DIR / f"shap_importance_bar_{v['tag']}_n593.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {out.name}")


def plot_shap_dependence(v):
    sv = v["sv"]
    fig, axes = plt.subplots(N_CLASSES, 4, figsize=(20, 4.6 * N_CLASSES))
    for k in range(N_CLASSES):
        sv_k = sv[:, :, k]
        mean_abs_k = np.abs(sv_k.values).mean(axis=0)
        top4 = np.argsort(mean_abs_k)[::-1][:4]
        for col, feat_idx in enumerate(top4):
            ax = axes[k, col]
            feat_name = sv_k.feature_names[feat_idx]
            plt.sca(ax)
            shap.plots.scatter(sv_k[:, feat_name], color=sv_k, ax=ax, show=False)
            ax.set_title(feat_name, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{CLASS_NAMES_DE[k]}\nSHAP-Wert", fontsize=10, fontweight="bold")
    fig.suptitle(f"XGBoost 4-Klassen – SHAP Dependence je Klasse ({v['title']})\n"
                 "x = Feature-Wert, y = SHAP-Beitrag zur jeweiligen Klasse, "
                 "Farbe = staerkstes interagierendes Feature",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"shap_dependence_{v['tag']}_n593.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gespeichert: {out.name}")


def plot_shap_waterfall(v):
    sv = v["sv"]
    proba = v["final_model"].predict_proba(v["X_imp"])
    y_true = v["y"].values
    y_hat = proba.argmax(axis=1)

    for k in range(N_CLASSES):
        correct_k = np.where((y_true == k) & (y_hat == k))[0]
        pool = correct_k if len(correct_k) > 0 else np.where(y_true == k)[0]
        i = pool[np.argmax(proba[pool, k])]

        fig = plt.figure(figsize=(10, 7))
        shap.plots.waterfall(sv[i, :, k], max_display=12, show=False)
        p = proba[i, k]
        true_lbl = CLASS_NAMES_DE[y_true[i]]
        plt.title(f"Einzelfall ({v['title']}) – Klasse {CLASS_NAMES_DE[k]}\n"
                  f"Wahres Label: {true_lbl} | P({CLASS_NAMES_DE[k]}) = {p:.2f}",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        out = OUT_DIR / f"shap_waterfall_{v['tag']}_{CLASS_ORDER[k]}_n593.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"gespeichert: {out.name}")


def stage_shap(tag, parts=None):
    v = load_cached(tag)
    explain_variant(v)
    parts = parts or ["overview", "bar", "dependence", "waterfall"]
    if "overview" in parts:
        plot_shap_overview(v)
    if "bar" in parts:
        plot_shap_importance_bar(v)
    if "dependence" in parts:
        plot_shap_dependence(v)
    if "waterfall" in parts:
        plot_shap_waterfall(v)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["loso", "figure", "shap", "all"], default="all")
    parser.add_argument("--variant", choices=["mit_acc", "ohne_acc"], default=None)
    parser.add_argument("--parts", nargs="*", default=None,
                         help="Teilmenge von: overview bar dependence waterfall")
    args = parser.parse_args()

    if args.stage == "loso":
        stage_loso(args.variant)
    elif args.stage == "figure":
        stage_figure()
    elif args.stage == "shap":
        stage_shap(args.variant, args.parts)
    elif args.stage == "all":
        stage_loso("mit_acc")
        stage_loso("ohne_acc")
        stage_figure()
        stage_shap("mit_acc")
        stage_shap("ohne_acc")
        print(f"\nFertig. Alle Outputs in: {OUT_DIR}")


if __name__ == "__main__":
    main()
