import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier

HERE = Path(__file__).parent
DATA_PATH = HERE / "master_features_clean.csv"
OUT_DIR = HERE / "shap_output_tabpfn"
OUT_DIR.mkdir(exist_ok=True)

ID_COLS     = ["subject", "phase_name", "protocol"]
LEAK_COLS   = ["label", "category"]
DESIGN_COLS = ["duration_sec"]
EMPTY_COLS  = ["LF_peak"]
ACC_COLS    = ["acc_ratio_down", "acc_mean", "x_std", "y_std", "z_std"]
GROUP       = "subject"

N_BACKGROUND = 50
N_EXPLAIN_PER_CLASS = 8
NSAMPLES = 64
RNG = 42

PRETTY = {
    "hr_mean": "Herzrate (Mittel)", "hr_std": "Herzrate (SD)",
    "mean_tonic_eda": "EDA tonisch (Mittel)", "std_tonic_eda": "EDA tonisch (SD)",
    "std_phasic_eda": "EDA phasisch (SD)", "tonic_ratio_down": "EDA-Trend",
    "peaks_density": "EDA-Peaks/min", "mean_recoverytime": "EDA-Erholzeit",
    "max_ibi": "IBI max", "min_ibi": "IBI min", "ibi_mean": "IBI (Mittel)",
    "rmssd": "RMSSD (HRV)", "LF_n": "LF (norm.)",
    "x_std": "ACC x (SD)", "y_std": "ACC y (SD)", "z_std": "ACC z (SD)",
    "acc_mean": "ACC (Mittel)", "acc_ratio_down": "ACC-Trend",
    "temp_mean": "Temp (Mittel)", "temp_std": "Temp (SD)", "temp_slope": "Temp-Trend",
}
CLASS_COLORS = {"aerobic": "#4C72B0", "rest": "#55A868",
                "sprint": "#C44E52", "stress": "#8172B3"}


def load_data():
    return pd.read_csv(DATA_PATH, sep=";")


def rename(cols):
    return [PRETTY.get(c, c) for c in cols]


def build_features(df, include_acc):
    """Wie im Kollegen-Skript: LF_peak raus (100% NaN), LF_n bleibt drin."""
    drop = set(ID_COLS + LEAK_COLS + DESIGN_COLS + EMPTY_COLS)
    if not include_acc:
        drop |= set(ACC_COLS)
    feature_cols = [c for c in df.columns if c not in drop]
    le = LabelEncoder()
    y = le.fit_transform(df["label"])
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    groups = df[GROUP].to_numpy()
    return X, y, groups, feature_cols, le


def fit_model(X, y):
    imp = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=X.columns, index=X.index)
    model = TabPFNClassifier(balance_probabilities=True, random_state=RNG, device="auto")
    model.fit(X_imp, y)
    return X_imp, model


def explain_variant(df, include_acc, le):
    tag = "mit_acc" if include_acc else "ohne_acc"
    title = "mit ACC" if include_acc else "ohne ACC"
    class_names = list(le.classes_)
    print(f"\n=== Variante: {title} ===")

    X, y, groups, feature_cols, _ = build_features(df, include_acc)
    X_imp, model = fit_model(X, y)

    def predict_fn(X_arr):
        return model.predict_proba(pd.DataFrame(X_arr, columns=feature_cols))

    rng = np.random.RandomState(RNG)
    background = X_imp.sample(n=N_BACKGROUND, random_state=RNG).to_numpy()

    explain_idx = []
    for cls in range(len(class_names)):
        idx_cls = np.where(y == cls)[0]
        chosen = rng.choice(idx_cls, size=min(N_EXPLAIN_PER_CLASS, len(idx_cls)), replace=False)
        explain_idx.extend(chosen)
    explain_idx = np.array(explain_idx)
    X_explain = X_imp.iloc[explain_idx]
    y_explain = y[explain_idx]

    print(f"Berechne SHAP fuer {len(X_explain)} Beispiele "
          f"(Hintergrund={N_BACKGROUND}, nsamples={NSAMPLES}) - 35-70 Min ...")
    explainer = shap.KernelExplainer(predict_fn, background)
    raw = explainer.shap_values(X_explain.to_numpy(), nsamples=NSAMPLES)

    base = np.array(explainer.expected_value)
    sv = shap.Explanation(
        values=raw,
        base_values=np.tile(base, (len(X_explain), 1)),
        data=X_explain.to_numpy(),
        feature_names=rename(feature_cols),
    )
    proba = model.predict_proba(X_explain)

    return dict(tag=tag, title=title, sv=sv, X=X_explain, y=y_explain, proba=proba,
                feature_cols=feature_cols, class_names=class_names, model=model)



def plot_overview(variants, class_names):
    for c, cls_name in enumerate(class_names):
        fig = plt.figure(figsize=(16, 12))
        for col, v in enumerate(variants):
            sv_c = v["sv"][..., c]
            plt.subplot(2, 2, col + 1)
            shap.plots.beeswarm(sv_c, max_display=12, show=False, plot_size=None)
            plt.title(f"Beeswarm – {v['title']}", fontsize=13, fontweight="bold")
            plt.subplot(2, 2, col + 3)
            shap.plots.bar(sv_c, max_display=12, show=False)
            plt.title(f"Mittlerer |SHAP| – {v['title']}", fontsize=13, fontweight="bold")
        fig.suptitle(f"TabPFN – SHAP-Erklaerung fuer Klasse '{cls_name}'\n"
                     "rot = hoher Feature-Wert, blau = niedrig; rechts von 0 = schiebt Richtung dieser Klasse",
                     fontsize=15, fontweight="bold", y=1.02)
        plt.tight_layout()
        out = OUT_DIR / f"shap_overview_{cls_name}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"gespeichert: {out.name}")


def plot_dependence(variants, class_names):
    for v in variants:
        for c, cls_name in enumerate(class_names):
            sv_c = v["sv"][..., c]
            order = np.argsort(np.abs(sv_c.values).mean(0))[::-1]
            top4 = [sv_c.feature_names[i] for i in order[:4]]
            fig, axes = plt.subplots(2, 2, figsize=(15, 11))
            for ax, feat in zip(axes.ravel(), top4):
                shap.plots.scatter(sv_c[:, feat], color=sv_c, ax=ax, show=False)
                ax.set_title(feat, fontsize=12, fontweight="bold")
            fig.suptitle(f"SHAP Dependence – Klasse '{cls_name}', {v['title']}\n"
                         "x = Feature-Wert, y = SHAP (Wirkung auf diese Klasse); "
                         "Farbe = staerkstes interagierendes Feature",
                         fontsize=14, fontweight="bold")
            plt.tight_layout()
            out = OUT_DIR / f"shap_dependence_{cls_name}_{v['tag']}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"gespeichert: {out.name}")


def plot_waterfall(variants, class_names):
    for v in variants:
        for c, cls_name in enumerate(class_names):
            idx_cls = np.where(v["y"] == c)[0]
            if len(idx_cls) == 0:
                continue
            i_best = idx_cls[np.argmax(v["proba"][idx_cls, c])]

            fig = plt.figure(figsize=(10, 7))
            shap.plots.waterfall(v["sv"][i_best, :, c], max_display=12, show=False)
            p = v["proba"][i_best, c]
            plt.title(f"Klarster Fall '{cls_name}' ({v['title']})\n"
                      f"P({cls_name}) = {p:.2f}",
                      fontsize=12, fontweight="bold")
            plt.tight_layout()
            out = OUT_DIR / f"shap_waterfall_{cls_name}_{v['tag']}.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"gespeichert: {out.name}")


def main():
    df = load_data()
    le = LabelEncoder().fit(df["label"])
    variants = [explain_variant(df, include_acc=False, le=le),
                explain_variant(df, include_acc=True, le=le)]
    class_names = list(le.classes_)

    plot_overview(variants, class_names)
    plot_dependence(variants, class_names)
    plot_waterfall(variants, class_names)
    print(f"\nAlle SHAP-Grafiken in: {OUT_DIR}")


if __name__ == "__main__":
    main()
