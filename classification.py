
import pandas as pd
import numpy as np
from pathlib import Path


CSV_PATH = Path("master_features.csv")

FEATURE_COLS = [
    "hr_mean", "hr_std",
    "mean_tonic_eda", "std_tonic_eda", "std_phasic_eda",
    "tonic_ratio_down", "peaks_density", "mean_recoverytime",
    "max_ibi", "min_ibi", "ibi_mean", "rmssd", "LF_n",
    "x_std", "y_std", "z_std", "acc_mean", "acc_ratio_down",
    "temp_mean", "temp_std", "temp_slope",
]


df = pd.read_csv(CSV_PATH, sep=";")

n_before = len(df)
df = df[df["duration_sec"] >= 10].copy()
n_dropped = n_before - len(df)
if n_dropped > 0:
    print(f"[Filter] {n_dropped} Bloecke mit duration_sec < 10 s entfernt.")

df = df[df["label"] != "rest"].copy()

df["y"] = (df["label"] == "stress").astype(int)

X = df[FEATURE_COLS]
y = df["y"]
subjects = df["subject"]


print("=" * 60)
print("Binäre Klassifikation: Stress (1) vs. Sport (0)")
print("=" * 60)

print(f"\nBlöcke gesamt:   {len(df)}")
print(f"  Stress  (1):   {(y == 1).sum()}")
print(f"  Sport   (0):   {(y == 0).sum()}")
print(f"  Verhältnis:    1 : {(y == 0).sum() / (y == 1).sum():.1f}")

print(f"\nProbanden: {subjects.nunique()} ({sorted(subjects.unique())})")

print("\nFeature-Mittelwerte je Klasse:")
summary = df.groupby("y")[FEATURE_COLS].mean().T
summary.columns = ["Sport (0)", "Stress (1)"]
print(summary.round(3).to_string())

print("\nNaN-Anteil je Feature:")
nan_rates = X.isna().mean().sort_values(ascending=False)
print(nan_rates[nan_rates > 0].round(3).to_string())


summary["Differenz (Sport-Stress)"] = (summary["Sport (0)"] - summary["Stress (1)"]).round(3)
summary["NaN-Anteil"] = nan_rates.reindex(summary.index).round(3)
summary.to_csv("feature_summary.csv", sep=";")
print("\nGespeichert: feature_summary.csv")


try:
    df.to_csv("binary_dataset.csv", sep=";", index=False)
    print("Gespeichert: binary_dataset.csv")
except PermissionError:
    print("[HINWEIS] binary_dataset.csv ist geoeffnet (Excel?) — Speichern uebersprungen.")
