
import pandas as pd
import numpy as np
from pathlib import Path

DATASET_PATH = Path(r"C:\Users\Paul\OneDrive\Dokumente\Studium\6. Semester\GADA\Daten\wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1\Wearable_Dataset")

SUBJECT = "f01"
CATEGORY = "STRESS"

def load_signal(file_path):
    
    df = pd.read_csv(file_path, header=None)

    start_time = pd.to_datetime(df.iloc[0, 0])

    sample_rate = float(df.iloc[1, 0])

    data = df.iloc[2:].reset_index(drop=True).astype(float).values

    return data, sample_rate, start_time

def load_tags(file_path, session_start):
    df = pd.read_csv(file_path, header=None)
    tags_utc = pd.to_datetime(df[0])
    offsets = [(t - session_start).total_seconds() for t in tags_utc]
    return offsets

if __name__ == "__main__":
    subject_path = DATASET_PATH / CATEGORY / SUBJECT
    print(f"Lade Daten von: {subject_path}")
    print("-" * 60)

    eda_data, eda_sr, eda_start = load_signal(subject_path / "EDA.csv")
    print(f"EDA:  {len(eda_data)} Werte bei {eda_sr} Hz "
          f"= {len(eda_data)/eda_sr/60:.1f} Min Dauer")

    hr_data, hr_sr, hr_start = load_signal(subject_path / "HR.csv")
    print(f"HR:   {len(hr_data)} Werte bei {hr_sr} Hz "
          f"= {len(hr_data)/hr_sr/60:.1f} Min Dauer")

    bvp_data, bvp_sr, _ = load_signal(subject_path / "BVP.csv")
    print(f"BVP:  {len(bvp_data)} Werte bei {bvp_sr} Hz")

    tags = load_tags(subject_path / "tags.csv", eda_start)
    print(f"\nTags ({len(tags)} Stueck):")
    for i, t in enumerate(tags):
        print(f"  Tag {i+1}: bei Sekunde {t:.1f} (= {t/60:.1f} Min)")

    print(f"\nEDA-Wertebereich: {float(eda_data.min()):.3f} bis {float(eda_data.max()):.3f} uS")
    print(f"HR-Wertebereich:  {float(hr_data.min()):.1f} bis {float(hr_data.max()):.1f} bpm")