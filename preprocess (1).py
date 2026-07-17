
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as sps

from load_data import load_signal, load_tags
from segment_data import is_v1, SEGMENTERS, FARBEN

DATASET_PATH = Path(r"C:\Users\Paul\OneDrive\Dokumente\Studium\6. Semester\GADA\Daten\wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1\Wearable_Dataset")

SUBJECT  = "f01"
CATEGORY = "STRESS"


def filter_bvp(bvp_raw, sample_rate=64):
    
    sos = sps.cheby2(4, 40, [0.5, 10], btype="bandpass", fs=sample_rate, output="sos")
    return sps.sosfiltfilt(sos, bvp_raw)


def filter_eda(eda_raw, sample_rate=4):
    
    sos = sps.butter(5, 1.99, btype="low", fs=sample_rate, output="sos")
    eda_filtered = sps.sosfiltfilt(sos, eda_raw)

    window_length = 101
    polyorder = 3
    eda_tonic  = sps.savgol_filter(eda_filtered, window_length, polyorder)
    eda_phasic = eda_filtered - eda_tonic

    return eda_filtered, eda_tonic, eda_phasic


def compute_acc_magnitude(acc_data):
    return np.sqrt(np.sum(acc_data ** 2, axis=1))


if __name__ == "__main__":
    version = "v1" if is_v1(SUBJECT) else "v2"
    subject_path = DATASET_PATH / CATEGORY / SUBJECT.upper()

    eda_raw,  eda_sr,  eda_start = load_signal(subject_path / "EDA.csv")
    bvp_raw,  bvp_sr,  _         = load_signal(subject_path / "BVP.csv")
    hr_data,  hr_sr,   _         = load_signal(subject_path / "HR.csv")
    acc_data, acc_sr,  _         = load_signal(subject_path / "ACC.csv")
    temp_data,temp_sr, _         = load_signal(subject_path / "TEMP.csv")
    tags = load_tags(subject_path / "tags.csv", eda_start)

    print(f"Preprocessing {SUBJECT.upper()} / {CATEGORY} (Protokoll {version})")
    print("-" * 70)

    bvp_filtered = filter_bvp(bvp_raw.flatten(), bvp_sr)
    eda_filtered, eda_tonic, eda_phasic = filter_eda(eda_raw.flatten(), eda_sr)
    acc_mag = compute_acc_magnitude(acc_data)

    print(f"BVP  Chebyshev II Bandpass 0.5-10 Hz")
    print(f"     Roh:       Mean={float(bvp_raw.mean()):>8.2f}  Std={float(bvp_raw.std()):>7.2f}")
    print(f"     Gefiltert: Mean={bvp_filtered.mean():>8.2f}  Std={bvp_filtered.std():>7.2f}")
    print()
    print(f"EDA  Butterworth Lowpass 1.99 Hz + Savitzky-Golay (Paper-Methode)")
    print(f"     Roh:    Mean={float(eda_raw.mean()):>6.3f} uS")
    print(f"     Tonic:  Mean={eda_tonic.mean():>6.3f} uS  (Baseline)")
    print(f"     Phasic: Mean={eda_phasic.mean():>6.3f} uS  Max={eda_phasic.max():.3f}  (Spikes)")
    print()
    print(f"ACC  Magnitude aus 3 Achsen:")
    print(f"     Mean={acc_mag.mean():.1f}  Std={acc_mag.std():.1f}")
    print(f"HR   direkt verwendet ({hr_sr} Hz)")
    print(f"TEMP direkt verwendet ({temp_sr} Hz)")

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    fig.suptitle(f"Preprocessing — {SUBJECT.upper()} / {CATEGORY}", fontsize=13)

    t_bvp = np.arange(len(bvp_raw)) / bvp_sr / 60
    axes[0].plot(t_bvp, bvp_raw.flatten(), color="lightgray", linewidth=0.3, label="Roh")
    axes[0].plot(t_bvp, bvp_filtered,      color="steelblue", linewidth=0.4, label="Gefiltert")
    axes[0].set_ylabel("BVP", fontsize=9)
    axes[0].legend(loc="upper right", fontsize=8)

    t_eda = np.arange(len(eda_raw)) / eda_sr / 60
    axes[1].plot(t_eda, eda_raw.flatten(), color="lightgray", linewidth=0.5, label="Roh")
    axes[1].plot(t_eda, eda_filtered,      color="steelblue", linewidth=0.8, label="Gefiltert")
    axes[1].set_ylabel("EDA (uS)", fontsize=9)
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t_eda, eda_tonic, color="darkgreen", linewidth=0.9)
    axes[2].set_ylabel("EDA Tonic\n(SCL)", fontsize=9)

    axes[3].plot(t_eda, eda_phasic, color="darkred", linewidth=0.6)
    axes[3].axhline(0, color="black", linewidth=0.3, linestyle="--", alpha=0.5)
    axes[3].set_ylabel("EDA Phasic\n(SCR)", fontsize=9)

    segmenter = SEGMENTERS[version][CATEGORY]
    blocks = segmenter(tags)
    for ax in axes:
        for start, end, label, name in blocks:
            ax.axvspan(start / 60, end / 60, color=FARBEN[label], alpha=0.2)

    axes[-1].set_xlabel("Zeit (Minuten)")
    plt.tight_layout()
    out = f"preprocess_{SUBJECT.upper()}_{CATEGORY}.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nPlot gespeichert als {out}")
