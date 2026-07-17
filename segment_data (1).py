
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from load_data import load_signal, load_tags

DATASET_PATH = Path(r"C:\Users\Paul\OneDrive\Dokumente\Studium\6. Semester\GADA\Daten\wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1\Wearable_Dataset")

SUBJECT  = "s01"
CATEGORY = "ANAEROBIC"


def is_v1(subject):
    return subject.upper().startswith("S")



def segment_stress_v1(tags):
    t = tags
    return [
        (0,      t[0],  "rest",   "pre-Baseline"),
        (t[0],   t[1],  "rest",   "Baseline"),
        (t[1],   t[2],  "skip",   "SL"),
        (t[2],   t[3],  "stress", "Stroop"),
        (t[3],   t[4],  "rest",   "First Rest"),
        (t[4],   t[5],  "stress", "TMCT"),
        (t[5],   t[6],  "rest",   "Second Rest"),
        (t[6],   t[7],  "stress", "Real Opinion"),
        (t[7],   t[8],  "skip",   "SL"),
        (t[8],   t[9],  "stress", "Opposite Opinion"),
        (t[9],   t[10], "skip",   "SL"),
        (t[10],  t[11], "stress", "Subtract Test"),
    ]


def segment_stress_v2(tags):
    t = tags
    return [
        (0,    t[0], "rest",   "Baseline"),
        (t[0], t[1], "skip",   "SL"),
        (t[1], t[2], "stress", "TMCT"),
        (t[2], t[3], "rest",   "First Rest"),
        (t[3], t[4], "stress", "Real Opinion"),
        (t[4], t[5], "skip",   "SL"),
        (t[5], t[6], "stress", "Opposite Opinion"),
        (t[6], t[7], "rest",   "Second Rest"),
        (t[7], t[8], "stress", "Subtract Test"),
    ]



def segment_aerobic_v1(tags):
    t = tags
    return [
        (0,      t[0],  "rest",    "Warm-up"),
        (t[0],   t[1],  "aerobic", "60 rpm"),
        (t[1],   t[2],  "aerobic", "70 rpm"),
        (t[2],   t[3],  "aerobic", "75 rpm"),
        (t[3],   t[4],  "aerobic", "80 rpm"),
        (t[4],   t[5],  "aerobic", "85 rpm"),
        (t[5],   t[6],  "aerobic", "90 rpm"),
        (t[6],   t[7],  "aerobic", "95 rpm"),
        (t[7],   t[8],  "aerobic", "100 rpm"),
        (t[8],   t[9],  "aerobic", "105 rpm"),
        (t[9],   t[10], "aerobic", "110 rpm"),
        (t[10],  t[11], "rest",    "Cool Down"),
    ]


def segment_aerobic_v2(tags):
    t = tags
    return [
        (0,    t[0], "rest",    "Baseline + Warm-up"),
        (t[0], t[1], "aerobic", "70 rpm"),
        (t[1], t[2], "aerobic", "75 rpm"),
        (t[2], t[3], "aerobic", "80 rpm"),
        (t[3], t[4], "aerobic", "85 rpm"),
        (t[4], t[5], "aerobic", "90/95 rpm"),
        (t[5], t[6], "rest",    "Cool Down"),
        (t[6], t[7], "rest",    "Rest"),
    ]



def segment_anaerobic_v1(tags):
    t = tags
    return [
        (0,      t[0],  "rest",   "Warm-up"),
        (t[0],   t[1],  "sprint", "Sprint 1"),
        (t[1],   t[2],  "rest",   "Cool Down 1"),
        (t[2],   t[3],  "sprint", "Sprint 2"),
        (t[3],   t[4],  "rest",   "Cool Down 2"),
        (t[4],   t[5],  "sprint", "Sprint 3"),
        (t[5],   t[6],  "rest",   "Cool Down 3"),
    ]


def segment_anaerobic_v2(tags):
    t = tags
    return [
        (0,    t[0],  "rest",   "Baseline"),
        (t[0], t[1],  "rest",   "Warm-up"),
        (t[1], t[2],  "sprint", "Sprint 1"),
        (t[2], t[3],  "rest",   "Cool Down 1"),
        (t[3], t[4],  "sprint", "Sprint 2"),
        (t[4], t[5],  "rest",   "Cool Down 2"),
        (t[5], t[6],  "sprint", "Sprint 3"),
        (t[6], t[7],  "rest",   "Cool Down 3"),
        (t[7], t[8],  "sprint", "Sprint 4"),
        (t[8], t[9],  "rest",   "Cool Down 4"),
        (t[9], t[10], "rest",   "Final Rest"),
    ]


def cut_signal(data, sample_rate, start_sek, end_sek):
    i_start = int(start_sek * sample_rate)
    i_end   = int(end_sek   * sample_rate)
    return data[i_start:i_end]


SEGMENTERS = {
    "v1": {
        "STRESS":    segment_stress_v1,
        "AEROBIC":   segment_aerobic_v1,
        "ANAEROBIC": segment_anaerobic_v1,
    },
    "v2": {
        "STRESS":    segment_stress_v2,
        "AEROBIC":   segment_aerobic_v2,
        "ANAEROBIC": segment_anaerobic_v2,
    },
}

FARBEN = {
    "rest":    "#a8d8a8",
    "stress":  "#f4a7a7",
    "aerobic": "#f4c889",
    "sprint":  "#e87d5b",
    "skip":    "#d8d8d8",
}


if __name__ == "__main__":
    version = "v1" if is_v1(SUBJECT) else "v2"
    subject_path = DATASET_PATH / CATEGORY / SUBJECT.upper()

    eda_data,  eda_sr,  eda_start = load_signal(subject_path / "EDA.csv")
    hr_data,   hr_sr,   _         = load_signal(subject_path / "HR.csv")
    bvp_data,  bvp_sr,  _         = load_signal(subject_path / "BVP.csv")
    acc_data,  acc_sr,  _         = load_signal(subject_path / "ACC.csv")
    temp_data, temp_sr, _         = load_signal(subject_path / "TEMP.csv")
    tags = load_tags(subject_path / "tags.csv", eda_start)

    print(f"Segmentierung {SUBJECT.upper()} / {CATEGORY} (Protokoll {version})")
    print(f"Anzahl Tags: {len(tags)}")
    print(f"Signale geladen: EDA ({eda_sr} Hz) | HR ({hr_sr} Hz) | "
          f"BVP ({bvp_sr} Hz) | ACC ({acc_sr} Hz) | TEMP ({temp_sr} Hz)")
    print("-" * 80)

    segmenter = SEGMENTERS[version][CATEGORY]
    blocks = segmenter(tags)

    print(f"{'Block':>5} | {'Label':7} | {'Name':20} | {'Min':>10} | "
          f"{'EDA':>6} | {'HR':>5} | {'BVP':>7} | {'ACC':>7} | {'TEMP':>6}")
    print("-" * 80)

    for i, (start, end, label, name) in enumerate(blocks):
        eda_block  = cut_signal(eda_data,  eda_sr,  start, end)
        hr_block   = cut_signal(hr_data,   hr_sr,   start, end)
        bvp_block  = cut_signal(bvp_data,  bvp_sr,  start, end)
        acc_block  = cut_signal(acc_data,  acc_sr,  start, end)
        temp_block = cut_signal(temp_data, temp_sr, start, end)

        zeitspanne = f"{start/60:.1f}-{end/60:.1f}"
        print(f"{i+1:>5} | {label:7} | {name:20} | {zeitspanne:>10} | "
              f"{len(eda_block):>6} | {len(hr_block):>5} | "
              f"{len(bvp_block):>7} | {len(acc_block):>7} | {len(temp_block):>6}")

    acc_mag = np.sqrt(np.sum(acc_data ** 2, axis=1))

    fig, axes = plt.subplots(5, 1, figsize=(15, 13), sharex=True)
    fig.suptitle(f"Segmentierung - {SUBJECT.upper()} / {CATEGORY} (alle Signale)", fontsize=13)

    signale = [
        (axes[0], eda_data.flatten(),  eda_sr,  "EDA (uS)"),
        (axes[1], hr_data.flatten(),   hr_sr,   "HR (bpm)"),
        (axes[2], bvp_data.flatten(),  bvp_sr,  "BVP (raw)"),
        (axes[3], acc_mag,             acc_sr,  "ACC Magnitude"),
        (axes[4], temp_data.flatten(), temp_sr, "TEMP (°C)"),
    ]

    for ax, data, sr, ylabel in signale:
        t = np.arange(len(data)) / sr / 60
        ax.plot(t, data, color="steelblue", linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=9)

        for start, end, label, name in blocks:
            ax.axvspan(start / 60, end / 60, color=FARBEN[label], alpha=0.4)
        for tag_sek in tags:
            ax.axvline(tag_sek / 60, color="black", linewidth=0.6,
                       linestyle="--", alpha=0.4)

    ymax = axes[0].get_ylim()[1]
    for start, end, label, name in blocks:
        mid = (start + end) / 2 / 60
        axes[0].text(mid, ymax * 0.95, name, ha="center", va="top",
                     fontsize=7, alpha=0.8)

    labels_used = sorted(set(b[2] for b in blocks))
    patches = [mpatches.Patch(color=FARBEN[l], label=l) for l in labels_used]
    axes[0].legend(handles=patches, loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Zeit (Minuten)")
    plt.tight_layout()
    out = f"segmentierung_{SUBJECT.upper()}_{CATEGORY}_alle_signale.png"
    plt.savefig(out, dpi=150)
    plt.show()
    print(f"\nPlot gespeichert als {out}")
