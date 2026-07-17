
import numpy as np
import pandas as pd
import neurokit2 as nk
from pathlib import Path

from load_data import load_signal, load_tags
from segment_data import is_v1, SEGMENTERS
from preprocess import filter_bvp, filter_eda

DATASET_PATH = Path(r"C:\Users\Paul\OneDrive\Dokumente\Studium\6. Semester\GADA\Daten\wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1\Wearable_Dataset")

SUBJECT  = "f01"
CATEGORY = "STRESS"


def cut(data, sample_rate, start_sek, end_sek):
    i_start = int(start_sek * sample_rate)
    i_end   = int(end_sek   * sample_rate)
    return data[i_start:i_end]


def percentile_of_derivative(signal_data, percentile):
    if len(signal_data) < 2:
        return np.nan
    return float(np.percentile(np.diff(signal_data), percentile))


def linear_slope(signal_data, sample_rate):
    if len(signal_data) < 2:
        return np.nan
    t = np.arange(len(signal_data)) / sample_rate
    return float(np.polyfit(t, signal_data, 1)[0])



def extract_hr_features(hr_block):
    return {
        "hr_mean": float(hr_block.mean()) if len(hr_block) else np.nan,
        "hr_std":  float(hr_block.std())  if len(hr_block) else np.nan,
    }



def extract_eda_features(tonic_block, phasic_block, sample_rate=4):
    feats = {
        "mean_tonic_eda":   float(tonic_block.mean()),
        "std_tonic_eda":    float(tonic_block.std()),
        "std_phasic_eda":   float(phasic_block.std()),
        "tonic_ratio_down": percentile_of_derivative(tonic_block, 5),
        "peaks_density":    np.nan,
        "mean_recoverytime": np.nan,
    }

    try:
        _, info = nk.eda_peaks(phasic_block, sampling_rate=sample_rate)
        n_peaks = len(info.get("SCR_Peaks", []))
        duration_min = len(phasic_block) / sample_rate / 60
        feats["peaks_density"] = n_peaks / duration_min if duration_min > 0 else np.nan

        rec = info.get("SCR_RecoveryTime", [])
        rec_clean = [t for t in rec if not (t is None or np.isnan(t))]
        if rec_clean:
            feats["mean_recoverytime"] = float(np.mean(rec_clean))
    except Exception:
        pass

    return feats



def extract_hrv_features(bvp_block, sample_rate=64):
    feats = {
        "max_ibi":  np.nan, "min_ibi": np.nan, "ibi_mean": np.nan,
        "rmssd":    np.nan, "LF_peak": np.nan, "LF_n":     np.nan,
    }

    try:
        info = nk.ppg_findpeaks(bvp_block, sampling_rate=sample_rate)
        peaks = info["PPG_Peaks"]
        if len(peaks) < 4:
            return feats

        ibis = np.diff(peaks) / sample_rate * 1000
        feats["max_ibi"]  = float(ibis.max())
        feats["min_ibi"]  = float(ibis.min())
        feats["ibi_mean"] = float(ibis.mean())

        hrv = nk.hrv(peaks, sampling_rate=sample_rate)
        if "HRV_RMSSD" in hrv.columns:
            feats["rmssd"] = float(hrv["HRV_RMSSD"].iloc[0])
        if "HRV_LFn" in hrv.columns:
            feats["LF_n"] = float(hrv["HRV_LFn"].iloc[0])
        for col in ("HRV_LF_Peak", "HRV_LFpeak"):
            if col in hrv.columns:
                feats["LF_peak"] = float(hrv[col].iloc[0])
                break
    except Exception:
        pass

    return feats



def extract_acc_features(acc_block):
    magnitude = np.sqrt(np.sum(acc_block ** 2, axis=1))
    return {
        "x_std":          float(acc_block[:, 0].std()),
        "y_std":          float(acc_block[:, 1].std()),
        "z_std":          float(acc_block[:, 2].std()),
        "acc_mean":       float(magnitude.mean()),
        "acc_ratio_down": percentile_of_derivative(magnitude, 5),
    }



def extract_temp_features(temp_block, sample_rate=4):
    return {
        "temp_mean":  float(temp_block.mean()),
        "temp_std":   float(temp_block.std()),
        "temp_slope": linear_slope(temp_block, sample_rate),
    }


def extract_features_for_block(eda_block, tonic_block, phasic_block,
                                bvp_block, hr_block, acc_block, temp_block,
                                eda_sr, bvp_sr, acc_sr, temp_sr):  
    feats = {}
    feats.update(extract_hr_features(hr_block))
    feats.update(extract_eda_features(tonic_block, phasic_block, eda_sr))
    feats.update(extract_hrv_features(bvp_block, bvp_sr))
    feats.update(extract_acc_features(acc_block))
    feats.update(extract_temp_features(temp_block, temp_sr))
    return feats


if __name__ == "__main__":
    version = "v1" if is_v1(SUBJECT) else "v2"
    subject_path = DATASET_PATH / CATEGORY / SUBJECT.upper()

    eda_raw,  eda_sr,  eda_start = load_signal(subject_path / "EDA.csv")
    bvp_raw,  bvp_sr,  _         = load_signal(subject_path / "BVP.csv")
    hr_data,  hr_sr,   _         = load_signal(subject_path / "HR.csv")
    acc_data, acc_sr,  _         = load_signal(subject_path / "ACC.csv")
    temp_data,temp_sr, _         = load_signal(subject_path / "TEMP.csv")
    tags = load_tags(subject_path / "tags.csv", eda_start)

    print(f"Feature Extraction {SUBJECT.upper()} / {CATEGORY} (Protokoll {version})")
    print("=" * 80)

    print("Filtere Signale ...")
    bvp_filtered = filter_bvp(bvp_raw.flatten(), bvp_sr)
    eda_filtered, eda_tonic, eda_phasic = filter_eda(eda_raw.flatten(), eda_sr)
    hr_flat   = hr_data.flatten()
    temp_flat = temp_data.flatten()

    segmenter = SEGMENTERS[version][CATEGORY]
    blocks    = segmenter(tags)

    print(f"Berechne Features fuer {len([b for b in blocks if b[2] != 'skip'])} Bloecke ...\n")
    rows = []
    for i, (start, end, label, name) in enumerate(blocks):
        if label == "skip":
            continue

        eda_b    = cut(eda_filtered, eda_sr, start, end)
        tonic_b  = cut(eda_tonic,    eda_sr, start, end)
        phasic_b = cut(eda_phasic,   eda_sr, start, end)
        bvp_b    = cut(bvp_filtered, bvp_sr, start, end)
        hr_b     = cut(hr_flat,      hr_sr,  start, end)
        acc_b    = cut(acc_data,     acc_sr, start, end)
        temp_b   = cut(temp_flat,    temp_sr, start, end)

        feats = extract_features_for_block(
            eda_b, tonic_b, phasic_b, bvp_b, hr_b, acc_b, temp_b,
            eda_sr, bvp_sr, acc_sr, temp_sr,
        )

        rows.append({
            "subject":      SUBJECT.upper(),
            "category":     CATEGORY,
            "phase_name":   name,
            "label":        label,
            "duration_sec": round(end - start, 1),
            **feats,
        })

    df = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda v: f"{v:>10.3f}")

    col_names = [f"{r['phase_name']} ({r['label']})" for r in rows]

    FEATURE_GROUPS = {
        "META":     ["duration_sec"],
        "HR  (2)":  ["hr_mean", "hr_std"],
        "EDA (6)":  ["mean_tonic_eda", "std_tonic_eda", "std_phasic_eda",
                     "tonic_ratio_down", "peaks_density", "mean_recoverytime"],
        "HRV (6)":  ["max_ibi", "min_ibi", "ibi_mean", "rmssd", "LF_peak", "LF_n"],
        "ACC (5)":  ["x_std", "y_std", "z_std", "acc_mean", "acc_ratio_down"],
        "TEMP (3)": ["temp_mean", "temp_std", "temp_slope"],
    }

    label_w   = max(len(f) for grp in FEATURE_GROUPS.values() for f in grp) + 2
    col_w     = max(12, max(len(c) for c in col_names) + 1)
    total_w   = label_w + col_w * len(col_names)

    print()
    header = " " * label_w + "".join(f"{c:>{col_w}}" for c in col_names)
    print(header)
    print("=" * len(header))

    for group_name, feature_list in FEATURE_GROUPS.items():
        print(f"\n{group_name}")
        print("-" * len(header))
        for feat in feature_list:
            row_vals = []
            for r in rows:
                v = r.get(feat)
                row_vals.append("     —    " if v is None or (isinstance(v, float) and np.isnan(v))
                                else f"{v:>{col_w-1}.3f}")
            print(f"  {feat:<{label_w-2}}" + "".join(f"{v:>{col_w}}" for v in row_vals))

    column_order = (
        ["subject", "category", "phase_name", "label", "duration_sec"]
        + [f for grp in FEATURE_GROUPS.values() for f in grp if f != "duration_sec"]
    )
    df = df[column_order]

    for col in df.select_dtypes(include="float").columns:
        df[col] = df[col].round(3)

    out = f"features_{SUBJECT.upper()}_{CATEGORY}.csv"
    df.to_csv(out, index=False, sep=";")
    print(f"\nGespeichert als {out}  ({df.shape[0]} Bloecke x {df.shape[1]} Spalten)")
